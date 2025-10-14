import warnings
import pandas as pd
from pandas.errors import PerformanceWarning
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_recall_fscore_support, roc_auc_score
from scipy.stats import pearsonr, spearmanr
import random
import json
import torch
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import confusion_matrix


# config
OUTPUT = "../work/data"
RESULTS_PATH = "../work/results"
CSV_NAME = "_etf.csv"

# column
DATETIME_COL = "datetime"
FEATURES = [
            "close",
            "log_volume", 
            "date_hour","date_day","date_month",
            "ema_ratio", "ema_spread", 
            "macd_histogram", "macd","macd_signal",
            "volatility_24h","obv",
            "sofr","vix",
            "log_return"
            ]
TARGET = "target_log_return"

# train/val/test
TRAIN_SIZE = 70
VALIDATION_SIZE = 15
TEST_SIZE = 15

# rolling window samples / frequencia
ROLLING_WINDOW_SIZE = 64
FREQUENCY='H'

# scale target
SCALE_TARGET=True

# forecast (1 en el futuro)
FORECAST_LENGTH=1

# metrics
METRICS_DIRECTION_MARGIN =  0.0001 

# torch device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} ({torch.cuda.get_device_name(0)})")

# reset_seed
SEED = 42
random.seed(SEED); 
np.random.seed(SEED); 
torch.manual_seed(SEED); 
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; 
torch.backends.cudnn.benchmark = False

# config pandas
pd.options.mode.copy_on_write = True
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0) # Auto-detect width

# config warnings
warnings.simplefilter("ignore", PerformanceWarning)

# def 
def load_df():
    # carga el dataframe de datos
    df = pd.read_csv(OUTPUT + f"/{CSV_NAME}", parse_dates=[DATETIME_COL])
    df = df.set_index(DATETIME_COL)
    df = df[FEATURES + [TARGET]]
    df = df.copy()
    return df

def calulate_train_validation_test_dates(df):
    # calcula las fechas de los datasets de TRAIN, VAL y TEST
    # explicación: hacemos el split cronológico (por dias)
    date_from = df.index[0].normalize()
    date_to = df.index[-1].normalize()+ timedelta(days=1)
    days = (date_to - date_from).days
    # train
    train_days = int((days * TRAIN_SIZE) / 100)
    train_date_from = date_from
    train_date_to = date_from + timedelta(days=train_days)
    # validation
    validation_days = int((days * VALIDATION_SIZE) / 100)
    validation_date_from = train_date_to
    validation_date_to = validation_date_from + timedelta(days=validation_days)
    # test
    test_days = int((days * TEST_SIZE) / 100)
    test_date_from = validation_date_to
    test_date_to = date_to
    # return
    return {
        "train_date_from": train_date_from,
        "train_date_to": train_date_to,
        "validation_date_from": validation_date_from,
        "validation_date_to": validation_date_to,
        "test_date_from": test_date_from,
        "test_date_to": test_date_to
    }

def scale_df(df, dates):
    # calcula  scaler para los log-returns con el data set de train
    df_train = df.loc[df.index < dates["train_date_to"]]    
    target_scaler = StandardScaler()
    target_scaler.fit(df_train[["target_log_return"]].to_numpy())
    # aplica el standardScaler sobre los log_returns de todo el dataset
    df_scaled = df.copy()
    df_scaled["target_log_return_scaled"] = target_scaler.transform(df_scaled[["target_log_return"]].to_numpy())
    df_scaled["log_return_scaled"] = target_scaler.transform(df_scaled[["log_return"]].to_numpy())
    # crea un MinMaxScaler scaler par el resto de features
    for feature in FEATURES:
        if feature != "log_return":
            feature_scaler = MinMaxScaler()    
            feature_scaler.fit(df_train[[feature]]) # fit en train 
            df_scaled[feature + "_scaled"] = feature_scaler.transform(df_scaled[[feature]])
    # return
    return (df_scaled, target_scaler)

def window_df(df_scaled):
    # calcula rolling window de todas las features
    df_scaled_window = pd.DataFrame(index=df_scaled.index)
    # windowize
    for feature in FEATURES:
        df_scaled_window[feature] = df_scaled[feature]
        for k in range(ROLLING_WINDOW_SIZE - 1, -1, -1):
            colname = f"{feature}_scaled_t-{k}"
            df_scaled_window[colname] = df_scaled[feature + "_scaled"].shift(k)
    # copy target
    df_scaled_window["target_log_return"] = df_scaled["target_log_return"]
    df_scaled_window["target_log_return_scaled"] = df_scaled["target_log_return_scaled"]
    # drop NaN
    df_scaled_window = df_scaled_window.dropna()
    # return
    return df_scaled_window

def split_df(df_scaled_window, dates):
    # particionamos el df en train/validation/test, segun fechas
    df_scaled_window_train =      df_scaled_window.loc[(dates["train_date_from"] <= df_scaled_window.index) & (df_scaled_window.index < dates["train_date_to"])]
    df_scaled_window_validation = df_scaled_window.loc[(dates["validation_date_from"] <= df_scaled_window.index) & (df_scaled_window.index < dates["validation_date_to"])]
    df_scaled_window_test =       df_scaled_window.loc[(dates["test_date_from"] <= df_scaled_window.index) & (df_scaled_window.index < dates["test_date_to"])]
    # return
    return df_scaled_window_train, df_scaled_window_validation, df_scaled_window_test

def to_tensor3d_df(df):
    # X
    column_names_reshaped = []
    for k in range(ROLLING_WINDOW_SIZE-1, -1, -1): 
        for feature in FEATURES:
            column_names_reshaped.append(feature + f"_scaled_t-{k}")
    X_flat = df[column_names_reshaped].to_numpy()
    X = X_flat.reshape(len(df), ROLLING_WINDOW_SIZE, len(FEATURES))
    # y
    y = df["target_log_return_scaled"].to_numpy()
    # retorna tensores
    return ( torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

def plot_train_validation_loss_by_epoch(train_loss_history, validation_loss_history, units = "Huber"):
    # plots train validataion loss
    epochs = range(1, len(train_loss_history) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss_history, label="Train Loss", marker='o')
    plt.plot(epochs, validation_loss_history,   label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (" + units + ")")
    plt.xticks(epochs)              
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # segundo chart
    fig, ax1 = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_loss_history) + 1)
    # Primary y-axis (left) → train loss
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(epochs, train_loss_history, color=color, marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    # Secondary y-axis (right) → validation loss
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Validation Loss', color=color)
    ax2.plot(epochs, validation_loss_history, color=color, marker='o', label='Val Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    # plot
    fig.tight_layout()
    plt.show()

def add_predictions_to_df(df, y_hat, target_scaler, close_base, margin=METRICS_DIRECTION_MARGIN):
    # predict_log_return
    df["predict_log_return_scaled"] = y_hat
    df["predict_log_return"] = target_scaler.inverse_transform(df[["predict_log_return_scaled"]])    
    # rebuild predict_close
    df["predict_close_accumulated"] = close_base * (df["predict_log_return"].cumsum().apply(lambda x: np.exp(x)))
    df["predict_close"] = df["close"].shift(1) * np.exp(df["predict_log_return"])
    # direction
    df["target_direction"] = [predict_direction(p, margin) for p in df["target_log_return"]]
    df["predict_direction"] = [predict_direction(p, margin) for p in df["predict_log_return"]]
    # errors
    df["error_log_return"] = df["target_log_return"] - df["predict_log_return"]
    df["error_log_return_absolute"] = abs(df["target_log_return"] - df["predict_log_return"])
    df["error_log_return_square"] = (df["target_log_return"] - df["predict_log_return"]) ** 2

def predict_direction(pred, margin=METRICS_DIRECTION_MARGIN):
    # conviente la magnitud en categoria 1,0,-1 aplicando un margen
    if pred > margin:
        return 1   
    elif pred < -margin:
        return -1  
    else:
        return 0   

def huber(y_true, y_pred, delta=1.0):
    # metrica de hubger
    err = y_true - y_pred
    abs_err = np.abs(err)
    return np.mean(np.where(abs_err <= delta, 0.5 * err**2, delta * (abs_err - 0.5 * delta)))    

def simulated_sharpe_ratio(real_returns, predicted_returns, risk_free=0.0, margin=METRICS_DIRECTION_MARGIN):
    # funcion para simular el ratio de sharpe
    real_returns = np.array(real_returns)
    predicted_returns = np.array(predicted_returns)
    # Trading signals with margin
    signals = np.where(predicted_returns > margin, 1, np.where(predicted_returns < -margin, -1, 0))
    # Strategy returns
    strategy_returns = real_returns * signals
    # Excess returns
    excess_returns = strategy_returns - risk_free
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)  # sample std
    # return
    return mean_excess / std_excess if std_excess != 0 else np.nan

def evaluate_metrics(y, y_predict, target_direction, predict_direction):
    result = {}
    # mse
    result["mse"] = mean_squared_error(y, y_predict)
    # rmse
    result["rmse"] = np.sqrt(result["mse"])
    # mae
    result["mae"] = mean_absolute_error(y, y_predict)
    # huber
    result["huber"] = huber(y, y_predict)
    # r2
    result["r2"] = r2_score(y, y_predict)
    # pearson_r
    result["pearson_r"], _ = pearsonr(y.squeeze(), y_predict.squeeze()) if len(y) > 1 else (np.nan, np.nan)
    # spearman_rho
    result["spearman_rho"], _ = spearmanr(y, y_predict) if len(y) > 1 else (np.nan, np.nan)
    # direction
    num_matches = np.sum(target_direction == predict_direction)
    total_elements = target_direction.size
    result["direction"] = (num_matches / total_elements) * 100
    # sharpe
    result["sharpe"] = simulated_sharpe_ratio(y, y_predict)    
    # return
    return result

def evaluate_metrics_df(df_scaled_window_test): 
    # evalua las metricas de los tre conjuntos
    return evaluate_metrics(df_scaled_window_test["target_log_return"], 
                            df_scaled_window_test["predict_log_return"],
                            df_scaled_window_test["target_direction"],
                            df_scaled_window_test["predict_direction"]
                            )

def plot_df(df_scaled_window_train, df_scaled_window_validation, df_scaled_window_test, plot_train_predict: bool = False, model_name: str = ""): 
    # plot
    df_scaled_window_train_plot = df_scaled_window_train.copy()
    df_scaled_window_train_plot = df_scaled_window_train_plot.reset_index()

    df_scaled_window_validation_plot = df_scaled_window_validation.copy()
    df_scaled_window_validation_plot = df_scaled_window_validation_plot.reset_index()

    df_scaled_window_test_plot = df_scaled_window_test.copy()
    df_scaled_window_test_plot = df_scaled_window_test_plot.reset_index()

    # plot log_return
    plt.figure(figsize=(14,6))
    plt.vlines(df_scaled_window_train_plot[DATETIME_COL], ymin=0, ymax=df_scaled_window_train_plot["target_log_return"], label="Train", color="blue")
    if plot_train_predict:
        plt.vlines(df_scaled_window_train_plot[DATETIME_COL], ymin=0, ymax=df_scaled_window_train_plot["predict_log_return"], label="Forecast", color="black")
    
    plt.axvline(df_scaled_window_validation_plot.iloc[0][DATETIME_COL], color='gray', linestyle='--')
    
    plt.vlines(df_scaled_window_validation_plot[DATETIME_COL], ymin=0, ymax=df_scaled_window_validation_plot["target_log_return"], label="Validation", color="red")
    plt.vlines(df_scaled_window_validation_plot[DATETIME_COL], ymin=0, ymax=df_scaled_window_validation_plot["predict_log_return"], color="black")

    plt.axvline(df_scaled_window_test_plot.iloc[0][DATETIME_COL], color='gray', linestyle='--')    

    plt.vlines(df_scaled_window_test_plot[DATETIME_COL], ymin=0, ymax=df_scaled_window_test_plot["target_log_return"], label="Test", color="green")
    plt.vlines(df_scaled_window_test_plot[DATETIME_COL], ymin=0, ymax=df_scaled_window_test_plot["predict_log_return"], label="Forecast", color="black")

    plt.xlabel("Date")
    plt.ylabel("Log return")
    plt.legend()
    plt.show()

    # plot log_return test only
    plt.figure(figsize=(14,6))
    plt.vlines(df_scaled_window_test_plot[DATETIME_COL], ymin=0, ymax=df_scaled_window_test_plot["target_log_return"], label="Test", color="green")
    plt.vlines(df_scaled_window_test_plot[DATETIME_COL], ymin=0, ymax=df_scaled_window_test_plot["predict_log_return"], label="Test forecast", color="black")
    plt.xlabel("Date")
    plt.ylabel("Log return")
    plt.legend()
    plt.show()

    # plot prices train / validation / test / test forecast
    plt.figure(figsize=(14,6))
    plt.plot(df_scaled_window_train_plot[DATETIME_COL], df_scaled_window_train_plot["close"], label="Train", color="blue")
    if plot_train_predict:
        plt.plot(df_scaled_window_train_plot[DATETIME_COL], df_scaled_window_train_plot["predict_close"], label="Forecast", color="black")

    plt.axvline(df_scaled_window_validation_plot.iloc[0][DATETIME_COL], color='gray', linestyle='--')
    plt.plot(df_scaled_window_validation_plot[DATETIME_COL], df_scaled_window_validation_plot["close"], label="Validation", color="red")
    plt.plot(df_scaled_window_validation_plot[DATETIME_COL], df_scaled_window_validation_plot["predict_close"], color="black")

    plt.axvline(df_scaled_window_test_plot.iloc[0][DATETIME_COL], color='gray', linestyle='--')
    plt.plot(df_scaled_window_test_plot[DATETIME_COL], df_scaled_window_test_plot["close"], label="Test", color="green")
    plt.plot(df_scaled_window_test_plot[DATETIME_COL], df_scaled_window_test_plot["predict_close"], label="Forecast", color="black")

    plt.xlabel("Date")
    plt.ylabel("Precio ($)")
    plt.legend()
    plt.show()

    # plot log_return test only
    plt.figure(figsize=(14,6))
    plt.plot(df_scaled_window_test_plot[DATETIME_COL], df_scaled_window_test_plot["close"], label="Test", color="green")
    plt.plot(df_scaled_window_test_plot[DATETIME_COL], df_scaled_window_test_plot["predict_close"], label="Forecast", color="black")
    plt.xlabel("Date")
    plt.ylabel("Precio ($)")
    plt.legend()
    plt.show()

    # plot prices acumulated
    plt.figure(figsize=(14,6))
    plt.axvline(df_scaled_window_validation_plot.iloc[0][DATETIME_COL], color='gray', linestyle='--')
    plt.axvline(df_scaled_window_test_plot.iloc[0][DATETIME_COL], color='gray', linestyle='--')

    plt.plot(df_scaled_window_train_plot[DATETIME_COL], df_scaled_window_train_plot["close"], label="Train", color="blue")
    if plot_train_predict:
        plt.plot(df_scaled_window_train_plot[DATETIME_COL], df_scaled_window_train_plot["predict_close_accumulated"], label="Forecast acumulado", color="black")
    
    plt.plot(df_scaled_window_validation_plot[DATETIME_COL], df_scaled_window_validation_plot["close"], label="Validation", color="red")
    plt.plot(df_scaled_window_validation_plot[DATETIME_COL], df_scaled_window_validation_plot["predict_close_accumulated"], color="black")
    
    plt.plot(df_scaled_window_test_plot[DATETIME_COL], df_scaled_window_test_plot["close"], label="Test", color="green")
    plt.plot(df_scaled_window_test_plot[DATETIME_COL], df_scaled_window_test_plot["predict_close_accumulated"], label="Forecast acumulado", color="black")

    plt.xlabel("Date")
    plt.ylabel("Precio ($)")
    plt.legend()
    plt.show()

def plot_confusion_matrix(df, true_col="target_direction", pred_col="predict_direction"):
    # Map numeric to text labels
    label_map = {-1: "Down", 0: "Neutral", 1: "Up"}
    labels = [-1, 0, 1]
    label_names = [label_map[l] for l in labels]
    y_true = df[true_col]
    y_pred = df[pred_col]
    # Raw confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Percentages (row-normalized)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, None] * 100
    # Convert to DataFrames
    cm_df_counts = pd.DataFrame(cm,
                                index=[f"True {l}" for l in label_names],
                                columns=[f"Pred {l}" for l in label_names])
    cm_df_percent = pd.DataFrame(cm_percent.round(2),
                                 index=[f"True {l}" for l in label_names],
                                 columns=[f"Pred {l}" for l in label_names])
    # Add % sign for display
    cm_df_percent_str = cm_df_percent.astype(str) + "%"
    # print
    print("Confusion Matrix (Counts):")
    print(cm_df_counts, "\n")
    print("Confusion Matrix (Percentages):")
    print(cm_df_percent_str)
    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    sns.heatmap(cm_df_counts, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    # Use the string version (with %)
    sns.heatmap(cm_df_percent, annot=cm_df_percent_str.values, fmt="", cmap="Blues",
                ax=axes[1], cbar_kws={'label': 'Percentage'})
    axes[1].set_title("Confusion Matrix (Percentages)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()
    # return
    return cm_df_counts, cm_df_percent

def save_result(name:str, metrics:dict, df_scaled_window_train, df_scaled_window_validation, df_scaled_window_test):
    # guarda resultados
    with open(RESULTS_PATH + "/metrics_" + name + ".json", "w") as f:
        json.dump(metrics, f)
    df_scaled_window_train.to_pickle(RESULTS_PATH + "/data_" + name + "_train.pkl")
    df_scaled_window_validation.to_pickle(RESULTS_PATH + "/data_" + name + "_validation.pkl")
    df_scaled_window_test.to_pickle(RESULTS_PATH + "/data_" + name + "_test.pkl")
