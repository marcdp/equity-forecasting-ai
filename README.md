# Equity Forecasting with Deep Learning

Applied AI/ML research comparing deep-learning architectures for **hourly equity time-series forecasting**, with evaluation across regression accuracy, directional prediction, and simulated trading performance.

This repository contains the research code developed for my Master's Thesis in Applied Artificial Intelligence at **Universidad Europea de Madrid (2025)**.

The study evaluates whether deep-learning models can provide useful predictive information for short-horizon financial forecasting when compared with simple statistical baselines.

**Models:** LSTM · N-BEATS · TSFEDL
**Frequency:** Hourly
**Period:** 2021–2024
**Forecast horizon:** One period ahead
**Assets:** AAPL · MSFT · NVDA · IBM · GOOG

[Read the Master's Thesis](thesis/masters-thesis.pdf) ·
[View the Thesis Presentation](thesis/masters-thesis-presentation.pdf)
---

## Research Context

Financial time-series forecasting is difficult because returns are noisy, non-stationary, and characterized by a very low signal-to-noise ratio.

This research investigates whether deep-learning architectures can extract useful information from a multivariate hourly financial dataset and whether improvements in traditional regression metrics correspond to improvements in **directional prediction** or **simulated trading performance**.

The original thesis is titled:

> **Mejora de la predicción de cotizaciones de activos de renta variable mediante inteligencia artificial**

English translation:

> **Enhancing Equity Price Prediction through Artificial Intelligence and Algorithmic Trading**

The project was developed as the final research project for the **Master's Degree in Applied Artificial Intelligence** at Universidad Europea de Madrid.

---

## Research Question

The experiment focuses on three related questions:

1. Can deep-learning models improve one-step-ahead financial return forecasts relative to simple baselines?
2. Do lower numerical forecast errors translate into better prediction of market direction?
3. Do models that perform better on regression metrics also produce stronger results under a simple simulated trading strategy?

This distinction is important because a model that minimizes numerical forecasting error is not necessarily the model that produces the most useful directional signal.

---

## Dataset

The experiment uses a synthetic equity basket constructed from five publicly traded companies:

| Ticker | Company   |
| ------ | --------- |
| AAPL   | Apple     |
| MSFT   | Microsoft |
| NVDA   | NVIDIA    |
| IBM    | IBM       |
| GOOG   | Alphabet  |

The research pipeline is configured to build a **market-cap-weighted synthetic ETF** from these assets.

### Market data

Equity market data is retrieved from the **Polygon.io Aggregates API** with the following configuration:

* **Period:** January 2021 through December 2024
* **Aggregation:** 1-hour bars
* **Adjusted prices:** enabled
* **Timestamp representation:** UTC
* **Sort order:** chronological
* **Assets:** AAPL, MSFT, NVDA, IBM, GOOG

The preparation notebook downloads and caches the underlying market data before constructing the research dataset.

Raw Polygon market data is not committed to this repository. Reproducing the complete dataset therefore requires access to the Polygon.io API.

### External variables

The model dataset also incorporates:

* **SOFR** — Secured Overnight Financing Rate
* **VIX** — volatility-market information

The repository currently includes supporting SOFR and VIX CSV files under:

```text
work/download/
├── README.md
├── SOFR.csv
└── VIX.csv
```

The original provider and retrieval metadata for these two committed files
was not preserved in the original research environment.

The temporal-alignment methodology and its information-availability
limitations are documented in [`work/download/README.md`](work/download/README.md).

---

## Feature Engineering

The forecasting dataset contains **15 input features**.

### Market and return features

* `close`
* `log_volume`
* `log_return`

### Calendar features

* `date_hour`
* `date_day`
* `date_month`

### Trend and momentum features

* `ema_ratio`
* `ema_spread`
* `macd`
* `macd_signal`
* `macd_histogram`

### Volatility and volume features

* `volatility_24h`
* `obv`

### External market features

* `sofr`
* `vix`

The forecasting target is:

```text
target_log_return
```

The experiment is configured for a **one-period-ahead forecast**.

---

## Experimental Design

Financial time series require particular care when splitting and transforming data because randomly mixing observations across time can introduce look-ahead leakage.

This project therefore uses a **chronological split**:

| Partition  | Share |
| ---------- | ----: |
| Training   |   70% |
| Validation |   15% |
| Test       |   15% |

The final test partition remains temporally later than the training and validation partitions.

### Rolling input window

Each prediction is constructed using a rolling window of:

```text
64 hourly observations
```

across the 15 model features.

The resulting model input therefore represents a sequence of historical observations rather than an isolated point in time.

### Scaling and leakage prevention

Scaling parameters are estimated **exclusively from the training partition**.

The implementation uses:

* `StandardScaler` for the target/log-return representation
* `MinMaxScaler` for the remaining features

The fitted training scalers are then applied to validation and test observations.

This prevents information from later validation or test periods from influencing the scaling parameters used during training.

### Reproducibility

The experiment uses a fixed random seed:

```text
SEED = 42
```

Random seeds are applied to NumPy, Python's random module, and PyTorch. The PyTorch configuration also enables deterministic cuDNN behavior where applicable.

---

## Models

The research compares three deep-learning approaches.

### LSTM

Long Short-Term Memory networks are recurrent neural networks designed to model temporal dependencies and are commonly applied to sequential time-series problems.

The LSTM experiment uses the 64-step multivariate history to estimate the next target log return.

### N-BEATS

N-BEATS is a deep neural forecasting architecture based on stacks of fully connected residual blocks.

It provides a substantially different modelling approach from recurrent architectures and is included to compare forecasting behavior across architecture families.

### TSFEDL

The TSFEDL experiment explores a hybrid deep-learning architecture combining convolutional and recurrent components for financial time-series modelling.

---

## Baselines

Deep-learning models should not be evaluated in isolation.

The repository includes several baseline experiments:

* zero-return prediction
* mean-return prediction
* random-distribution prediction
* random-uniform prediction

These baselines provide an important reference point because financial returns are commonly concentrated close to zero. A sophisticated model can therefore produce apparently good regression metrics without learning a signal that is materially better than a trivial forecast.

---

## Evaluation Metrics

The models are evaluated using several complementary perspectives.

### Regression metrics

* Mean Squared Error — **MSE**
* Root Mean Squared Error — **RMSE**
* Mean Absolute Error — **MAE**
* Huber loss
* **R²**

### Association metrics

* Pearson correlation
* Spearman correlation

### Directional accuracy

Predicted and actual log returns are transformed into three directional states:

```text
 1  positive movement
 0  neutral movement
-1  negative movement
```

A directional margin of:

```text
0.0001
```

is used to distinguish meaningful positive and negative movements from values treated as neutral.

### Simulated Sharpe metric

The project also evaluates a simple trading-oriented metric.

Predicted returns are translated into signals:

```text
predicted return > margin   -> long
predicted return < -margin  -> short
otherwise                   -> neutral
```

The simulated strategy return is then calculated from the realized return and the generated signal.

The reported value is computed as:

```text
mean excess strategy return
---------------------------
standard deviation of strategy returns
```

This metric should be interpreted as an **experimental simulated Sharpe-like measure**, not as evidence of a deployable trading strategy.

The current simulation does **not** model:

* transaction costs
* bid/ask spreads
* slippage
* market impact
* execution latency
* short-selling costs
* turnover constraints

The value is also not annualized in the implementation.

---

## Test-Set Results

The following table summarizes the principal results from the final chronological test partition.

| Model                        |         RMSE |          MAE |           R² | Directional Accuracy | Simulated Sharpe |
| ---------------------------- | -----------: | -----------: | -----------: | -------------------: | ---------------: |
| Random distribution baseline |     0.002873 |     0.001934 |    -0.583155 |               40.75% |          -0.0148 |
| Mean baseline                |     0.002284 |     0.001091 |    -0.000008 |               56.75% |              N/A |
| Zero baseline                |     0.002284 | **0.001071** |    -0.000395 |               56.75% |              N/A |
| **LSTM**                     | **0.002282** |     0.001083 | **0.001273** |               60.66% |           0.0263 |
| **N-BEATS**                  |     0.002387 |     0.001189 |    -0.092777 |           **67.47%** |       **0.0546** |
| **TSFEDL**                   |     0.002283 |     0.001095 |     0.000780 |               60.28% |           0.0303 |

The repository also contains a random-uniform baseline notebook. The table above reports the baseline rows currently available in the consolidated metrics output together with the three deep-learning models.

---

## Key Findings

The results do not support a simple conclusion that one deep-learning architecture is universally superior.

### 1. Regression improvements over trivial baselines are small

LSTM produced the lowest RMSE among the evaluated deep-learning models:

```text
LSTM RMSE         0.002282
Zero baseline     0.002284
Mean baseline     0.002284
```

The difference is extremely small.

More importantly, the zero-return baseline obtained a slightly lower MAE than the deep-learning models:

```text
Zero baseline MAE  0.001071
LSTM MAE           0.001083
TSFEDL MAE         0.001095
```

This demonstrates why simple baselines are essential when evaluating financial forecasting models.

Because hourly returns are frequently close to zero, predicting values close to zero can be difficult to outperform using magnitude-based error metrics.

### 2. Directional performance tells a different story

The relationship changes when predictions are evaluated according to market direction.

The deterministic zero and mean baselines achieved approximately:

```text
56.75%
```

directional accuracy.

The deep-learning models achieved:

```text
LSTM      60.66%
N-BEATS   67.47%
TSFEDL    60.28%
```

N-BEATS therefore produced the strongest directional result despite having **worse RMSE, MAE, and R² than LSTM**.

### 3. The best regression model is not the best directional model

This is one of the central observations of the experiment.

LSTM provided the strongest regression performance among the deep-learning models, while N-BEATS produced the strongest directional accuracy and simulated trading-oriented metric.

N-BEATS achieved:

```text
Directional accuracy: 67.47%
Simulated Sharpe:       0.0546
R²:                    -0.0928
```

The negative R² and relatively weak regression error make it inappropriate to describe N-BEATS as universally superior.

Instead, its result suggests that a model can contain useful information about **direction** even when its estimates of **return magnitude** remain weak.

### 4. There is no universal winner

The experiments illustrate the importance of selecting evaluation metrics according to the intended objective.

If the objective is minimizing regression error, LSTM performs best among the deep-learning models.

If directional classification is more important, N-BEATS performs substantially better.

If the objective is simply minimizing absolute return error, even a zero-return baseline remains highly competitive.

The result is therefore more nuanced than "`AI beats the baseline`":

> **Forecast accuracy, directional accuracy, and simulated trading utility are related but distinct objectives.**

---

## Interpretation

The low and sometimes negative R² values are an important part of the result rather than something to hide.

Short-horizon financial returns are highly noisy. The experiments show that sophisticated neural models can struggle to improve magnitude-based predictions over simple near-zero forecasts.

At the same time, some models appear to capture directional information that is less visible in traditional regression metrics.

The research therefore supports a cautious interpretation:

* deep learning did not uniformly outperform trivial baselines;
* model ranking changes depending on the evaluation objective;
* directional information may be easier to exploit than precise return magnitude;
* simulated trading metrics must not be interpreted as real-world profitability without realistic execution modelling;
* strong baselines are essential when evaluating financial ML systems.

---

## Repository Structure

```text
equity-forecasting-ai/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
│
├── src/
│   ├── prepare.ipynb
│   ├── eda.ipynb
│   ├── model_baseline_mean.ipynb
│   ├── model_baseline_random_distribution.ipynb
│   ├── model_baseline_random_uniform.ipynb
│   ├── model_baseline_zero.ipynb
│   ├── model_lstm.ipynb
│   ├── model_nbeats.ipynb
│   ├── model_tsfedl.ipynb
│   ├── metrics.ipynb
│   ├── chart_evolucion_modelos.ipynb
│   └── utils.py
│
├── work/
│   └── download/
│       ├── README.md
│       ├── SOFR.csv
│       └── VIX.csv
│
└── thesis/
    ├── README.md
    └── masters-thesis.pdf
```

The repository reflects the original research workflow. A future cleanup may separate reusable Python modules from notebooks more explicitly, but the research notebooks are preserved to maintain traceability to the thesis experiments.

---

## Reproducing the Research

### 1. Clone the repository

```bash
git clone https://github.com/marcdp/equity-forecasting-ai.git
cd equity-forecasting-ai
```

### 2. Create a Python environment

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```


### 4. Configure Polygon.io access

`prepare.ipynb` retrieves equity data from Polygon.io.

The original research environment obtains the Polygon API credential using `dprojectstools.secrets.SecretsManager`.

If running the project outside that environment, adapt the credential-loading section of `prepare.ipynb` to obtain your Polygon API key through your preferred secret-management mechanism.

API keys must never be committed to the repository.

### 5. Execute the notebooks

The intended research workflow is approximately:

```text
1. src/prepare.ipynb
2. src/eda.ipynb

3. Baseline experiments
   ├── src/model_baseline_zero.ipynb
   ├── src/model_baseline_mean.ipynb
   ├── src/model_baseline_random_uniform.ipynb
   └── src/model_baseline_random_distribution.ipynb

4. Deep-learning experiments
   ├── src/model_lstm.ipynb
   ├── src/model_nbeats.ipynb
   └── src/model_tsfedl.ipynb

5. src/metrics.ipynb

6. src/chart_evolucion_modelos.ipynb
```

The preparation stage generates the dataset consumed by the model notebooks.

---

## Reproducibility Notes

Several decisions in the implementation are specifically intended to reduce common sources of error in time-series ML experiments:

* chronological rather than random train/test splitting;
* preprocessing scalers fitted only on training observations;
* fixed random seeds;
* deterministic PyTorch/cuDNN settings where supported;
* explicit simple baselines;
* a separate untouched final test period;
* common evaluation utilities across model families.


The repository remains a research codebase rather than a production ML
system. Exact dependency version pinning, automated tests, and stronger
environment reproducibility remain areas for improvement.

The external-data workflow and temporal-alignment assumptions are documented
in `work/download/README.md`; however, the exact original source metadata for
the committed SOFR and VIX files was not preserved in the original research
environment.

---

## Limitations

The results should be interpreted within the scope of the experiment.

### Dataset scope

The synthetic basket contains only five large technology-oriented US equities.

It is not representative of the entire equity market, and the use of a fixed preselected universe can introduce selection or survivorship effects.

### Time period

The study covers 2021–2024.

Financial relationships change across market regimes, and results from this period should not be assumed to generalize to other periods.

### Validation design

The experiment uses a single chronological 70/15/15 train-validation-test split.

A broader research design could additionally use repeated walk-forward evaluation across multiple market regimes.

### Forecast horizon

The study focuses on one-step-ahead hourly forecasting.

The conclusions do not automatically extend to daily, weekly, or higher-frequency prediction.

### External variables

SOFR and VIX are included as model features and are joined to hourly
observations by calendar date.

This temporal alignment is documented in `work/download/README.md`.
However, the original research did not preserve sufficient publication-time
metadata to establish whether every same-day external observation would have
been available at each hourly prediction timestamp.

For a strict real-time forecasting system, these variables should be aligned
according to their actual information-availability time, potentially using an
explicit lag.

### Trading simulation

The trading-oriented evaluation is deliberately simple.

It does not model realistic market frictions and therefore cannot be interpreted as evidence of an executable or profitable investment strategy.

### Predictive strength

R² values remain close to zero or negative across the evaluated models.

This is consistent with the difficulty of predicting short-horizon financial returns and reinforces the need for cautious interpretation of directional and simulated trading results.

---

## Master's Thesis

The complete thesis is included in this repository:

**[Download / view the Master's Thesis](thesis/masters-thesis.pdf)**

**Author:** Marc Delos
**Program:** Master's Degree in Applied Artificial Intelligence
**Institution:** Universidad Europea de Madrid
**Year:** 2025

**Original title:**

> Mejora de la predicción de cotizaciones de activos de renta variable mediante inteligencia artificial

**English title:**

> Enhancing Equity Price Prediction through Artificial Intelligence and Algorithmic Trading

---

## Citation

If this repository or the associated research is referenced academically, please cite the thesis and this repository.

Citation metadata is also provided in:

```text
CITATION.cff
```

---

## License and Data Rights

Source code in this repository is licensed under the
[MIT License](LICENSE), unless otherwise stated.

The software license does not automatically apply to the Master's thesis
or to third-party financial and economic data.

- [`LICENSE`](LICENSE) — source-code license
- [`thesis/README.md`](thesis/README.md) — thesis copyright and reuse information
- [`work/download/README.md`](work/download/README.md) — external-data provenance and rights

The Polygon.io market data used to reproduce the primary equity dataset
remains subject to Polygon.io's applicable terms.

The committed SOFR and VIX files contain third-party data. Their original
provider and redistribution terms were not preserved in the original
research repository and should therefore be verified before redistribution.
---

## Disclaimer

This repository documents academic research in artificial intelligence, financial time-series forecasting, and simulated algorithmic-trading evaluation.

It is **not investment advice**, and the experimental results should not be interpreted as evidence of a production-ready or profitable trading strategy.
