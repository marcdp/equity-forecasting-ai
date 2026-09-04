# Results

This directory contains the consolidated evaluation results from the final chronological test partition of the equity forecasting experiments.

The results are derived from the experiments implemented in the repository notebooks and summarize the performance of simple baselines and the three evaluated deep-learning architectures.

## Files

### `metrics.csv`

Machine-readable summary of the final test-set results.

It contains the following fields:

* `model` — model or baseline identifier
* `category` — baseline or deep-learning model
* `evaluation_split` — dataset partition used for evaluation
* `mse` — Mean Squared Error
* `rmse` — Root Mean Squared Error
* `mae` — Mean Absolute Error
* `r2` — coefficient of determination
* `directional_accuracy_pct` — percentage of predictions with the correct directional classification
* `simulated_sharpe` — experimental trading-oriented Sharpe-like metric

## Evaluated Models

The consolidated file currently includes:

### Baselines

* random-distribution baseline
* mean-return baseline
* zero-return baseline

### Deep-learning models

* LSTM
* N-BEATS
* TSFEDL

The repository also contains a `model_baseline_random_uniform.ipynb` experiment.

However, a reliable final consolidated test-set row for the random-uniform baseline was not available when this artifact was created. Its metrics are therefore intentionally excluded rather than reconstructed or inferred.

## Evaluation Context

All rows in `metrics.csv` correspond to the final chronological test partition used in the research.

The dataset is split chronologically into:

```text
Training:   70%
Validation: 15%
Test:       15%
```

The test partition remains temporally later than the training and validation periods.

This is important for financial time-series evaluation because random splitting could introduce look-ahead leakage.

## Directional Accuracy

Directional accuracy is calculated by converting predicted and actual log returns into three states:

```text
 1  positive movement
 0  neutral movement
-1  negative movement
```

The classification threshold used by the experiment is:

```text
±0.0001
```

Values inside this margin are treated as neutral.

The reported directional accuracy therefore depends on this experiment-specific threshold and should not be interpreted as a generic binary up/down classification metric.

## Simulated Sharpe Metric

The `simulated_sharpe` column reports an experimental trading-oriented metric derived from model-generated long, short, and neutral signals.

Predictions are converted into signals as follows:

```text
predicted return >  0.0001  -> long
predicted return < -0.0001  -> short
otherwise                   -> neutral
```

The reported value is calculated from the mean excess strategy return divided by the standard deviation of strategy returns.

This value is:

* not annualized;
* based on a frictionless simulation;
* not adjusted for transaction costs;
* not adjusted for bid/ask spreads;
* not adjusted for slippage;
* not adjusted for market impact;
* not adjusted for execution latency;
* not adjusted for short-selling costs.

It should therefore be interpreted only as a comparative experimental metric and not as evidence of a production-ready or profitable trading strategy.

## Interpretation

The results show that model performance depends strongly on the evaluation objective.

LSTM produced the strongest regression performance among the evaluated deep-learning models, but its improvement over simple zero and mean baselines was very small.

N-BEATS produced weaker regression metrics but achieved the strongest directional accuracy and the highest simulated trading-oriented metric.

This illustrates one of the central findings of the research:

> **Forecast accuracy, directional accuracy, and simulated trading utility are related but distinct objectives.**

A model that minimizes numerical forecasting error is not necessarily the model that produces the strongest directional signal.

## Reproducibility

The values in this directory are intended to provide a stable, inspectable summary of the research results without requiring a reviewer to execute the full notebook workflow.

The underlying experiment implementations and evaluation logic remain available in the repository under:

```text
src/
```

In particular:

```text
src/metrics.ipynb
src/model_baseline_*.ipynb
src/model_lstm.ipynb
src/model_nbeats.ipynb
src/model_tsfedl.ipynb
```

For the complete methodology, limitations, and interpretation of the experiment, see the main repository [`README.md`](../README.md) and the Master's thesis under [`thesis/`](../thesis/).
