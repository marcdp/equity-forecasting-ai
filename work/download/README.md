# External Data

This directory contains external daily time-series data used as auxiliary features in the equity forecasting experiments.

These files are inputs to the data-preparation pipeline and are combined with the hourly synthetic equity dataset generated in `src/prepare.ipynb`.

## Files

### `SOFR.csv`

Contains daily observations of the **Secured Overnight Financing Rate (SOFR)**.

Format:

```text
date,sofr
2021-01-04,0.10
2021-01-05,0.11
...
```

The dataset contains:

* `date` — calendar date of the observation
* `sofr` — SOFR value for that date

SOFR is used as a macroeconomic / interest-rate feature in the forecasting dataset.

### `VIX.csv`

Contains daily observations of the **CBOE Volatility Index (VIX)**.

Format:

```text
date,vix
2021-01-04,26.97
2021-01-05,25.34
...
```

The dataset contains:

* `date` — calendar date of the observation
* `vix` — VIX value for that date

VIX is used as a market-volatility feature in the forecasting dataset.

## Temporal Alignment

The primary equity dataset uses hourly observations, whereas SOFR and VIX are stored at daily frequency.

During dataset preparation:

1. the `date` field of each external dataset is converted to a calendar date;
2. the hourly equity observations are assigned their corresponding calendar date;
3. the external data is joined to the equity dataset using that date.

As a consequence, hourly observations belonging to the same calendar date receive the same daily SOFR and VIX values.

This reflects the methodology used in the original Master's thesis experiments.

### Information-availability caveat

A calendar-date join does not by itself guarantee that a daily value would have been available at every hour represented by that date.

For a strict real-time or production forecasting system, the publication or observation time of each external variable should be considered explicitly. Depending on the source and definition of the daily observation, an appropriate lag may be required to ensure that only information available at prediction time is used.

The experiments in this repository reproduce the temporal alignment used in the original research and should therefore be interpreted within that methodological scope.

## Data Provenance

The original repository did not preserve sufficient metadata to establish the exact download source and retrieval timestamp for these two CSV files.

Before using or redistributing these files outside the context of reproducing this research, the original provider, licensing terms, and redistribution conditions should be verified.

Recommended provenance information to preserve for future datasets includes:

* provider;
* exact series or index identifier;
* source URL or API endpoint;
* retrieval date;
* observation frequency;
* publication timing;
* date range;
* transformation or missing-value handling;
* licensing and redistribution terms.

## Data Rights

These files contain third-party financial and economic data and are not covered automatically by any software license applied to the source code in this repository.

Users are responsible for verifying the applicable terms of the original data providers before redistributing or using the data for purposes beyond research reproduction.
