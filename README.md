# Decoupled Local/Global Time-Series Representations on FreshRetailNet-50K

This repository adapts the algorithmic structure from *Decoupling Local and Global Representations of Time Series* to FreshRetailNet-50K.

The model keeps the paper's core data structure:

- `X in R^{d x T}`: one store-product time series.
- non-overlapping windows `X_t in R^{d x delta}`.
- local latent sequence `Z_l = [z_0, ..., z_t]`, one vector per window.
- global latent vector `z_g`, one vector per store-product series.
- mask channel for censored or unavailable observations.
- GP prior over local latents and standard normal prior over global latents.
- counterfactual regularization by decoding `(Z_l, z_g*)`, re-encoding the generated sample globally, and penalizing likelihood preference for the original `z_g`.

FreshRetailNet mapping:

- sample identity: `store_id, product_id`
- hourly sales: `hours_sale`
- stockout/censoring mask: `hours_stock_status == 1`
- covariates: discount, holiday/activity flags, precipitation, temperature, humidity, wind, hour-of-day and day-of-week sin/cos features
- subgroup evaluation target: `city_id` by default
- forecasting target: held-out final `forecast_windows` days

## Setup

```bash
uv sync
```

## Prepare Data

Download the Hugging Face dataset to local parquet files:

```bash
uv run python scripts/prepare_freshretailnet.py --out data/freshretailnet
```

You can also place `train.parquet` and `eval.parquet` under `data/freshretailnet`.

For a quick smoke experiment, set `max_train_series` and `max_eval_series` in `configs/freshretailnet.json` to small values such as `256` and `64`.

## Train

```bash
uv run decoupled-ts train --config configs/freshretailnet.json
```

The best checkpoint is written to `runs/freshretailnet_glr/best.pt`.

## Evaluate Paper-Style Experiments

```bash
uv run decoupled-ts evaluate --checkpoint runs/freshretailnet_glr/best.pt
```

This runs FreshRetailNet-adapted versions of the paper's three representation evaluations:

- downstream prediction from learned local/global representations
- subgroup identification from global representations
- multi-window forecasting using GP conditional local latent prediction

Metrics are written to `runs/freshretailnet_glr/metrics.json`.
