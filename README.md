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

Preprocessed tensors are cached under `data/freshretailnet/cache` when `use_cache` is enabled, so the slow row-to-series conversion is skipped on later runs with the same config.

## Train

```bash
uv run decoupled-ts train --config configs/freshretailnet.json
```

The best checkpoint is written to `runs/freshretailnet_glr/best.pt`.

The training config uses `device: "auto"` to prefer CUDA, then Apple MPS, then CPU. CUDA runs use AMP and TF32 where available. Data loading uses worker processes and prefetching; tune `batch_size` and `num_workers` in `configs/freshretailnet.json` for your machine.

## Evaluate Paper-Style Experiments

```bash
uv run decoupled-ts evaluate --checkpoint runs/freshretailnet_glr/best.pt
```

This runs FreshRetailNet-adapted versions of the paper's three representation evaluations:

- downstream prediction from learned local/global representations
- subgroup identification from global representations
- multi-window forecasting using GP conditional local latent prediction

Metrics are written to `runs/freshretailnet_glr/metrics.json`.

## Retail Multi-Grain Experiments

The retail-specific extension keeps the original static/dynamic idea, but splits the local representation into day-level and hour-level factors:

- `z_global`: store-product or store-category baseline demand.
- `z_day`: date-level movement such as weekday, holiday, weather, promotion, or event effects.
- `z_hour`: repeated intra-day demand shape such as morning, lunch, evening, or late-night peaks.
- `z_interaction`: optional day x hour interaction such as promotion-day lunch peaks.

Run a quick smoke test:

```bash
uv run decoupled-ts retail-experiment --config configs/retail_multigrain_smoke.json
```

Run the full synthetic ablation suite:

```bash
uv run decoupled-ts retail-experiment --config configs/retail_multigrain.json
```

Run on FreshRetailNet after preparing local parquet files:

```bash
uv run decoupled-ts retail-experiment --config configs/retail_multigrain_freshretailnet.json
```

The runner trains these variants from the config:

- `baseline_flatten_mlp`
- `global_only`
- `global_day`
- `global_hour`
- `global_day_hour`
- `global_day_hour_interaction`

Outputs are collected under `train.output_dir`:

- root `run.log`: overall experiment log
- root `summary.json` and `summary.csv`: final WAPE/MAE/RMSE/Bias comparison
- per-variant `run.log`: training log
- per-variant `history.jsonl`: epoch losses
- per-variant `metrics.json`: test metrics
- per-variant `z_global.npy`, `z_day.npy`, `z_hour.npy`: latent arrays for probes and visualization

## Residual Diagnostics

To validate the residual-centered hypothesis before training representation models, run:

```bash
uv run decoupled-ts residual-diagnostics --config configs/2-Exp-1_residual_diagnostics_smoke.json
uv run decoupled-ts residual-diagnostics --config configs/2-Exp-1_residual_diagnostics_synthetic.json
uv run decoupled-ts residual-diagnostics --config configs/2-Exp-1_residual_diagnostics_freshretailnet.json
```

This compares baseline components such as recent same-hour means, then analyzes structure in `r = y - b` by hour, weekday, and subgroup. Outputs are written under `analysis.output_dir`, including `baseline_metrics.json`, `summary.json`, and residual CSV heatmaps.

To train residual representation models and evaluate `b + r_hat` correction:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-2_to_6_residual_smoke.json
uv run decoupled-ts residual-experiment --config configs/2-Exp-2_to_6_residual_synthetic.json
uv run decoupled-ts residual-experiment --config configs/2-Exp-2_to_6_residual_freshretailnet.json
```

To compare residual models with and without counterfactual-style latent swap regularization:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-7_swap_regularization_smoke.json
uv run decoupled-ts residual-experiment --config configs/2-Exp-7_swap_regularization_synthetic.json
uv run decoupled-ts residual-experiment --config configs/2-Exp-7_swap_regularization_freshretailnet.json
```

To validate the hypothesis under a synthetic dataset with explicit residual global/day/hour/interaction structure:

```bash
uv run decoupled-ts residual-diagnostics --config configs/2-Exp-8_structured_residual_diagnostics_smoke.json
uv run decoupled-ts residual-experiment --config configs/2-Exp-8_structured_residual_smoke.json
uv run decoupled-ts residual-diagnostics --config configs/2-Exp-8_structured_residual_diagnostics_synthetic.json
uv run decoupled-ts residual-experiment --config configs/2-Exp-8_structured_residual_synthetic.json
```
