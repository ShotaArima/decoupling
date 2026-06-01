from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import load_config
from .experiment_logging import setup_run_logger, write_json
from .metrics import regression_metrics
from .retail_data import build_retail_data
from .retail_experiments import predict_naive_batch
from .retail_models import flatten_to_grid


def _safe_qcut(values: np.ndarray, bins: int) -> np.ndarray:
    if len(np.unique(values)) <= 1:
        return np.zeros_like(values, dtype=np.int64)
    return pd.qcut(values, q=min(bins, len(np.unique(values))), labels=False, duplicates="drop").astype(np.int64)


def _bin_by_quantile(values: np.ndarray, bins: int) -> np.ndarray:
    return np.asarray(_safe_qcut(values, bins), dtype=np.int64)


def _metrics_by_group(frame: pd.DataFrame, group_col: str, model_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, group_df in frame.groupby(group_col, dropna=False):
        for model_col in model_cols:
            metrics = regression_metrics(group_df["y_true"].to_numpy(), group_df[model_col].to_numpy())
            rows.append(
                {
                    "group_col": group_col,
                    "group": group,
                    "model": model_col.removeprefix("pred_"),
                    "n": len(group_df),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _pairwise_comparison(frame: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    abs_errors = {col: np.abs(frame[col].to_numpy() - frame["y_true"].to_numpy()) for col in model_cols}
    for left in model_cols:
        for right in model_cols:
            if left == right:
                continue
            diff = abs_errors[left] - abs_errors[right]
            rows.append(
                {
                    "left_model": left.removeprefix("pred_"),
                    "right_model": right.removeprefix("pred_"),
                    "mean_abs_error_delta_left_minus_right": float(np.mean(diff)),
                    "left_better_rate": float(np.mean(diff < 0.0)),
                    "ties_rate": float(np.mean(diff == 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _load_external_predictions(config: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    merged = frame
    for item in config.get("prediction_sources", []):
        name = item["name"]
        path = Path(item["path"])
        if not path.exists():
            if bool(config.get("ignore_missing_prediction_sources", True)):
                continue
            raise FileNotFoundError(f"prediction source does not exist: {path}")
        pred_df = pd.read_csv(path)
        if "index" not in pred_df or "y_pred" not in pred_df:
            raise ValueError(f"prediction source must contain index and y_pred columns: {path}")
        source = pred_df[["index", "y_pred"]].rename(columns={"y_pred": f"pred_{name}"})
        merged = merged.merge(source, on="index", how="left", validate="one_to_one")
    return merged


def _collect_analysis_frame(config: dict[str, Any]) -> pd.DataFrame:
    bundle = build_retail_data(config)
    loader = DataLoader(bundle.test, batch_size=int(config["analysis"].get("batch_size", 256)), shuffle=False)
    recent_days = int(config["dataset"].get("recent_days", 7))
    rows: list[dict[str, Any]] = []
    offset = 0

    for batch in loader:
        x = batch["x"]
        mask = batch["mask"]
        sales_grid = flatten_to_grid(x)[..., 0]
        sales_mask = flatten_to_grid(mask)[..., 0]
        observed_sales = sales_grid * sales_mask
        y_true = batch["target"].numpy()

        preds = {
            "pred_naive_last_day": predict_naive_batch(batch, "naive_last_day", bundle.forecast_days, recent_days).numpy(),
            "pred_naive_recent_mean": predict_naive_batch(batch, "naive_recent_mean", bundle.forecast_days, recent_days).numpy(),
            "pred_naive_same_hour_recent_mean": predict_naive_batch(
                batch, "naive_same_hour_recent_mean", bundle.forecast_days, recent_days
            ).numpy(),
        }
        history_total = observed_sales.sum(dim=(1, 2)).numpy()
        history_daily_mean = observed_sales.sum(dim=(1, 2)).numpy() / max(bundle.days, 1)
        history_zero_rate = (sales_grid == 0.0).float().mean(dim=(1, 2)).numpy()
        history_stockout_rate = (1.0 - sales_mask).mean(dim=(1, 2)).numpy()
        recent_total = observed_sales[:, -recent_days:, :].sum(dim=(1, 2)).numpy()
        same_hour_baseline = preds["pred_naive_same_hour_recent_mean"]
        static_ids = batch.get("static_ids")

        for i in range(len(y_true)):
            row: dict[str, Any] = {
                "index": offset + i,
                "y_true": float(y_true[i]),
                "history_total": float(history_total[i]),
                "history_daily_mean": float(history_daily_mean[i]),
                "history_zero_rate": float(history_zero_rate[i]),
                "history_stockout_rate": float(history_stockout_rate[i]),
                "recent_total": float(recent_total[i]),
                "same_hour_baseline": float(same_hour_baseline[i]),
                "same_hour_error": float(same_hour_baseline[i] - y_true[i]),
                "subgroup": int(batch["subgroup"][i]),
            }
            if static_ids is not None:
                for j, value in enumerate(static_ids[i].tolist()):
                    row[f"static_id_{j}"] = int(value)
            for key, value in preds.items():
                row[key] = float(value[i])
            rows.append(row)
        offset += len(y_true)

    frame = pd.DataFrame(rows)
    quantile_bins = int(config["analysis"].get("quantile_bins", 5))
    frame["target_quantile"] = _bin_by_quantile(frame["y_true"].to_numpy(), quantile_bins)
    frame["zero_rate_quantile"] = _bin_by_quantile(frame["history_zero_rate"].to_numpy(), quantile_bins)
    frame["stockout_rate_quantile"] = _bin_by_quantile(frame["history_stockout_rate"].to_numpy(), quantile_bins)
    return _load_external_predictions(config["analysis"], frame)


def _write_summary(frame: pd.DataFrame, model_cols: list[str], out_dir: Path) -> dict[str, Any]:
    rows = []
    for col in model_cols:
        metrics = regression_metrics(frame["y_true"].to_numpy(), frame[col].to_numpy())
        rows.append({"model": col.removeprefix("pred_"), **metrics})
    summary_df = pd.DataFrame(rows).sort_values("wape")
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    correlations = {
        "same_hour_target_corr": float(np.corrcoef(frame["same_hour_baseline"], frame["y_true"])[0, 1])
        if len(frame) > 1
        else 0.0,
        "history_daily_mean_target_corr": float(np.corrcoef(frame["history_daily_mean"], frame["y_true"])[0, 1])
        if len(frame) > 1
        else 0.0,
        "history_zero_rate_target_corr": float(np.corrcoef(frame["history_zero_rate"], frame["y_true"])[0, 1])
        if len(frame) > 1
        else 0.0,
        "history_stockout_rate_target_corr": float(np.corrcoef(frame["history_stockout_rate"], frame["y_true"])[0, 1])
        if len(frame) > 1
        else 0.0,
    }
    payload = {
        "summary": rows,
        "correlations": correlations,
        "n": int(len(frame)),
    }
    write_json(out_dir / "summary.json", payload)
    write_json(out_dir / "correlations.json", correlations)
    return payload


def run_same_hour_analysis(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    out_dir = Path(config["analysis"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_run_logger(out_dir, name="same_hour_analysis")
    logger.info("Starting same-hour analysis config=%s", config_path)
    write_json(out_dir / "resolved_config.json", config)

    frame = _collect_analysis_frame(config)
    frame.to_csv(out_dir / "all_predictions_with_features.csv", index=False)
    model_cols = [col for col in frame.columns if col.startswith("pred_")]
    summary = _write_summary(frame, model_cols, out_dir)
    _metrics_by_group(frame, "target_quantile", model_cols).to_csv(out_dir / "error_by_target_quantile.csv", index=False)
    _metrics_by_group(frame, "zero_rate_quantile", model_cols).to_csv(out_dir / "error_by_zero_rate.csv", index=False)
    _metrics_by_group(frame, "stockout_rate_quantile", model_cols).to_csv(out_dir / "error_by_stockout_rate.csv", index=False)
    _pairwise_comparison(frame, model_cols).to_csv(out_dir / "model_pairwise_comparison.csv", index=False)

    if "static_id_1" in frame.columns:
        _metrics_by_group(frame, "static_id_1", model_cols).to_csv(out_dir / "error_by_store_id.csv", index=False)
    if "static_id_4" in frame.columns:
        _metrics_by_group(frame, "static_id_4", model_cols).to_csv(out_dir / "error_by_second_category_id.csv", index=False)
    if "static_id_5" in frame.columns:
        _metrics_by_group(frame, "static_id_5", model_cols).to_csv(out_dir / "error_by_third_category_id.csv", index=False)

    logger.info("Same-hour analysis complete output_dir=%s models=%s", out_dir, model_cols)
    return summary
