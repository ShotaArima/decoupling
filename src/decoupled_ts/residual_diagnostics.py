from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader

from .data import load_config
from .experiment_logging import setup_run_logger, write_json
from .metrics import regression_metrics
from .retail_data import build_retail_data
from .retail_models import flatten_to_grid


BASELINE_METHODS = (
    "overall_mean",
    "series_mean",
    "recent_mean",
    "same_hour_recent_mean",
    "weekday_same_hour_mean",
)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, fallback: np.ndarray | float) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.broadcast_to(fallback, numerator.shape).astype(np.float32).copy(), where=denominator > 0)


def _collect_split(dataset, batch_size: int) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    grids, masks, subgroups, static_ids = [], [], [], []
    for batch in loader:
        grids.append(flatten_to_grid(batch["x"]).numpy())
        masks.append(flatten_to_grid(batch["mask"]).numpy())
        subgroups.append(batch["subgroup"].numpy())
        static_ids.append(batch["static_ids"].numpy())
    return {
        "grid": np.concatenate(grids, axis=0),
        "mask": np.concatenate(masks, axis=0),
        "subgroup": np.concatenate(subgroups, axis=0),
        "static_ids": np.concatenate(static_ids, axis=0),
    }


def _rolling_recent_mean(sales: np.ndarray, observed: np.ndarray, recent_days: int) -> np.ndarray:
    samples, days, hours = sales.shape
    series_sum = (sales * observed).sum(axis=(1, 2), keepdims=True)
    series_count = observed.sum(axis=(1, 2), keepdims=True)
    series_mean = _safe_divide(series_sum, series_count, float((sales * observed).sum() / max(observed.sum(), 1.0)))
    baseline = np.empty_like(sales, dtype=np.float32)
    for day in range(days):
        start = max(0, day - recent_days)
        if start == day:
            baseline[:, day, :] = series_mean[:, 0, 0][:, None]
            continue
        window_sales = sales[:, start:day, :] * observed[:, start:day, :]
        window_count = observed[:, start:day, :].sum(axis=(1, 2), keepdims=True)
        window_mean = _safe_divide(window_sales.sum(axis=(1, 2), keepdims=True), window_count, series_mean)
        baseline[:, day, :] = window_mean[:, 0, 0][:, None]
    return baseline


def _rolling_same_hour_mean(sales: np.ndarray, observed: np.ndarray, recent_days: int) -> np.ndarray:
    samples, days, hours = sales.shape
    series_hour_sum = (sales * observed).sum(axis=1)
    series_hour_count = observed.sum(axis=1)
    series_mean = _safe_divide(
        (sales * observed).sum(axis=(1, 2), keepdims=True),
        observed.sum(axis=(1, 2), keepdims=True),
        float((sales * observed).sum() / max(observed.sum(), 1.0)),
    )
    series_hour_mean = _safe_divide(series_hour_sum, series_hour_count, series_mean[:, 0, 0][:, None])
    baseline = np.empty_like(sales, dtype=np.float32)
    for day in range(days):
        start = max(0, day - recent_days)
        if start == day:
            baseline[:, day, :] = series_hour_mean
            continue
        window_sales = sales[:, start:day, :] * observed[:, start:day, :]
        window_count = observed[:, start:day, :].sum(axis=1)
        baseline[:, day, :] = _safe_divide(window_sales.sum(axis=1), window_count, series_hour_mean)
    return baseline


def _weekday_same_hour_mean(sales: np.ndarray, observed: np.ndarray, weekday: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    _, days, hours = sales.shape
    weekday_day = weekday[:, :, 0] if weekday.ndim == 3 else weekday
    baseline = np.empty_like(sales, dtype=np.float32)
    for day in range(days):
        same_weekday = weekday_day[:, :day] == weekday_day[:, day : day + 1]
        day_sales = sales[:, :day, :] * observed[:, :day, :] * same_weekday[:, :, None]
        day_count = observed[:, :day, :] * same_weekday[:, :, None]
        if day == 0:
            baseline[:, day, :] = fallback[:, day, :]
        else:
            baseline[:, day, :] = _safe_divide(day_sales.sum(axis=1), day_count.sum(axis=1), fallback[:, day, :])
    return baseline


def compute_baselines(sales: np.ndarray, observed: np.ndarray, weekday: np.ndarray, recent_days: int) -> dict[str, np.ndarray]:
    overall = float((sales * observed).sum() / max(observed.sum(), 1.0))
    series_mean = _safe_divide(
        (sales * observed).sum(axis=(1, 2), keepdims=True),
        observed.sum(axis=(1, 2), keepdims=True),
        overall,
    )
    recent = _rolling_recent_mean(sales, observed, recent_days)
    same_hour = _rolling_same_hour_mean(sales, observed, recent_days)
    return {
        "overall_mean": np.full_like(sales, overall, dtype=np.float32),
        "series_mean": np.broadcast_to(series_mean, sales.shape).astype(np.float32),
        "recent_mean": recent,
        "same_hour_recent_mean": same_hour,
        "weekday_same_hour_mean": _weekday_same_hour_mean(sales, observed, weekday, same_hour),
    }


def baseline_metrics(sales: np.ndarray, observed: np.ndarray, baselines: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    y = sales[observed > 0]
    y_var = float(np.var(y))
    results = {}
    for name, baseline in baselines.items():
        pred = baseline[observed > 0]
        residual = y - pred
        metrics = regression_metrics(y, pred)
        sse = float(np.sum(np.square(residual)))
        sst = float(np.sum(np.square(y - y.mean())))
        metrics["r2"] = 1.0 - sse / max(sst, 1e-6)
        metrics["residual_variance_ratio"] = float(np.var(residual) / max(y_var, 1e-6))
        metrics["residual_std"] = float(np.std(residual))
        results[name] = metrics
    return results


def _feature_indices(config: dict[str, Any]) -> dict[str, int | None]:
    data_cfg = config["dataset"]
    if data_cfg["name"] in {"synthetic_retail", "structured_residual_retail", "component_residual_retail"}:
        return {
            "discount": 2,
            "holiday": 3,
            "weather": 4,
            "hour_sin": 5,
            "hour_cos": 6,
            "dow_sin": 7,
            "dow_cos": 8,
        }
    numeric_cols = list(data_cfg["daily_numeric_columns"])
    offset = 2 + len(numeric_cols)
    return {
        "discount": 2 + numeric_cols.index("discount") if "discount" in numeric_cols else None,
        "holiday": 2 + numeric_cols.index("holiday_flag") if "holiday_flag" in numeric_cols else None,
        "weather": 2 + numeric_cols.index("avg_temperature") if "avg_temperature" in numeric_cols else None,
        "hour_sin": offset,
        "hour_cos": offset + 1,
        "dow_sin": offset + 2,
        "dow_cos": offset + 3,
    }


def _labels_from_grid(grid: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray]:
    idx = _feature_indices(config)
    hour_angle = np.arctan2(grid[..., int(idx["hour_sin"])], grid[..., int(idx["hour_cos"])])
    dow_angle = np.arctan2(grid[..., int(idx["dow_sin"])], grid[..., int(idx["dow_cos"])])
    labels = {
        "hour": np.rint((np.mod(hour_angle, 2 * np.pi)) / (2 * np.pi) * 24).astype(np.int64) % 24,
        "weekday": np.rint((np.mod(dow_angle, 2 * np.pi)) / (2 * np.pi) * 7).astype(np.int64) % 7,
    }
    for key in ("discount", "holiday", "weather"):
        labels[key] = np.zeros(grid.shape[:3], dtype=np.float32) if idx[key] is None else grid[..., int(idx[key])].astype(np.float32)
    return labels


def _write_group_summary(path: Path, residual: np.ndarray, sales: np.ndarray, observed: np.ndarray, group: np.ndarray, group_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[group_name, "count", "sales_mean", "residual_mean", "residual_abs_mean", "residual_std"])
        writer.writeheader()
        for value in sorted(np.unique(group[observed > 0]).tolist()):
            keep = (group == value) & (observed > 0)
            row_residual = residual[keep]
            writer.writerow(
                {
                    group_name: int(value),
                    "count": int(keep.sum()),
                    "sales_mean": float(sales[keep].mean()),
                    "residual_mean": float(row_residual.mean()),
                    "residual_abs_mean": float(np.abs(row_residual).mean()),
                    "residual_std": float(row_residual.std()),
                }
            )


def _write_heatmap(path: Path, residual: np.ndarray, observed: np.ndarray, weekday: np.ndarray, hour: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["weekday", *[f"hour_{h}" for h in range(24)]])
        for day in range(7):
            row = [day]
            for h in range(24):
                keep = (weekday == day) & (hour == h) & (observed > 0)
                row.append(float(residual[keep].mean()) if keep.any() else "")
            writer.writerow(row)


def _variance_explained_by_group(residual: np.ndarray, observed: np.ndarray, group: np.ndarray) -> float:
    keep = observed > 0
    y = residual[keep]
    if y.size == 0:
        return 0.0
    total_var = float(np.var(y))
    if total_var <= 1e-12:
        return 0.0
    g = group[keep]
    explained = 0.0
    for value in np.unique(g):
        group_y = y[g == value]
        if group_y.size:
            explained += group_y.size / y.size * float((group_y.mean() - y.mean()) ** 2)
    return float(explained / total_var)


def _bin_by_quantiles(values: np.ndarray, observed: np.ndarray, bins: int) -> np.ndarray:
    keep_values = values[observed > 0]
    if keep_values.size == 0 or np.allclose(keep_values, keep_values[0]):
        return np.zeros_like(values, dtype=np.int64)
    quantiles = np.quantile(keep_values, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    return np.digitize(values, np.unique(quantiles)).astype(np.int64)


def _write_high_residual_summary(
    path: Path,
    residual: np.ndarray,
    sales: np.ndarray,
    observed: np.ndarray,
    labels: dict[str, np.ndarray],
    subgroup: np.ndarray,
    top_rate: float,
) -> dict[str, float]:
    keep = observed > 0
    abs_residual = np.abs(residual)
    threshold = float(np.quantile(abs_residual[keep], 1.0 - top_rate)) if keep.any() else 0.0
    high = keep & (abs_residual >= threshold)
    low = keep & (abs_residual < threshold)
    fields = ["segment", "count", "sales_mean", "abs_residual_mean", "discount_mean", "holiday_rate", "stockout_rate", "most_common_hour", "most_common_weekday", "subgroup_classes"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, mask in (("top_abs_residual", high), ("other_observed", low)):
            if not mask.any():
                continue
            hour_values, hour_counts = np.unique(labels["hour"][mask], return_counts=True)
            weekday_values, weekday_counts = np.unique(labels["weekday"][mask], return_counts=True)
            writer.writerow(
                {
                    "segment": name,
                    "count": int(mask.sum()),
                    "sales_mean": float(sales[mask].mean()),
                    "abs_residual_mean": float(abs_residual[mask].mean()),
                    "discount_mean": float(labels["discount"][mask].mean()),
                    "holiday_rate": float(labels["holiday"][mask].mean()),
                    "stockout_rate": float((1.0 - observed)[mask].mean()),
                    "most_common_hour": int(hour_values[np.argmax(hour_counts)]),
                    "most_common_weekday": int(weekday_values[np.argmax(weekday_counts)]),
                    "subgroup_classes": int(len(np.unique(subgroup[mask]))),
                }
            )
    return {
        "top_abs_residual_rate": float(top_rate),
        "top_abs_residual_threshold": threshold,
        "top_abs_residual_count": float(high.sum()),
        "top_abs_residual_mean_abs": float(abs_residual[high].mean()) if high.any() else 0.0,
        "other_observed_mean_abs": float(abs_residual[low].mean()) if low.any() else 0.0,
    }


def _baseline_metrics_by_series_activity(
    sales: np.ndarray,
    observed: np.ndarray,
    baselines: dict[str, np.ndarray],
    out_dir: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    observed_counts = observed.sum(axis=(1, 2)).clip(min=1.0)
    series_mean = (sales * observed).sum(axis=(1, 2)) / observed_counts
    active = series_mean > 0.0
    segments = {"zero_mean_series": ~active, "nonzero_mean_series": active}
    rows = []
    results: dict[str, dict[str, dict[str, float]]] = {}
    for segment_name, sample_mask in segments.items():
        if not sample_mask.any():
            continue
        segment_metrics = baseline_metrics(sales[sample_mask], observed[sample_mask], {k: v[sample_mask] for k, v in baselines.items()})
        results[segment_name] = segment_metrics
        for baseline_name, metrics in segment_metrics.items():
            rows.append({"segment": segment_name, "baseline": baseline_name, **metrics})
    if rows:
        with (out_dir / "baseline_metrics_by_series_activity.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["segment", "baseline", *sorted({key for row in rows for key in row if key not in {"segment", "baseline"}})]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return results


def _run_nonlinear_probes(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, float]:
    if len(y) < 100:
        return {}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    split = max(1, int(len(y) * 0.8))
    train_idx = order[:split]
    test_idx = order[split:]
    if len(test_idx) == 0:
        return {}
    metrics: dict[str, float] = {}
    rf = RandomForestRegressor(n_estimators=80, min_samples_leaf=20, random_state=seed, n_jobs=-1)
    rf.fit(x[train_idx], y[train_idx])
    rf_pred = rf.predict(x[test_idx])
    metrics["random_forest_probe_r2"] = float(rf.score(x[test_idx], y[test_idx]))
    metrics["random_forest_probe_mae"] = float(np.mean(np.abs(rf_pred - y[test_idx])))
    gb = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=seed)
    gb.fit(x[train_idx], y[train_idx])
    gb_pred = gb.predict(x[test_idx])
    metrics["gradient_boosting_probe_r2"] = float(gb.score(x[test_idx], y[test_idx]))
    metrics["gradient_boosting_probe_mae"] = float(np.mean(np.abs(gb_pred - y[test_idx])))
    return metrics


def residual_structure_diagnostics(
    split_data: dict[str, np.ndarray],
    config: dict[str, Any],
    baseline: np.ndarray,
    baselines: dict[str, np.ndarray],
    out_dir: Path,
) -> dict[str, float]:
    grid = split_data["grid"]
    sales = grid[..., 0].astype(np.float32)
    observed = split_data["mask"][..., 0].astype(np.float32)
    residual = sales - baseline
    labels = _labels_from_grid(grid, config)
    subgroup = np.broadcast_to(split_data["subgroup"][:, None, None], sales.shape)
    stockout_rate = np.broadcast_to((1.0 - observed).mean(axis=(1, 2))[:, None, None], sales.shape)
    stockout_rate_bin = _bin_by_quantiles(stockout_rate, observed, int(config.get("analysis", {}).get("diagnostic_bins", 5)))
    discount_bin = _bin_by_quantiles(labels["discount"], observed, int(config.get("analysis", {}).get("diagnostic_bins", 5)))
    _write_group_summary(out_dir / "residual_by_hour.csv", residual, sales, observed, labels["hour"], "hour")
    _write_group_summary(out_dir / "residual_by_weekday.csv", residual, sales, observed, labels["weekday"], "weekday")
    _write_group_summary(out_dir / "residual_by_subgroup.csv", residual, sales, observed, subgroup, "subgroup")
    _write_group_summary(out_dir / "residual_by_stockout_rate_bin.csv", residual, sales, observed, stockout_rate_bin, "stockout_rate_bin")
    _write_group_summary(out_dir / "residual_by_discount_bin.csv", residual, sales, observed, discount_bin, "discount_bin")
    _write_heatmap(out_dir / "residual_weekday_hour_heatmap.csv", residual, observed, labels["weekday"], labels["hour"])
    top_summary = _write_high_residual_summary(
        out_dir / "high_abs_residual_top10_summary.csv",
        residual,
        sales,
        observed,
        labels,
        subgroup,
        top_rate=float(config.get("analysis", {}).get("top_residual_rate", 0.1)),
    )
    activity_metrics = _baseline_metrics_by_series_activity(sales, observed, baselines, out_dir)

    keep = observed > 0
    features = [
        labels["discount"][keep],
        labels["holiday"][keep],
        labels["weather"][keep],
        labels["hour"][keep].astype(np.float32),
        labels["weekday"][keep].astype(np.float32),
        subgroup[keep].astype(np.float32),
        stockout_rate[keep].astype(np.float32),
        discount_bin[keep].astype(np.float32),
    ]
    x = np.stack(features, axis=1)
    y = residual[keep]
    max_rows = int(config.get("analysis", {}).get("max_regression_rows", 200_000))
    if len(y) > max_rows:
        rng = np.random.default_rng(int(config["seed"]))
        sample = rng.choice(len(y), size=max_rows, replace=False)
        x = x[sample]
        y = y[sample]
    reg = LinearRegression()
    reg.fit(x, y)
    pred = reg.predict(x)
    group_explained = {
        "group_variance_explained_hour": _variance_explained_by_group(residual, observed, labels["hour"]),
        "group_variance_explained_weekday": _variance_explained_by_group(residual, observed, labels["weekday"]),
        "group_variance_explained_subgroup": _variance_explained_by_group(residual, observed, subgroup),
        "group_variance_explained_stockout_rate_bin": _variance_explained_by_group(residual, observed, stockout_rate_bin),
        "group_variance_explained_discount_bin": _variance_explained_by_group(residual, observed, discount_bin),
    }
    probe_metrics = _run_nonlinear_probes(x, y, int(config["seed"])) if bool(config.get("analysis", {}).get("nonlinear_probes", True)) else {}
    return {
        "residual_mean": float(y.mean()),
        "residual_abs_mean": float(np.abs(y).mean()),
        "residual_std": float(y.std()),
        "observed_cells": float(keep.sum()),
        "linear_probe_r2": float(reg.score(x, y)),
        "linear_probe_mae": float(np.mean(np.abs(pred - y))),
        **group_explained,
        **probe_metrics,
        **{f"high_residual_{key}": value for key, value in top_summary.items()},
        "series_activity_segments": float(len(activity_metrics)),
    }


def run_residual_diagnostics(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    analysis_cfg = config.get("analysis", {})
    out_dir = Path(analysis_cfg.get("output_dir", config["train"]["output_dir"] + "_residual_diagnostics"))
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_run_logger(out_dir, name="residual")
    logger.info("Residual diagnostics start config=%s", config_path)
    write_json(out_dir / "resolved_config.json", config)

    bundle = build_retail_data(config)
    split_name = str(analysis_cfg.get("split", "train"))
    dataset = getattr(bundle, split_name)
    split_data = _collect_split(dataset, batch_size=int(analysis_cfg.get("batch_size", config["train"].get("batch_size", 256))))
    grid = split_data["grid"]
    sales = grid[..., 0].astype(np.float32)
    observed = split_data["mask"][..., 0].astype(np.float32)
    labels = _labels_from_grid(grid, config)
    recent_days = int(config["dataset"].get("recent_days", analysis_cfg.get("recent_days", 7)))
    baselines = compute_baselines(sales, observed, labels["weekday"], recent_days)
    metrics = baseline_metrics(sales, observed, baselines)
    write_json(out_dir / "baseline_metrics.json", metrics)

    chosen = str(analysis_cfg.get("baseline_method", "same_hour_recent_mean"))
    if chosen not in baselines:
        raise ValueError(f"unknown baseline_method={chosen!r}; choose one of {BASELINE_METHODS}")
    residual_metrics = residual_structure_diagnostics(split_data, config, baselines[chosen], baselines, out_dir)
    summary = {
        "split": split_name,
        "examples": float(len(dataset)),
        "baseline_method": chosen,
        "baseline_metrics": metrics,
        "residual_structure": residual_metrics,
    }
    write_json(out_dir / "summary.json", summary)
    logger.info("Residual diagnostics complete: %s", summary)
    return summary
