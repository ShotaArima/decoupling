from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_config
from .experiment_logging import write_json


def _read_summary(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"summary does not contain results list: {path}")
    return rows


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    if isinstance(value, int | float):
        return float(value)
    return None


def _comparison_metric_value(row: dict[str, Any], metric: str) -> float | None:
    value = _metric_value(row, metric)
    if value is None:
        return None
    if metric.endswith("_bias"):
        return abs(value)
    return value


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int, rng: np.random.Generator, alpha: float) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    if values.size == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(n_bootstrap, values.size), replace=True).mean(axis=1)
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return lo, hi


def _paired_rows(rows: list[dict[str, Any]], scenario: str, left_name: str, right_name: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    left = {
        int(row["seed"]): row
        for row in rows
        if row.get("scenario") == scenario and row.get("name") == left_name and "seed" in row
    }
    right = {
        int(row["seed"]): row
        for row in rows
        if row.get("scenario") == scenario and row.get("name") == right_name and "seed" in row
    }
    seeds = sorted(set(left) & set(right))
    return [(left[seed], right[seed]) for seed in seeds]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _baseline_delta_rows(rows: list[dict[str, Any]], config: dict[str, Any], rng: np.random.Generator) -> list[dict[str, Any]]:
    analysis_cfg = config.get("analysis", {})
    scenarios = [str(value) for value in analysis_cfg.get("scenarios", sorted({row.get("scenario", "") for row in rows}))]
    model_names = [str(value) for value in analysis_cfg.get("models", sorted({row.get("name", "") for row in rows}))]
    metrics = [str(value) for value in analysis_cfg.get("baseline_delta_metrics", ["corrected_cell_mae", "high_residual_top10_corrected_mae"])]
    n_bootstrap = int(analysis_cfg.get("n_bootstrap", 10000))
    alpha = 1.0 - float(analysis_cfg.get("confidence", 0.95))
    out_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for model_name in model_names:
            subset = [row for row in rows if row.get("scenario") == scenario and row.get("name") == model_name]
            for metric in metrics:
                baseline_metric = str(analysis_cfg.get("baseline_metric_map", {}).get(metric, "baseline_cell_mae"))
                if metric.startswith("calibrated_high_residual_top10"):
                    baseline_metric = "calibrated_high_residual_top10_baseline_mae"
                elif metric.startswith("high_residual_top10"):
                    baseline_metric = "high_residual_top10_baseline_mae"
                values = []
                for row in subset:
                    candidate = _metric_value(row, metric)
                    baseline = _metric_value(row, baseline_metric)
                    if candidate is not None and baseline is not None:
                        values.append(candidate - baseline)
                if not values:
                    continue
                arr = np.asarray(values, dtype=np.float64)
                ci_low, ci_high = _bootstrap_ci(arr, n_bootstrap, rng, alpha)
                out_rows.append(
                    {
                        "scenario": scenario,
                        "model": model_name,
                        "metric": metric,
                        "baseline_metric": baseline_metric,
                        "runs": int(arr.size),
                        "delta_mean": float(arr.mean()),
                        "delta_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                        "delta_ci_low": ci_low,
                        "delta_ci_high": ci_high,
                        "improved_runs": int(np.sum(arr < 0.0)),
                        "worse_runs": int(np.sum(arr > 0.0)),
                        "all_runs_improved": bool(np.all(arr < 0.0)),
                    }
                )
    return out_rows


def _paired_model_rows(rows: list[dict[str, Any]], config: dict[str, Any], rng: np.random.Generator) -> list[dict[str, Any]]:
    analysis_cfg = config.get("analysis", {})
    comparisons = analysis_cfg.get("model_comparisons", [])
    metrics = [str(value) for value in analysis_cfg.get("model_comparison_metrics", ["calibrated_corrected_cell_mae"])]
    n_bootstrap = int(analysis_cfg.get("n_bootstrap", 10000))
    alpha = 1.0 - float(analysis_cfg.get("confidence", 0.95))
    out_rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        scenario = str(comparison["scenario"])
        left_name = str(comparison["left"])
        right_name = str(comparison["right"])
        pairs = _paired_rows(rows, scenario, left_name, right_name)
        for metric in metrics:
            values = []
            for left, right in pairs:
                left_value = _comparison_metric_value(left, metric)
                right_value = _comparison_metric_value(right, metric)
                if left_value is not None and right_value is not None:
                    values.append(left_value - right_value)
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            ci_low, ci_high = _bootstrap_ci(arr, n_bootstrap, rng, alpha)
            out_rows.append(
                {
                    "scenario": scenario,
                    "left": left_name,
                    "right": right_name,
                    "metric": metric,
                    "runs": int(arr.size),
                    "delta_mean_left_minus_right": float(arr.mean()),
                    "delta_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    "delta_ci_low": ci_low,
                    "delta_ci_high": ci_high,
                    "left_better_runs": int(np.sum(arr < 0.0)),
                    "right_better_runs": int(np.sum(arr > 0.0)),
                    "all_runs_left_better": bool(np.all(arr < 0.0)),
                }
            )
    return out_rows


def run_residual_result_analysis(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    analysis_cfg = config.get("analysis", {})
    summary_path = Path(str(analysis_cfg["summary_path"]))
    out_dir = Path(str(analysis_cfg.get("output_dir", "runs/residual_result_analysis")))
    seed = int(analysis_cfg.get("seed", 17))
    rng = np.random.default_rng(seed)
    rows = _read_summary(summary_path)
    baseline_delta = _baseline_delta_rows(rows, config, rng)
    paired_model_delta = _paired_model_rows(rows, config, rng)
    payload = {
        "summary_path": str(summary_path),
        "baseline_delta": baseline_delta,
        "paired_model_delta": paired_model_delta,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", payload)
    _write_csv(out_dir / "baseline_delta.csv", baseline_delta)
    _write_csv(out_dir / "paired_model_delta.csv", paired_model_delta)
    return payload
