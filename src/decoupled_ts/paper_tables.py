from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .data import load_config
from .experiment_logging import write_json


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"input summary not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _format_value(value: Any) -> str:
    number = _as_float(value)
    if number is not None:
        return f"{number:.4f}"
    if value is None:
        return ""
    return str(value)


def _select_columns(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    return {column: row.get(column) for column in columns}


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if rows:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |")
    else:
        lines.append("No rows.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _aggregate_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("aggregate")
    if not isinstance(rows, list):
        raise ValueError("summary does not contain aggregate rows")
    return [row for row in rows if isinstance(row, dict)]


def _synthetic_component_rows(payload: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows = _aggregate_rows(payload)
    scenarios = cfg.get("scenarios")
    models = cfg.get("models")
    if scenarios is not None:
        scenario_set = {str(value) for value in scenarios}
        rows = [row for row in rows if str(row.get("scenario")) in scenario_set]
    if models is not None:
        model_set = {str(value) for value in models}
        rows = [row for row in rows if str(row.get("name")) in model_set]

    columns = [
        "scenario",
        "name",
        "runs",
        "residual_mae_mean",
        "residual_r2_mean",
        "component_total_true_residual_mae_mean",
        "component_global_corr_mean",
        "component_day_corr_mean",
        "component_hour_corr_mean",
        "component_interaction_corr_mean",
        "component_ablation_without_global_mae_delta_mean",
        "component_ablation_without_day_mae_delta_mean",
        "component_ablation_without_hour_mae_delta_mean",
        "component_ablation_without_interaction_mae_delta_mean",
    ]
    return [_select_columns(row, columns) for row in rows], columns


def _freshretail_correction_rows(payload: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows = _aggregate_rows(payload)
    scenarios = cfg.get("scenarios")
    models = cfg.get("models")
    if scenarios is not None:
        scenario_set = {str(value) for value in scenarios}
        rows = [row for row in rows if str(row.get("scenario")) in scenario_set]
    if models is not None:
        model_set = {str(value) for value in models}
        rows = [row for row in rows if str(row.get("name")) in model_set]

    columns = [
        "scenario",
        "name",
        "runs",
        "baseline_cell_mae_mean",
        "corrected_cell_mae_mean",
        "calibrated_corrected_cell_mae_mean",
        "corrected_cell_bias_mean",
        "calibrated_corrected_cell_bias_mean",
        "high_residual_top10_baseline_mae_mean",
        "high_residual_top10_corrected_mae_mean",
        "calibrated_high_residual_top10_corrected_mae_mean",
        "component_ablation_without_hour_mae_delta_mean",
        "calibrated_component_ablation_without_hour_mae_delta_mean",
        "hour_component_residual_profile_corr_mean",
        "calibrated_hour_component_residual_profile_corr_mean",
    ]
    return [_select_columns(row, columns) for row in rows], columns


def _statistical_validation_rows(payload: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("baseline_delta", []):
        if isinstance(row, dict):
            rows.append({"analysis": "baseline_delta", **row})
    for row in payload.get("paired_model_delta", []):
        if isinstance(row, dict):
            rows.append({"analysis": "paired_model_delta", **row})

    scenarios = cfg.get("scenarios")
    metrics = cfg.get("metrics")
    if scenarios is not None:
        scenario_set = {str(value) for value in scenarios}
        rows = [row for row in rows if str(row.get("scenario")) in scenario_set]
    if metrics is not None:
        metric_set = {str(value) for value in metrics}
        rows = [row for row in rows if str(row.get("metric")) in metric_set]

    columns = [
        "analysis",
        "scenario",
        "model",
        "left",
        "right",
        "metric",
        "baseline_metric",
        "runs",
        "delta_mean",
        "delta_mean_left_minus_right",
        "delta_ci_low",
        "delta_ci_high",
        "improved_runs",
        "worse_runs",
        "left_better_runs",
        "right_better_runs",
        "all_runs_improved",
        "all_runs_left_better",
    ]
    return [_select_columns(row, columns) for row in rows], columns


def _build_table(name: str, table_cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    path = Path(str(table_cfg["summary_path"]))
    payload = _load_json(path)
    kind = str(table_cfg["kind"])
    if kind == "synthetic_component_recovery":
        rows, columns = _synthetic_component_rows(payload, table_cfg)
        title = "Synthetic Component Recovery"
    elif kind == "freshretail_correction":
        rows, columns = _freshretail_correction_rows(payload, table_cfg)
        title = "FreshRetailNet Correction"
    elif kind == "statistical_validation":
        rows, columns = _statistical_validation_rows(payload, table_cfg)
        title = "Statistical Validation"
    else:
        raise ValueError(f"unknown paper table kind: {kind}")

    csv_path = out_dir / f"{name}.csv"
    md_path = out_dir / f"{name}.md"
    _write_csv(csv_path, rows, columns)
    _write_markdown(md_path, title, rows, columns)
    return {
        "name": name,
        "kind": kind,
        "summary_path": str(path),
        "rows": len(rows),
        "csv_path": str(csv_path),
        "markdown_path": str(md_path),
    }


def run_paper_tables(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    paper_cfg = config.get("paper_tables", {})
    out_dir = Path(str(paper_cfg.get("output_dir", "runs/2-Exp-23_paper_tables")))
    tables_cfg = paper_cfg.get("tables", {})
    if not isinstance(tables_cfg, dict) or not tables_cfg:
        raise ValueError("paper_tables.tables must contain at least one table")

    out_dir.mkdir(parents=True, exist_ok=True)
    table_summaries = []
    for name, table_cfg in tables_cfg.items():
        if not isinstance(table_cfg, dict):
            raise ValueError(f"table config must be an object: {name}")
        table_summaries.append(_build_table(str(name), table_cfg, out_dir))

    payload = {
        "config_path": config_path,
        "output_dir": str(out_dir),
        "tables": table_summaries,
    }
    write_json(out_dir / "summary.json", payload)
    return payload
