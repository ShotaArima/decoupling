from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

from .data import load_config
from .experiment_logging import setup_run_logger, write_json
from .residual_experiments import run_residual_experiments


def _mean_std(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted({(str(row.get("subset", "default")), row["name"]) for row in rows})
    metric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {"name", "seed", "subset"} and isinstance(value, int | float)
        }
    )
    summary = []
    for subset_name, variant in groups:
        subset = [row for row in rows if row["name"] == variant and str(row.get("subset", "default")) == subset_name]
        out: dict[str, Any] = {"subset": subset_name, "name": variant, "runs": len(subset)}
        for key in metric_keys:
            values = [float(row[key]) for row in subset if key in row]
            if not values:
                continue
            mean = sum(values) / len(values)
            var = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = var**0.5
        summary.append(out)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_residual_sweep(config_path: str) -> dict[str, Any]:
    base_config = load_config(config_path)
    sweep_cfg = base_config.get("sweep", {})
    seeds = [int(seed) for seed in sweep_cfg.get("seeds", [base_config["seed"]])]
    root_out = Path(sweep_cfg.get("output_dir", str(base_config["train"]["output_dir"]) + "_sweep"))
    subsets = sweep_cfg.get("subsets") or [{"name": "default", "subset_filter": base_config.get("dataset", {}).get("subset_filter", {})}]
    root_out.mkdir(parents=True, exist_ok=True)
    logger = setup_run_logger(root_out, name="residual_sweep")
    logger.info("Residual sweep start config=%s seeds=%s subsets=%s", config_path, seeds, [row.get("name", "default") for row in subsets])
    write_json(root_out / "base_config.json", base_config)

    all_rows: list[dict[str, Any]] = []
    for subset in subsets:
        subset_name = str(subset.get("name", "default"))
        for seed in seeds:
            config = deepcopy(base_config)
            config["seed"] = seed
            config["dataset"]["subset_filter"] = {
                **config.get("dataset", {}).get("subset_filter", {}),
                **subset.get("subset_filter", {}),
            }
            run_name = f"{subset_name}_seed_{seed}"
            config["train"]["output_dir"] = str(root_out / run_name)
            config_path_for_seed = root_out / f"{run_name}_config.json"
            write_json(config_path_for_seed, config)
            logger.info("Running subset=%s seed=%d output_dir=%s", subset_name, seed, config["train"]["output_dir"])
            result = run_residual_experiments(str(config_path_for_seed))
            for row in result["results"]:
                all_rows.append({"subset": subset_name, "seed": seed, **row})

    aggregate = _mean_std(all_rows)
    _write_csv(root_out / "all_results.csv", all_rows)
    _write_csv(root_out / "aggregate.csv", aggregate)
    summary = {"seeds": seeds, "results": all_rows, "aggregate": aggregate}
    write_json(root_out / "summary.json", summary)
    logger.info("Residual sweep complete output_dir=%s", root_out)
    return summary
