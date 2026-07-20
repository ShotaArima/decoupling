"""Analyze FreshRetailNet output-component separation for 2-Exp-35.

The reference labels are clusters fitted to observed residual hour profiles,
not to the estimated component. This avoids using the displayed embedding to
define its own labels. Metrics are evaluated in the original 24-dimensional
profile space; t-SNE is used only for visualization.

Usage:
    uv run --with matplotlib python scripts/analyze_2_exp_35_freshretailnet_tsne.py \
        --run-dir runs/2-Exp-35_freshretailnet_component_tsne/series_mean_all \
        --out-dir figures/2-Exp-35
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.manifold import TSNE

from decoupled_ts.component_analysis import (
    component_constraint_metrics,
    component_pairwise_correlations,
    evaluate_hour_component,
    safe_silhouette,
)


REQUIRED_ARRAYS = (
    "global_component",
    "day_component",
    "hour_component",
    "interaction_component",
    "residual",
    "observed",
    "static_ids",
)


def parse_seed(path: Path) -> int:
    try:
        return int(path.name.removeprefix("seed_"))
    except ValueError as exc:
        raise ValueError(f"invalid seed directory name: {path.name}") from exc


def load_arrays(variant_dir: Path) -> dict[str, np.ndarray]:
    missing = [name for name in REQUIRED_ARRAYS if not (variant_dir / f"{name}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"{variant_dir}: missing arrays: {', '.join(missing)}")
    return {name: np.load(variant_dir / f"{name}.npy") for name in REQUIRED_ARRAYS}


def numeric_summary(rows: list[dict[str, Any]], variants: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    excluded = {"seed", "variant", "cluster_counts"}
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        metrics: dict[str, Any] = {"runs": len(selected)}
        keys = sorted({key for row in selected for key in row if key not in excluded})
        for key in keys:
            values = [float(row[key]) for row in selected if isinstance(row.get(key), int | float)]
            if not values:
                continue
            metrics[f"{key}_mean"] = float(np.mean(values))
            metrics[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary[variant] = metrics
    return summary


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=True) if isinstance(value, list | dict) else value
                    for key, value in row.items()
                }
            )


def tsne_embedding(features: np.ndarray, perplexity: float, random_state: int) -> np.ndarray:
    if len(features) < 4:
        raise ValueError("t-SNE requires at least four valid series")
    effective_perplexity = min(float(perplexity), max(2.0, (len(features) - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    ).fit_transform(features)


def plot_scatter(
    representative: dict[str, dict[str, Any]],
    variants: list[str],
    metadata_name: str,
    perplexity: float,
    random_state: int,
    out_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata_available = any(len(np.unique(item["metadata"])) > 1 for item in representative.values())
    columns = 2 if metadata_available else 1
    fig, axes = plt.subplots(
        len(variants),
        columns,
        figsize=(3.3 * columns, 2.8 * len(variants)),
        squeeze=False,
    )
    palette = plt.get_cmap("tab10")
    for row_index, variant in enumerate(variants):
        item = representative[variant]
        details = item["details"]
        embedding = tsne_embedding(details["estimated_normalized"], perplexity, random_state)
        label_sets = [("Empirical residual-profile cluster", details["empirical_cluster"])]
        if metadata_available:
            label_sets.append((metadata_name, item["metadata"]))
        for column_index, (label_name, labels) in enumerate(label_sets):
            ax = axes[row_index, column_index]
            unique_labels = np.unique(labels)
            for color_index, label in enumerate(unique_labels):
                keep = labels == label
                ax.scatter(
                    embedding[keep, 0],
                    embedding[keep, 1],
                    s=10,
                    alpha=0.7,
                    linewidths=0,
                    color=palette(color_index % 10),
                    label=str(label),
                )
            ax.set_title(f"{variant}\ncolor: {label_name}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if len(unique_labels) <= 10:
                ax.legend(loc="best", fontsize=6, frameon=True, title=label_name, title_fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "freshretailnet_hour_component_tsne.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "freshretailnet_hour_component_tsne.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_profiles(
    representative: dict[str, dict[str, Any]],
    variants: list[str],
    n_clusters: int,
    out_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(variants),
        n_clusters,
        figsize=(2.35 * n_clusters, 2.2 * len(variants)),
        squeeze=False,
        sharex=True,
    )
    hours = np.arange(24)
    for row_index, variant in enumerate(variants):
        details = representative[variant]["details"]
        for cluster in range(n_clusters):
            ax = axes[row_index, cluster]
            keep = details["empirical_cluster"] == cluster
            empirical = details["empirical_profile"][keep].mean(axis=0)
            estimated = details["estimated_profile"][keep].mean(axis=0)
            empirical = empirical - empirical.mean()
            estimated = estimated - estimated.mean()
            ax.axhline(0.0, color="0.8", linewidth=0.6)
            ax.plot(hours, empirical, color="0.2", linewidth=1.2, label="observed residual")
            ax.plot(hours, estimated, color="#b2473e", linewidth=1.2, linestyle="--", label="estimated hour")
            ax.set_title(f"Cluster {cluster} (n={int(keep.sum())})", fontsize=8)
            ax.set_xticks(range(0, 24, 6))
            if row_index == len(variants) - 1:
                ax.set_xlabel("Hour")
            if cluster == 0:
                ax.set_ylabel(variant, fontsize=8)
    axes[0, 0].legend(loc="best", fontsize=6, frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "freshretailnet_hour_cluster_profiles.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "freshretailnet_hour_cluster_profiles.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="scenario directory containing seed_* directories")
    parser.add_argument("--out-dir", default="figures/2-Exp-35")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["output_decomp_no_center", "output_decomp_centered"],
    )
    parser.add_argument("--representative-seed", type=int, default=17)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument(
        "--metadata-index",
        type=int,
        default=0,
        help="column in static_ids.npy; 0 is city_id in the 2-Exp-35 configs",
    )
    parser.add_argument("--metadata-name", default="city_id")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    seed_dirs = sorted(run_dir.glob("seed_*"), key=parse_seed)
    if not seed_dirs:
        raise FileNotFoundError(f"no seed_* directories found under {run_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    representative: dict[str, dict[str, Any]] = {}
    for seed_dir in seed_dirs:
        seed = parse_seed(seed_dir)
        for variant in args.variants:
            arrays = load_arrays(seed_dir / variant)
            metrics, details = evaluate_hour_component(
                arrays["hour_component"],
                arrays["residual"],
                arrays["observed"],
                n_clusters=args.clusters,
                random_state=args.random_state,
            )
            metrics.update(component_constraint_metrics(arrays))
            metrics.update(component_pairwise_correlations(arrays))
            valid_static_ids = arrays["static_ids"][details["valid"]]
            if args.metadata_index >= valid_static_ids.shape[1]:
                raise ValueError(
                    f"metadata-index {args.metadata_index} is out of range for static_ids shape {valid_static_ids.shape}"
                )
            metadata = valid_static_ids[:, args.metadata_index].astype(int)
            metadata_silhouette = safe_silhouette(details["estimated_normalized"], metadata)
            if metadata_silhouette is not None:
                metrics[f"{args.metadata_name}_silhouette_in_component_space"] = metadata_silhouette
            rows.append({"seed": seed, "variant": variant, **metrics})
            if seed == args.representative_seed:
                representative[variant] = {
                    "arrays": arrays,
                    "details": details,
                    "metadata": metadata,
                }

    missing_representative = [variant for variant in args.variants if variant not in representative]
    if missing_representative:
        available = ", ".join(str(parse_seed(path)) for path in seed_dirs)
        raise ValueError(
            f"representative seed {args.representative_seed} is unavailable for "
            f"{', '.join(missing_representative)}; available seeds: {available}"
        )

    summary = {
        "run_dir": str(run_dir),
        "reference": "KMeans labels fitted to observed residual hour profiles",
        "metrics_space": "mean-centered, L2-normalized 24-dimensional hour profiles",
        "rows": rows,
        "aggregate": numeric_summary(rows, args.variants),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_rows(out_dir / "metrics_by_seed.csv", rows)

    first = representative[args.variants[0]]
    valid_index_sets = [
        set(np.flatnonzero(representative[variant]["details"]["valid"]).tolist())
        for variant in args.variants
    ]
    valid_indices = sorted(set.intersection(*valid_index_sets))
    local_index_maps = {
        variant: {
            int(original_index): local_index
            for local_index, original_index in enumerate(
                np.flatnonzero(representative[variant]["details"]["valid"])
            )
        }
        for variant in args.variants
    }
    first_local_index = local_index_maps[args.variants[0]]
    assignments = []
    for original_index in valid_indices:
        local_index = first_local_index[original_index]
        row: dict[str, Any] = {
            "series_index": int(original_index),
            "empirical_cluster": int(first["details"]["empirical_cluster"][local_index]),
            args.metadata_name: int(first["metadata"][local_index]),
        }
        for variant in args.variants:
            details = representative[variant]["details"]
            variant_local_index = local_index_maps[variant][original_index]
            row[f"{variant}_estimated_cluster"] = int(details["estimated_cluster"][variant_local_index])
            row[f"{variant}_profile_corr"] = float(details["profile_corr"][variant_local_index])
        assignments.append(row)
    write_rows(out_dir / "representative_seed_assignments.csv", assignments)

    plot_scatter(
        representative,
        args.variants,
        args.metadata_name,
        args.tsne_perplexity,
        args.random_state,
        out_dir,
    )
    plot_cluster_profiles(representative, args.variants, args.clusters, out_dir)
    print(f"wrote 2-Exp-35 analysis to {out_dir}")


if __name__ == "__main__":
    main()
