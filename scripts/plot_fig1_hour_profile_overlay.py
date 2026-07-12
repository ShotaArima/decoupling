"""Paper Fig.1: residual hour profile vs learned hour component, per baseline.

Two panels with a shared y-axis:
  (a) baseline = series mean          -> hourly structure remains, component matches
  (b) baseline = same-hour recent mean -> hourly structure is absorbed, residual is flat

Reads `profiles_by_hour.csv` produced by the 2-Exp-21 visualization run.

Usage (from the repository root):
    uv run --with matplotlib python scripts/plot_fig1_hour_profile_overlay.py \
        --run-dir runs/2-Exp-21_freshretailnet_visualization \
        --out figures_paper/fig1_hour_profile_overlay.pdf
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


PANELS = [
    ("series_mean_all", "(a) Baseline: series mean"),
    ("same_hour_recent_mean_d7_all", "(b) Baseline: same-hour recent mean (7d)"),
]


def load_profiles(csv_path: Path) -> dict[str, np.ndarray]:
    values: dict[str, dict[int, float]] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            values.setdefault(row["source"], {})[int(row["hour"])] = float(row["value"])
    hours = sorted(next(iter(values.values())).keys())
    return {source: np.array([per_hour[h] for h in hours]) for source, per_hour in values.items()}


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/2-Exp-21_freshretailnet_visualization")
    parser.add_argument("--model", default="bias_constrained_001")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", default="figures_paper/fig1_hour_profile_overlay.pdf")
    parser.add_argument("--png", action="store_true", help="also write a .png next to the pdf")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    run_dir = Path(args.run_dir)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, (target, title) in zip(axes, PANELS, strict=True):
        csv_path = run_dir / target / f"seed_{args.seed}" / args.model / "visualization" / "profiles_by_hour.csv"
        profiles = load_profiles(csv_path)
        residual = profiles["residual"]
        component = profiles["hour_component"]
        hours = np.arange(len(residual))
        corr = pearson(residual, component)

        ax.axhline(0.0, color="0.75", linewidth=0.6, zorder=0)
        ax.plot(hours, residual, color="#1f4e79", linewidth=1.4, marker="o", markersize=2.5, label="mean residual $\\bar{r}_h$")
        ax.plot(hours, component, color="#c0504d", linewidth=1.4, linestyle="--", marker="s", markersize=2.5, label="hour component $\\bar{c}_h$")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("Hour of day")
        ax.set_xticks(range(0, 24, 4))
        ax.text(
            0.03,
            0.94,
            f"corr = {corr:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", linewidth=0.6),
        )
        print(f"{target}: corr(residual, hour_component) = {corr:.4f}, residual range [{residual.min():.4f}, {residual.max():.4f}]")

    axes[0].set_ylabel("Mean residual / component")
    axes[0].legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.9)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    if args.png:
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
