"""Paper Fig.2: subgroup-wise recovery of the true hour component on synthetic data.

For each subgroup (peak hour 7 / 12 / 18 / 21), overlays the true hour component
profile with the estimated hour component averaged over the test series of that
subgroup. Optionally appends a t-SNE panel of per-series estimated hour-component
profiles colored by subgroup (output-space analogue of the t-SNE evidence in
"Decoupling Local and Global Representations of Time Series").

Expects the .npy arrays saved by a 2-Exp-29 run with `save_latent_arrays: true`:
    hour_component.npy  (N, D, H)  estimated hour component
    true_hour.npy       (N, D, H)  true hour component
    subgroup.npy        (N,)       subgroup index 0..3

Usage (from the repository root):
    uv run --with matplotlib python scripts/plot_fig2_component_recovery.py \
        --variant-dir runs/2-Exp-29_synthetic_component_recovery_figure/base/seed_17/output_decomp_centered \
        --out figures_paper/fig2_component_recovery.pdf --tsne
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PEAK_HOURS = {0: 7, 1: 12, 2: 18, 3: 21}


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def to_series_profile(values: np.ndarray) -> np.ndarray:
    """(N, D, H) or (N, H) -> per-series hour profile (N, H)."""
    if values.ndim == 3:
        return values.mean(axis=1)
    if values.ndim == 2:
        return values
    raise ValueError(f"unexpected shape {values.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant-dir",
        default="runs/2-Exp-29_synthetic_component_recovery_figure/base/seed_17/output_decomp_centered",
    )
    parser.add_argument("--out", default="figures_paper/fig2_component_recovery.pdf")
    parser.add_argument("--tsne", action="store_true", help="append a t-SNE panel of estimated profiles")
    parser.add_argument("--png", action="store_true", help="also write a .png next to the pdf")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
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

    variant_dir = Path(args.variant_dir)
    est = to_series_profile(np.load(variant_dir / "hour_component.npy"))
    true = to_series_profile(np.load(variant_dir / "true_hour.npy"))
    subgroup = np.load(variant_dir / "subgroup.npy").astype(int)
    if not (est.shape == true.shape and est.shape[0] == subgroup.shape[0]):
        raise ValueError(f"shape mismatch: est={est.shape} true={true.shape} subgroup={subgroup.shape}")
    hours = np.arange(est.shape[1])

    groups = sorted(np.unique(subgroup).tolist())
    n_panels = len(groups) + (1 if args.tsne else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(1.75 * n_panels, 2.0), sharey=False)
    axes = np.atleast_1d(axes)

    colors = ["#1f4e79", "#c0504d", "#4f8f4f", "#8064a2"]
    for panel, g in enumerate(groups):
        ax = axes[panel]
        keep = subgroup == g
        true_mean = true[keep].mean(axis=0)
        est_mean = est[keep].mean(axis=0)
        profile_corr = pearson(true_mean, est_mean)
        per_series_corr = np.median([pearson(true[i], est[i]) for i in np.where(keep)[0]])

        ax.axhline(0.0, color="0.75", linewidth=0.6, zorder=0)
        ax.plot(hours, true_mean, color="0.25", linewidth=1.4, label="true $c_h$")
        ax.plot(hours, est_mean, color=colors[g % len(colors)], linewidth=1.4, linestyle="--", label="estimated $\\hat{c}_h$")
        peak = PEAK_HOURS.get(g)
        title = f"Subgroup {g}" + (f" (peak {peak}:00)" if peak is not None else "")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Hour of day")
        ax.set_xticks(range(0, 24, 6))
        if panel == 0:
            ax.set_ylabel("Hour component")
        ax.text(
            0.03,
            0.95,
            f"corr = {profile_corr:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.7", linewidth=0.5),
        )
        print(
            f"subgroup {g} (n={int(keep.sum())}): profile corr = {profile_corr:.4f}, "
            f"median per-series corr = {per_series_corr:.4f}"
        )
    axes[0].legend(loc="lower right", fontsize=6.5, frameon=True, framealpha=0.9)

    if args.tsne:
        from sklearn.manifold import TSNE

        perplexity = min(args.tsne_perplexity, max(5.0, (est.shape[0] - 1) / 3.0))
        emb = TSNE(n_components=2, perplexity=perplexity, random_state=17, init="pca").fit_transform(est)
        ax = axes[-1]
        for g in groups:
            keep = subgroup == g
            ax.scatter(
                emb[keep, 0],
                emb[keep, 1],
                s=4,
                color=colors[g % len(colors)],
                label=f"sg {g}",
                alpha=0.7,
                linewidths=0,
            )
        ax.set_title("t-SNE of $\\hat{c}$ profiles", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best", fontsize=6, frameon=True, framealpha=0.9, handletextpad=0.2)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    if args.png:
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
