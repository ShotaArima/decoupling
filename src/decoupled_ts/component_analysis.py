from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def to_hour_profile(values: np.ndarray) -> np.ndarray:
    """Convert a component grid to one hour profile per series."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3:
        return values.mean(axis=1)
    if values.ndim == 2:
        return values
    raise ValueError(f"expected shape (N, D, H) or (N, H), got {values.shape}")


def masked_hour_profile(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Compute an observed-cell mean over days for each series and hour."""
    values = np.asarray(values, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if values.shape != observed.shape or values.ndim != 3:
        raise ValueError(f"expected matching (N, D, H) arrays, got {values.shape} and {observed.shape}")
    numerator = np.sum(values * observed, axis=1)
    denominator = np.sum(observed, axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def normalize_profile_rows(values: np.ndarray, eps: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Mean-center and L2-normalize profiles, returning normalized rows and valid-row mask."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2D profile matrix, got {values.shape}")
    centered = values - values.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    valid = norm[:, 0] > eps
    normalized = np.divide(centered, norm, out=np.zeros_like(centered), where=norm > eps)
    return normalized, valid


def row_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_normalized, left_valid = normalize_profile_rows(left)
    right_normalized, right_valid = normalize_profile_rows(right)
    valid = left_valid & right_valid
    correlations = np.full(left_normalized.shape[0], np.nan, dtype=np.float64)
    correlations[valid] = np.sum(left_normalized[valid] * right_normalized[valid], axis=1)
    return correlations


def safe_silhouette(features: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    if counts.size < 2 or np.any(counts < 2) or counts.size >= len(labels):
        return None
    return float(silhouette_score(features, labels))


def evaluate_hour_component(
    hour_component: np.ndarray,
    residual: np.ndarray,
    observed: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 17,
) -> tuple[dict[str, float | int | list[int]], dict[str, np.ndarray]]:
    """Compare estimated hour profiles with independently observed residual profiles."""
    estimated = to_hour_profile(hour_component)
    empirical = masked_hour_profile(residual, observed)
    estimated_normalized, estimated_valid = normalize_profile_rows(estimated)
    empirical_normalized, empirical_valid = normalize_profile_rows(empirical)
    valid = estimated_valid & empirical_valid
    if int(valid.sum()) <= n_clusters:
        raise ValueError(f"need more than {n_clusters} non-constant profiles, got {int(valid.sum())}")

    estimated_valid_rows = estimated_normalized[valid]
    empirical_valid_rows = empirical_normalized[valid]
    empirical_clusters = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20).fit_predict(empirical_valid_rows)
    estimated_clusters = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20).fit_predict(estimated_valid_rows)
    correlations = row_correlations(empirical[valid], estimated[valid])
    aggregate_corr = row_correlations(empirical[valid].mean(axis=0, keepdims=True), estimated[valid].mean(axis=0, keepdims=True))[0]
    cluster_counts = np.bincount(empirical_clusters, minlength=n_clusters)

    metrics: dict[str, float | int | list[int]] = {
        "n_series": int(estimated.shape[0]),
        "n_valid_series": int(valid.sum()),
        "n_clusters": int(n_clusters),
        "cluster_counts": cluster_counts.astype(int).tolist(),
        "profile_corr_mean": float(np.nanmean(correlations)),
        "profile_corr_median": float(np.nanmedian(correlations)),
        "aggregate_profile_corr": float(aggregate_corr),
        "cluster_recovery_ari": float(adjusted_rand_score(empirical_clusters, estimated_clusters)),
        "cluster_recovery_nmi": float(normalized_mutual_info_score(empirical_clusters, estimated_clusters)),
    }
    silhouette = safe_silhouette(estimated_valid_rows, empirical_clusters)
    if silhouette is not None:
        metrics["empirical_cluster_silhouette_in_component_space"] = silhouette
    return metrics, {
        "valid": valid,
        "estimated_profile": estimated[valid],
        "empirical_profile": empirical[valid],
        "estimated_normalized": estimated_valid_rows,
        "empirical_normalized": empirical_valid_rows,
        "empirical_cluster": empirical_clusters,
        "estimated_cluster": estimated_clusters,
        "profile_corr": correlations,
    }


def component_constraint_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """Measure the exact output-space centering constraints used by the model."""
    metrics: dict[str, float] = {}
    if "day_component" in arrays:
        metrics["day_center_abs_mean"] = float(np.mean(np.abs(np.asarray(arrays["day_component"]).mean(axis=1))))
    if "hour_component" in arrays:
        metrics["hour_center_abs_mean"] = float(np.mean(np.abs(np.asarray(arrays["hour_component"]).mean(axis=2))))
    if "interaction_component" in arrays:
        interaction = np.asarray(arrays["interaction_component"])
        metrics["interaction_day_marginal_abs_mean"] = float(np.mean(np.abs(interaction.mean(axis=1))))
        metrics["interaction_hour_marginal_abs_mean"] = float(np.mean(np.abs(interaction.mean(axis=2))))
    return metrics


def component_pairwise_correlations(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """Report output-grid correlations as a descriptive leakage diagnostic."""
    names = [
        name
        for name in ("global_component", "day_component", "hour_component", "interaction_component")
        if name in arrays
    ]
    metrics: dict[str, float] = {}
    for left_name, right_name in combinations(names, 2):
        left = np.asarray(arrays[left_name], dtype=np.float64).reshape(-1)
        right = np.asarray(arrays[right_name], dtype=np.float64).reshape(-1)
        left = left - left.mean()
        right = right - right.mean()
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        corr = float(np.dot(left, right) / denom) if denom > 0 else 0.0
        metrics[f"component_corr_{left_name.removesuffix('_component')}_{right_name.removesuffix('_component')}"] = corr
    return metrics
