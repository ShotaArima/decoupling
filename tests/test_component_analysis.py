from __future__ import annotations

import numpy as np

from decoupled_ts.component_analysis import component_constraint_metrics, evaluate_hour_component
from decoupled_ts.residual_experiments import save_latent_outputs


def test_evaluate_hour_component_recovers_known_profile_groups() -> None:
    rng = np.random.default_rng(17)
    hours = np.arange(24)
    prototypes = np.stack(
        [
            np.cos(2 * np.pi * (hours - peak) / 24.0)
            for peak in (6, 12, 18, 21)
        ]
    )
    labels = np.repeat(np.arange(4), 20)
    empirical_profiles = prototypes[labels] + rng.normal(0.0, 0.03, size=(80, 24))
    estimated_profiles = prototypes[labels] + rng.normal(0.0, 0.03, size=(80, 24))
    residual = np.repeat(empirical_profiles[:, None, :], 7, axis=1)
    hour_component = np.repeat(estimated_profiles[:, None, :], 7, axis=1)
    observed = np.ones_like(residual)

    metrics, details = evaluate_hour_component(
        hour_component,
        residual,
        observed,
        n_clusters=4,
        random_state=17,
    )

    assert metrics["profile_corr_median"] > 0.99
    assert metrics["cluster_recovery_ari"] > 0.99
    assert metrics["cluster_recovery_nmi"] > 0.99
    assert details["estimated_normalized"].shape == (80, 24)


def test_component_constraint_metrics_use_expected_margins() -> None:
    rng = np.random.default_rng(23)
    day = rng.normal(size=(5, 7, 1))
    day = day - day.mean(axis=1, keepdims=True)
    day = np.repeat(day, 24, axis=2)
    hour = rng.normal(size=(5, 1, 24))
    hour = hour - hour.mean(axis=2, keepdims=True)
    hour = np.repeat(hour, 7, axis=1)
    interaction = rng.normal(size=(5, 7, 24))
    interaction = interaction - interaction.mean(axis=1, keepdims=True)
    interaction = interaction - interaction.mean(axis=2, keepdims=True)

    metrics = component_constraint_metrics(
        {
            "day_component": day,
            "hour_component": hour,
            "interaction_component": interaction,
        }
    )

    assert metrics["day_center_abs_mean"] < 1e-12
    assert metrics["hour_center_abs_mean"] < 1e-12
    assert metrics["interaction_day_marginal_abs_mean"] < 1e-12
    assert metrics["interaction_hour_marginal_abs_mean"] < 1e-12


def test_component_analysis_arrays_require_explicit_output_flag(tmp_path) -> None:
    arrays = {
        "z_hour": np.ones((2, 24, 3), dtype=np.float32),
        "residual": np.ones((2, 7, 24), dtype=np.float32),
        "observed": np.ones((2, 7, 24), dtype=np.float32),
        "static_ids": np.ones((2, 7), dtype=np.int64),
    }

    without_analysis = tmp_path / "without"
    without_analysis.mkdir()
    save_latent_outputs(arrays, without_analysis, {"save_latent_arrays": True})
    assert (without_analysis / "z_hour.npy").exists()
    assert not (without_analysis / "residual.npy").exists()

    with_analysis = tmp_path / "with"
    with_analysis.mkdir()
    save_latent_outputs(
        arrays,
        with_analysis,
        {
            "save_latent_arrays": True,
            "save_component_analysis_arrays": True,
        },
    )
    assert (with_analysis / "residual.npy").exists()
    assert (with_analysis / "observed.npy").exists()
    assert (with_analysis / "static_ids.npy").exists()
