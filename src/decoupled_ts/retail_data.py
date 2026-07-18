from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .data import FreshRetailNetSeries, infer_input_dim


LOGGER = logging.getLogger(__name__)


@dataclass
class RetailDataBundle:
    train: Dataset
    valid: Dataset
    test: Dataset
    input_dim: int
    days: int
    hours: int
    forecast_days: int


class SyntheticRetailSeries(Dataset):
    """Synthetic data with known global/day/hour/interaction demand factors."""

    def __init__(
        self,
        n_series: int,
        days: int,
        hours: int,
        forecast_days: int,
        seed: int,
        noise_std: float = 0.15,
        stockout_rate: float = 0.06,
    ):
        rng = np.random.default_rng(seed)
        feature_dim = 10
        xs, masks, targets, subgroups = [], [], [], []
        hour_grid = np.arange(hours, dtype=np.float32)
        hour_sin = np.sin(2 * np.pi * hour_grid / hours)
        hour_cos = np.cos(2 * np.pi * hour_grid / hours)
        for idx in range(n_series):
            subgroup = int(rng.integers(0, 4))
            base = rng.normal(2.0 + 0.3 * subgroup, 0.35)
            amplitude = rng.uniform(0.6, 1.4)
            peak = rng.choice([8, 12, 18, 21])
            total_days = days + forecast_days
            all_sales = []
            all_features = []
            all_masks = []
            for d in range(total_days):
                weekday = d % 7
                holiday = 1.0 if weekday in (5, 6) else 0.0
                discount = float(rng.binomial(1, 0.12)) * rng.uniform(0.05, 0.35)
                weather = rng.normal(0.0, 1.0)
                day_effect = 0.25 * holiday + 1.4 * discount - 0.06 * max(weather, 0.0)
                pattern = amplitude * np.exp(-0.5 * ((hour_grid - peak) / 3.5) ** 2)
                interaction = (0.45 * discount) * np.exp(-0.5 * ((hour_grid - 13.0) / 2.5) ** 2)
                demand = np.exp(base + day_effect + pattern + interaction)
                sales = rng.poisson(np.maximum(demand / 8.0, 0.05)).astype(np.float32)
                stockout = rng.binomial(1, stockout_rate + 0.03 * (sales > np.quantile(sales, 0.75))).astype(np.float32)
                observed_sales = sales.copy()
                observed_sales[stockout == 1.0] *= rng.uniform(0.0, 0.2)
                features = np.stack(
                    [
                        observed_sales,
                        stockout,
                        np.full(hours, discount, dtype=np.float32),
                        np.full(hours, holiday, dtype=np.float32),
                        np.full(hours, weather, dtype=np.float32),
                        hour_sin,
                        hour_cos,
                        np.full(hours, np.sin(2 * np.pi * weekday / 7), dtype=np.float32),
                        np.full(hours, np.cos(2 * np.pi * weekday / 7), dtype=np.float32),
                        np.full(hours, subgroup / 3.0, dtype=np.float32),
                    ],
                    axis=0,
                )
                mask = np.ones_like(features, dtype=np.float32)
                mask[0] = 1.0 - stockout
                all_sales.append(sales)
                all_features.append(features)
                all_masks.append(mask)
            history = np.concatenate(all_features[:days], axis=1)
            history_mask = np.concatenate(all_masks[:days], axis=1)
            future_sales = np.stack(all_sales[days:], axis=0).sum(dtype=np.float32)
            xs.append(history)
            masks.append(history_mask)
            targets.append(float(future_sales))
            subgroups.append(subgroup)

        self.tensors = {
            "x": torch.from_numpy(np.stack(xs).astype(np.float32)),
            "mask": torch.from_numpy(np.stack(masks).astype(np.float32)),
            "target": torch.tensor(targets, dtype=torch.float32),
            "subgroup": torch.tensor(subgroups, dtype=torch.long),
            "static_ids": torch.tensor(subgroups, dtype=torch.long)[:, None],
        }

    def __len__(self) -> int:
        return int(self.tensors["x"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self.tensors.items()}


class StructuredResidualRetailSeries(Dataset):
    """Synthetic data where same-hour baseline is strong and residual factors are known."""

    def __init__(
        self,
        n_series: int,
        days: int,
        hours: int,
        forecast_days: int,
        seed: int,
        noise_std: float = 0.2,
        stockout_rate: float = 0.04,
        residual_scale: float = 1.0,
    ):
        rng = np.random.default_rng(seed)
        xs, masks, targets, subgroups = [], [], [], []
        hour_grid = np.arange(hours, dtype=np.float32)
        hour_sin = np.sin(2 * np.pi * hour_grid / hours)
        hour_cos = np.cos(2 * np.pi * hour_grid / hours)
        peaks = np.array([7, 12, 18, 21])
        weekday_effects = np.array([-0.25, -0.10, 0.05, 0.12, 0.24, 0.40, 0.28], dtype=np.float32)
        for _ in range(n_series):
            subgroup = int(rng.integers(0, 4))
            base_level = rng.uniform(8.0, 18.0) + subgroup * 1.2
            baseline_amp = rng.uniform(2.0, 5.0)
            baseline_peak = float(peaks[subgroup])
            residual_global = rng.normal(0.0, 0.55) + (subgroup - 1.5) * 0.25
            residual_hour_shape = np.sin(2 * np.pi * (hour_grid - peaks[subgroup]) / hours).astype(np.float32)
            residual_hour_shape += 0.55 * np.exp(-0.5 * ((hour_grid - peaks[subgroup]) / 2.3) ** 2).astype(np.float32)
            promo_shape = np.exp(-0.5 * ((hour_grid - 13.0) / 2.0) ** 2).astype(np.float32)
            evening_shape = np.exp(-0.5 * ((hour_grid - 19.0) / 2.5) ** 2).astype(np.float32)
            total_days = days + forecast_days
            all_sales = []
            all_features = []
            all_masks = []
            for d in range(total_days):
                weekday = d % 7
                holiday = 1.0 if weekday in (5, 6) else 0.0
                promo_active = float(rng.binomial(1, 0.18 + 0.10 * holiday))
                discount = promo_active * float(rng.uniform(0.08, 0.38))
                weather = float(rng.normal(0.0, 1.0))
                baseline = base_level + baseline_amp * (1.0 + np.cos(2 * np.pi * (hour_grid - baseline_peak) / hours))
                day_residual = residual_scale * (weekday_effects[weekday] + 1.7 * discount + 0.35 * holiday - 0.10 * max(weather, 0.0))
                hour_residual = residual_scale * (0.75 + 0.20 * holiday) * residual_hour_shape
                interaction = residual_scale * (3.2 * discount * promo_shape + 0.28 * holiday * evening_shape)
                residual = residual_global + day_residual + hour_residual + interaction
                sales = np.maximum(baseline + residual + rng.normal(0.0, noise_std, size=hours), 0.05).astype(np.float32)
                stockout_prob = stockout_rate + 0.04 * (sales > np.quantile(sales, 0.80))
                stockout = rng.binomial(1, np.clip(stockout_prob, 0.0, 0.35)).astype(np.float32)
                observed_sales = sales.copy()
                observed_sales[stockout == 1.0] *= rng.uniform(0.0, 0.25)
                features = np.stack(
                    [
                        observed_sales,
                        stockout,
                        np.full(hours, discount, dtype=np.float32),
                        np.full(hours, holiday, dtype=np.float32),
                        np.full(hours, weather, dtype=np.float32),
                        hour_sin,
                        hour_cos,
                        np.full(hours, np.sin(2 * np.pi * weekday / 7), dtype=np.float32),
                        np.full(hours, np.cos(2 * np.pi * weekday / 7), dtype=np.float32),
                        np.full(hours, subgroup / 3.0, dtype=np.float32),
                    ],
                    axis=0,
                )
                mask = np.ones_like(features, dtype=np.float32)
                mask[0] = 1.0 - stockout
                all_sales.append(sales)
                all_features.append(features)
                all_masks.append(mask)
            xs.append(np.concatenate(all_features[:days], axis=1))
            masks.append(np.concatenate(all_masks[:days], axis=1))
            targets.append(float(np.stack(all_sales[days:], axis=0).sum(dtype=np.float32)))
            subgroups.append(subgroup)

        self.tensors = {
            "x": torch.from_numpy(np.stack(xs).astype(np.float32)),
            "mask": torch.from_numpy(np.stack(masks).astype(np.float32)),
            "target": torch.tensor(targets, dtype=torch.float32),
            "subgroup": torch.tensor(subgroups, dtype=torch.long),
            "static_ids": torch.tensor(subgroups, dtype=torch.long)[:, None],
        }

    def __len__(self) -> int:
        return int(self.tensors["x"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self.tensors.items()}


class ComponentResidualRetailSeries(Dataset):
    """Synthetic residual data with stored global/day/hour/interaction components."""

    def __init__(
        self,
        n_series: int,
        days: int,
        hours: int,
        forecast_days: int,
        seed: int,
        noise_std: float = 0.1,
        stockout_rate: float = 0.03,
        global_scale: float = 0.5,
        day_scale: float = 0.7,
        hour_scale: float = 0.8,
        interaction_scale: float = 0.6,
    ):
        rng = np.random.default_rng(seed)
        xs, masks, targets, subgroups = [], [], [], []
        true_baseline, true_global, true_day, true_hour, true_interaction, true_residual, noisy_true_residual = [], [], [], [], [], [], []
        hour_grid = np.arange(hours, dtype=np.float32)
        hour_sin = np.sin(2 * np.pi * hour_grid / hours)
        hour_cos = np.cos(2 * np.pi * hour_grid / hours)
        weekday_effects = np.array([-0.35, -0.15, 0.05, 0.12, 0.25, 0.42, 0.30], dtype=np.float32)
        peaks = np.array([7, 12, 18, 21])
        total_days = days + forecast_days

        for _ in range(n_series):
            subgroup = int(rng.integers(0, 4))
            base_level = rng.uniform(8.0, 18.0) + 1.0 * subgroup
            baseline_amp = rng.uniform(1.5, 4.0)
            baseline_peak = float(peaks[subgroup])
            g_scalar = global_scale * (rng.normal(0.0, 0.7) + (subgroup - 1.5) * 0.25)
            raw_hour = np.sin(2 * np.pi * (hour_grid - peaks[subgroup]) / hours).astype(np.float32)
            raw_hour += 0.65 * np.exp(-0.5 * ((hour_grid - peaks[subgroup]) / 2.5) ** 2).astype(np.float32)
            hour_component = hour_scale * (raw_hour - raw_hour.mean())
            promo_shape = np.exp(-0.5 * ((hour_grid - 13.0) / 2.2) ** 2).astype(np.float32)
            promo_shape = promo_shape - promo_shape.mean()
            all_sales, all_features, all_masks = [], [], []
            all_baseline, all_g, all_day, all_hour, all_interaction, all_residual = [], [], [], [], [], []
            day_values = []
            interaction_values = []

            for d in range(total_days):
                weekday = d % 7
                holiday = 1.0 if weekday in (5, 6) else 0.0
                discount = float(rng.binomial(1, 0.18 + 0.10 * holiday)) * float(rng.uniform(0.08, 0.38))
                weather = float(rng.normal(0.0, 1.0))
                baseline = base_level + baseline_amp * (1.0 + np.cos(2 * np.pi * (hour_grid - baseline_peak) / hours))
                day_value = day_scale * (weekday_effects[weekday] + 1.5 * discount + 0.25 * holiday - 0.08 * max(weather, 0.0))
                interaction = interaction_scale * (2.8 * discount + 0.35 * holiday) * promo_shape
                day_values.append(day_value)
                interaction_values.append(interaction)
                all_baseline.append(baseline.astype(np.float32))
                all_g.append(np.full(hours, g_scalar, dtype=np.float32))
                all_day.append(np.full(hours, day_value, dtype=np.float32))
                all_hour.append(hour_component.astype(np.float32))
                all_interaction.append(interaction.astype(np.float32))

                residual = g_scalar + day_value + hour_component + interaction
                sales = np.maximum(baseline + residual + rng.normal(0.0, noise_std, size=hours), 0.05).astype(np.float32)
                stockout_prob = stockout_rate + 0.04 * (sales > np.quantile(sales, 0.80))
                stockout = rng.binomial(1, np.clip(stockout_prob, 0.0, 0.35)).astype(np.float32)
                observed_sales = sales.copy()
                observed_sales[stockout == 1.0] *= rng.uniform(0.0, 0.25)
                features = np.stack(
                    [
                        observed_sales,
                        stockout,
                        np.full(hours, discount, dtype=np.float32),
                        np.full(hours, holiday, dtype=np.float32),
                        np.full(hours, weather, dtype=np.float32),
                        hour_sin,
                        hour_cos,
                        np.full(hours, np.sin(2 * np.pi * weekday / 7), dtype=np.float32),
                        np.full(hours, np.cos(2 * np.pi * weekday / 7), dtype=np.float32),
                        np.full(hours, subgroup / 3.0, dtype=np.float32),
                    ],
                    axis=0,
                )
                mask = np.ones_like(features, dtype=np.float32)
                mask[0] = 1.0 - stockout
                all_sales.append(sales)
                all_features.append(features)
                all_masks.append(mask)

            day_center = float(np.mean(day_values[:days]))
            day_grids = [row - day_center for row in all_day[:days]]
            interaction_grid = np.stack(all_interaction[:days]).astype(np.float32)
            interaction_grid = interaction_grid - interaction_grid.mean(axis=0, keepdims=True)
            interaction_grid = interaction_grid - interaction_grid.mean(axis=1, keepdims=True)
            residual_grid = np.stack(all_g[:days]) + np.stack(day_grids) + np.stack(all_hour[:days]) + interaction_grid
            noisy_residual_grid = residual_grid + rng.normal(0.0, noise_std, size=residual_grid.shape).astype(np.float32)

            xs.append(np.concatenate(all_features[:days], axis=1))
            masks.append(np.concatenate(all_masks[:days], axis=1))
            targets.append(float(np.stack(all_sales[days:], axis=0).sum(dtype=np.float32)))
            subgroups.append(subgroup)
            true_baseline.append(np.stack(all_baseline[:days]).astype(np.float32))
            true_global.append(np.stack(all_g[:days]).astype(np.float32))
            true_day.append(np.stack(day_grids).astype(np.float32))
            true_hour.append(np.stack(all_hour[:days]).astype(np.float32))
            true_interaction.append(interaction_grid.astype(np.float32))
            true_residual.append(residual_grid.astype(np.float32))
            noisy_true_residual.append(noisy_residual_grid.astype(np.float32))

        self.tensors = {
            "x": torch.from_numpy(np.stack(xs).astype(np.float32)),
            "mask": torch.from_numpy(np.stack(masks).astype(np.float32)),
            "target": torch.tensor(targets, dtype=torch.float32),
            "subgroup": torch.tensor(subgroups, dtype=torch.long),
            "static_ids": torch.tensor(subgroups, dtype=torch.long)[:, None],
            "true_baseline": torch.from_numpy(np.stack(true_baseline).astype(np.float32)),
            "true_global": torch.from_numpy(np.stack(true_global).astype(np.float32)),
            "true_day": torch.from_numpy(np.stack(true_day).astype(np.float32)),
            "true_hour": torch.from_numpy(np.stack(true_hour).astype(np.float32)),
            "true_interaction": torch.from_numpy(np.stack(true_interaction).astype(np.float32)),
            "true_residual": torch.from_numpy(np.stack(true_residual).astype(np.float32)),
            "noisy_true_residual": torch.from_numpy(np.stack(noisy_true_residual).astype(np.float32)),
        }

    def __len__(self) -> int:
        return int(self.tensors["x"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self.tensors.items()}


def _same_hour_recent_mean_np(sales: np.ndarray, observed: np.ndarray, recent_days: int) -> np.ndarray:
    series_sum = (sales * observed).sum(axis=(1, 2), keepdims=True)
    series_count = observed.sum(axis=(1, 2), keepdims=True)
    fallback = np.divide(series_sum, np.clip(series_count, 1.0, None))
    hour_sum = (sales * observed).sum(axis=1)
    hour_count = observed.sum(axis=1)
    hour_mean = np.divide(
        hour_sum,
        np.clip(hour_count, 1.0, None),
        out=np.broadcast_to(fallback[:, 0, 0][:, None], hour_sum.shape).copy(),
        where=hour_count > 0,
    )
    baseline = np.empty_like(sales, dtype=np.float32)
    for day in range(sales.shape[1]):
        start = max(0, day - recent_days)
        if start == day:
            baseline[:, day, :] = hour_mean
            continue
        window_sales = sales[:, start:day, :] * observed[:, start:day, :]
        window_count = observed[:, start:day, :].sum(axis=1)
        day_mean = np.divide(
            window_sales.sum(axis=1),
            np.clip(window_count, 1.0, None),
            out=hour_mean.copy(),
            where=window_count > 0,
        )
        baseline[:, day, :] = day_mean
    return baseline


def _series_mean_np(sales: np.ndarray, observed: np.ndarray) -> np.ndarray:
    series_sum = (sales * observed).sum(axis=(1, 2), keepdims=True)
    series_count = observed.sum(axis=(1, 2), keepdims=True)
    return np.broadcast_to(series_sum / np.clip(series_count, 1.0, None), sales.shape).astype(np.float32)


def _weekday_same_hour_mean_np(sales: np.ndarray, observed: np.ndarray) -> np.ndarray:
    fallback = _series_mean_np(sales, observed)
    baseline = np.empty_like(sales, dtype=np.float32)
    weekday_ids = np.arange(sales.shape[1]) % 7
    for weekday in range(7):
        day_sel = weekday_ids == weekday
        if not np.any(day_sel):
            continue
        wk_sales = sales[:, day_sel, :] * observed[:, day_sel, :]
        wk_count = observed[:, day_sel, :].sum(axis=1)
        wk_mean = np.divide(
            wk_sales.sum(axis=1),
            np.clip(wk_count, 1.0, None),
            out=fallback[:, 0, :].copy(),
            where=wk_count > 0,
        )
        baseline[:, day_sel, :] = wk_mean[:, None, :]
    return baseline


def _baseline_residual_np(sales: np.ndarray, observed: np.ndarray, method: str, recent_days: int) -> tuple[np.ndarray, np.ndarray]:
    if method == "series_mean":
        baseline = _series_mean_np(sales, observed)
        return baseline, sales - baseline
    if method == "weekday_same_hour_mean":
        baseline = _weekday_same_hour_mean_np(sales, observed)
        return baseline, sales - baseline
    if method == "same_hour_recent_mean":
        baseline = _same_hour_recent_mean_np(sales, observed, recent_days)
        return baseline, sales - baseline
    if method == "log1p_same_hour_recent_mean":
        log_sales = np.log1p(np.clip(sales, 0.0, None))
        baseline_log = _same_hour_recent_mean_np(log_sales, observed, recent_days)
        return np.expm1(baseline_log).clip(0.0), log_sales - baseline_log
    raise ValueError(f"unknown subset residual baseline_method: {method}")


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or float(np.std(x)) <= 1e-8 or float(np.std(y)) <= 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _fixed_effect_reproducibility(
    residual: np.ndarray,
    observed: np.ndarray,
    group_ids: np.ndarray,
    split_day: int,
    min_count: int,
) -> np.ndarray:
    scores = np.zeros(residual.shape[0], dtype=np.float32)
    groups = np.unique(group_ids)
    train_days = np.arange(residual.shape[1]) < split_day
    valid_days = ~train_days
    for i in range(residual.shape[0]):
        train_effects = []
        valid_effects = []
        for group in groups:
            group_mask = group_ids == group
            train_mask = observed[i] > 0
            train_mask = train_mask & group_mask & train_days[:, None]
            valid_mask = observed[i] > 0
            valid_mask = valid_mask & group_mask & valid_days[:, None]
            if int(train_mask.sum()) < min_count or int(valid_mask.sum()) < min_count:
                continue
            train_effects.append(float(residual[i][train_mask].mean()))
            valid_effects.append(float(residual[i][valid_mask].mean()))
        scores[i] = max(0.0, _safe_corr(np.asarray(train_effects), np.asarray(valid_effects)))
    return scores


def _binary_effect_reproducibility(
    residual: np.ndarray,
    observed: np.ndarray,
    positive: np.ndarray,
    split_day: int,
    min_count: int,
) -> np.ndarray:
    scores = np.zeros(residual.shape[0], dtype=np.float32)
    train_days = np.arange(residual.shape[1]) < split_day
    valid_days = ~train_days
    for i in range(residual.shape[0]):
        diffs = []
        for day_mask in (train_days, valid_days):
            pos = (observed[i] > 0) & positive[i] & day_mask[:, None]
            neg = (observed[i] > 0) & (~positive[i]) & day_mask[:, None]
            if int(pos.sum()) < min_count or int(neg.sum()) < min_count:
                diffs.append(None)
            else:
                diffs.append(float(residual[i][pos].mean() - residual[i][neg].mean()))
        if diffs[0] is None or diffs[1] is None:
            continue
        train_diff = float(diffs[0])
        valid_diff = float(diffs[1])
        if abs(train_diff) <= 1e-8 or abs(valid_diff) <= 1e-8 or np.sign(train_diff) != np.sign(valid_diff):
            continue
        ratio = min(abs(valid_diff / train_diff), abs(train_diff / valid_diff))
        scores[i] = float(np.clip(ratio, 0.0, 1.0))
    return scores


def _stockout_near_mask(observed: np.ndarray) -> np.ndarray:
    # observed > 0 marks an observed cell (paper's m=1); observed <= 0 marks a
    # stockout cell.  Same convention as decoupled_ts.data.observed_from_stock.
    stockout = observed <= 0
    near = stockout.copy()
    near[:, :, 1:] |= stockout[:, :, :-1]
    near[:, :, :-1] |= stockout[:, :, 1:]
    return near


def _filter_dataset(dataset: Dataset, cfg: dict[str, Any]) -> Dataset:
    filter_cfg = cfg.get("subset_filter", {})
    if not bool(filter_cfg.get("enabled", False)):
        return dataset
    loader = DataLoader(dataset, batch_size=int(filter_cfg.get("batch_size", 512)), shuffle=False)
    rows = []
    offset = 0
    recent_days = int(filter_cfg.get("recent_days", cfg.get("recent_days", 7)))
    residual_method = str(filter_cfg.get("baseline_method", "same_hour_recent_mean"))
    validation_fraction = float(filter_cfg.get("structure_validation_fraction", 0.35))
    min_effect_count = int(filter_cfg.get("min_effect_count", 4))
    hours = int(cfg.get("hours", cfg.get("window_size", 24)))
    for batch in loader:
        x = batch["x"].numpy()
        mask = batch["mask"].numpy()
        sales = x[:, 0, :].reshape(x.shape[0], -1, hours)
        observed = mask[:, 0, :].reshape(mask.shape[0], -1, hours)
        grid = x.transpose(0, 2, 1).reshape(x.shape[0], -1, hours, x.shape[1])
        labels = _labels_from_retail_grid(grid, cfg)
        observed_count = np.clip(observed.sum(axis=(1, 2)), 1.0, None)
        observed_sales = sales * observed
        mean_sales = observed_sales.sum(axis=(1, 2)) / observed_count
        nonzero_rate = ((sales > 0.0) * observed).sum(axis=(1, 2)) / observed_count
        observed_rate = observed.mean(axis=(1, 2))
        _, residual = _baseline_residual_np(sales, observed, residual_method, recent_days)
        residual_mean = (residual * observed).sum(axis=(1, 2), keepdims=True) / observed_count[:, None, None]
        residual_std = np.sqrt((((residual - residual_mean) ** 2) * observed).sum(axis=(1, 2)) / observed_count)
        residual_abs_mean = (np.abs(residual) * observed).sum(axis=(1, 2)) / observed_count
        total_var = (((residual - residual_mean) ** 2) * observed).sum(axis=(1, 2)) / observed_count
        hour_mean = (residual * observed).sum(axis=1) / np.clip(observed.sum(axis=1), 1.0, None)
        hour_weight = observed.sum(axis=1) / observed_count[:, None]
        hour_eta = (((hour_mean - residual_mean[:, 0, 0][:, None]) ** 2) * hour_weight).sum(axis=1) / np.clip(total_var, 1e-8, None)
        day_count = sales.shape[1]
        weekday_ids = np.arange(day_count) % 7
        weekday_eta = np.zeros(x.shape[0], dtype=np.float32)
        for weekday in range(7):
            day_sel = weekday_ids == weekday
            wk_obs = observed[:, day_sel, :].sum(axis=(1, 2))
            wk_mean = (residual[:, day_sel, :] * observed[:, day_sel, :]).sum(axis=(1, 2)) / np.clip(wk_obs, 1.0, None)
            wk_weight = wk_obs / observed_count
            weekday_eta += ((wk_mean - residual_mean[:, 0, 0]) ** 2) * wk_weight
        weekday_eta = weekday_eta / np.clip(total_var, 1e-8, None)
        discount = x[:, 2, :].reshape(x.shape[0], -1, hours)
        discount_mean = (discount * observed).sum(axis=(1, 2)) / observed_count
        discount_std = np.sqrt((((discount - discount_mean[:, None, None]) ** 2) * observed).sum(axis=(1, 2)) / observed_count)
        split_day = int(np.clip(np.floor(sales.shape[1] * (1.0 - validation_fraction)), 1, sales.shape[1] - 1))
        hour_group = np.broadcast_to(np.arange(hours)[None, :], sales.shape[1:])
        weekday_group = np.broadcast_to((np.arange(sales.shape[1]) % 7)[:, None], sales.shape[1:])
        hour_repro = _fixed_effect_reproducibility(residual, observed, hour_group, split_day, min_effect_count)
        weekday_repro = _fixed_effect_reproducibility(residual, observed, weekday_group, split_day, min_effect_count)
        discount_repro = _binary_effect_reproducibility(residual, observed, discount > 0.0, split_day, min_effect_count)
        stockout_repro = _binary_effect_reproducibility(residual, observed, _stockout_near_mask(observed), split_day, min_effect_count)
        repro_score = hour_repro + weekday_repro + discount_repro + stockout_repro
        structure_score = np.asarray(hour_eta) + np.asarray(weekday_eta) + np.asarray(discount_std)
        for i in range(x.shape[0]):
            rows.append(
                {
                    "index": offset + i,
                    "mean_sales": float(mean_sales[i]),
                    "nonzero_rate": float(nonzero_rate[i]),
                    "observed_rate": float(observed_rate[i]),
                    "stockout_rate": float(1.0 - observed_rate[i]),
                    "residual_std": float(residual_std[i]),
                    "residual_abs_mean": float(residual_abs_mean[i]),
                    "residual_hour_eta": float(np.clip(hour_eta[i], 0.0, 1.0)),
                    "residual_weekday_eta": float(np.clip(weekday_eta[i], 0.0, 1.0)),
                    "discount_std": float(discount_std[i]),
                    "residual_structure_score": float(structure_score[i]),
                    "residual_hour_repro": float(hour_repro[i]),
                    "residual_weekday_repro": float(weekday_repro[i]),
                    "discount_repro": float(discount_repro[i]),
                    "stockout_repro": float(stockout_repro[i]),
                    "residual_repro_score": float(repro_score[i]),
                }
            )
        offset += x.shape[0]
    keep = []
    for row in rows:
        if row["mean_sales"] < float(filter_cfg.get("min_mean_sales", -np.inf)):
            continue
        if row["mean_sales"] > float(filter_cfg.get("max_mean_sales", np.inf)):
            continue
        if row["nonzero_rate"] < float(filter_cfg.get("min_nonzero_rate", -np.inf)):
            continue
        if row["observed_rate"] < float(filter_cfg.get("min_observed_rate", -np.inf)):
            continue
        if row["stockout_rate"] > float(filter_cfg.get("max_stockout_rate", np.inf)):
            continue
        if row["residual_std"] < float(filter_cfg.get("min_residual_std", -np.inf)):
            continue
        if row["residual_abs_mean"] < float(filter_cfg.get("min_residual_abs_mean", -np.inf)):
            continue
        if row["residual_hour_eta"] < float(filter_cfg.get("min_residual_hour_eta", -np.inf)):
            continue
        if row["residual_weekday_eta"] < float(filter_cfg.get("min_residual_weekday_eta", -np.inf)):
            continue
        if row["discount_std"] < float(filter_cfg.get("min_discount_std", -np.inf)):
            continue
        if row["residual_hour_repro"] < float(filter_cfg.get("min_residual_hour_repro", -np.inf)):
            continue
        if row["residual_weekday_repro"] < float(filter_cfg.get("min_residual_weekday_repro", -np.inf)):
            continue
        if row["discount_repro"] < float(filter_cfg.get("min_discount_repro", -np.inf)):
            continue
        if row["stockout_repro"] < float(filter_cfg.get("min_stockout_repro", -np.inf)):
            continue
        if row["residual_repro_score"] < float(filter_cfg.get("min_residual_repro_score", -np.inf)):
            continue
        keep.append(row)
    sort_key = str(filter_cfg.get("sort_by", "residual_std"))
    if sort_key:
        keep = sorted(keep, key=lambda row: row.get(sort_key, 0.0), reverse=bool(filter_cfg.get("descending", True)))
    top_k = filter_cfg.get("top_k")
    if top_k is not None:
        keep = keep[: int(top_k)]
    top_fraction = filter_cfg.get("top_fraction")
    if top_fraction is not None:
        keep = keep[: max(1, int(len(keep) * float(top_fraction)))]
    if not keep:
        raise RuntimeError(f"subset_filter removed all examples from dataset of size {len(dataset)}")
    indices = [int(row["index"]) for row in keep]
    return Subset(dataset, indices)


def _maybe_filter_bundle(bundle: RetailDataBundle, config: dict[str, Any]) -> RetailDataBundle:
    data_cfg = config["dataset"]
    if not bool(data_cfg.get("subset_filter", {}).get("enabled", False)):
        return bundle
    return RetailDataBundle(
        train=_filter_dataset(bundle.train, data_cfg),
        valid=_filter_dataset(bundle.valid, data_cfg),
        test=_filter_dataset(bundle.test, data_cfg),
        input_dim=bundle.input_dim,
        days=bundle.days,
        hours=bundle.hours,
        forecast_days=bundle.forecast_days,
    )


def build_retail_data(config: dict[str, Any]) -> RetailDataBundle:
    data_cfg = config["dataset"]
    if data_cfg["name"] in {"synthetic_retail", "structured_residual_retail", "component_residual_retail"}:
        dataset_map = {
            "synthetic_retail": SyntheticRetailSeries,
            "structured_residual_retail": StructuredResidualRetailSeries,
            "component_residual_retail": ComponentResidualRetailSeries,
        }
        dataset_cls = dataset_map[data_cfg["name"]]
        kwargs = {}
        if data_cfg["name"] == "structured_residual_retail":
            kwargs["residual_scale"] = float(data_cfg.get("residual_scale", 1.0))
        if data_cfg["name"] == "component_residual_retail":
            kwargs.update(
                {
                    "global_scale": float(data_cfg.get("global_scale", 0.5)),
                    "day_scale": float(data_cfg.get("day_scale", 0.7)),
                    "hour_scale": float(data_cfg.get("hour_scale", 0.8)),
                    "interaction_scale": float(data_cfg.get("interaction_scale", 0.6)),
                }
            )
        full = dataset_cls(
            n_series=int(data_cfg["n_series"]),
            days=int(data_cfg["days"]),
            hours=int(data_cfg.get("hours", 24)),
            forecast_days=int(data_cfg["forecast_days"]),
            seed=int(config["seed"]),
            noise_std=float(data_cfg.get("noise_std", 0.15)),
            stockout_rate=float(data_cfg.get("stockout_rate", 0.06)),
            **kwargs,
        )
        valid_size = int(len(full) * float(data_cfg.get("valid_ratio", 0.15)))
        test_size = int(len(full) * float(data_cfg.get("test_ratio", 0.15)))
        train_size = len(full) - valid_size - test_size
        generator = torch.Generator().manual_seed(int(config["seed"]))
        train, valid, test = random_split(full, [train_size, valid_size, test_size], generator=generator)
        return _maybe_filter_bundle(RetailDataBundle(
            train=train,
            valid=valid,
            test=test,
            input_dim=10,
            days=int(data_cfg["days"]),
            hours=int(data_cfg.get("hours", 24)),
            forecast_days=int(data_cfg["forecast_days"]),
        ), config)

    if data_cfg["name"] == "freshretailnet":
        merged = dict(data_cfg)
        merged["series_days"] = int(data_cfg["history_days"]) + int(data_cfg["forecast_days"])
        merged["forecast_windows"] = int(data_cfg["forecast_days"])
        train_split = str(merged["train_split"])
        eval_split = str(merged["eval_split"])
        max_train_series = int(merged["max_train_series"])
        max_eval_series = int(merged["max_eval_series"])

        validation_source = str(merged.get("validation_source", "eval"))
        if validation_source == "train_holdout":
            max_valid_series = int(merged.get("max_valid_series", max_eval_series))
            train_offset = int(merged.get("series_start_offset", 0))
            valid_offset = int(merged.get("validation_series_start_offset", train_offset + max_train_series))
            train_end = train_offset + max_train_series
            valid_end = valid_offset + max_valid_series
            if max_valid_series <= 0:
                raise ValueError("max_valid_series must be positive for validation_source=train_holdout")
            if max(train_offset, valid_offset) < min(train_end, valid_end):
                raise ValueError("FreshRetailNet train and validation series ranges overlap")
            pool_offset = min(train_offset, valid_offset)
            pool_end = max(train_end, valid_end)
            pool_config = dict(merged)
            pool_config["series_start_offset"] = pool_offset
            train_pool = FreshRetailNetSeries.from_config(pool_config, train_split, max_series=pool_end - pool_offset)
            if len(train_pool) < pool_end - pool_offset:
                raise RuntimeError("FreshRetailNet train split does not contain enough series for the requested train/validation ranges")
            train = Subset(train_pool, range(train_offset - pool_offset, train_end - pool_offset))
            valid = Subset(train_pool, range(valid_offset - pool_offset, valid_end - pool_offset))
            test = FreshRetailNetSeries.from_config(merged, eval_split, max_series=max_eval_series)
            LOGGER.info(
                "FreshRetailNet split: train=%s[%d:%d] valid=%s[%d:%d] test=%s[:%d]",
                train_split,
                train_offset,
                train_end,
                train_split,
                valid_offset,
                valid_end,
                eval_split,
                max_eval_series,
            )
        elif validation_source == "eval":
            train = FreshRetailNetSeries.from_config(merged, train_split, max_series=max_train_series)
            valid = FreshRetailNetSeries.from_config(merged, eval_split, max_series=max_eval_series)
            test = valid
            LOGGER.warning("FreshRetailNet validation and test both use split=%s", eval_split)
        else:
            raise ValueError(f"unknown FreshRetailNet validation_source: {validation_source}")
        return _maybe_filter_bundle(RetailDataBundle(
            train=train,
            valid=valid,
            test=test,
            input_dim=infer_input_dim(merged),
            days=int(data_cfg["history_days"]),
            hours=24,
            forecast_days=int(data_cfg["forecast_days"]),
        ), config)

    raise ValueError(f"unknown dataset name: {data_cfg['name']}")
