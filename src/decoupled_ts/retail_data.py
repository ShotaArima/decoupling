from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from .data import FreshRetailNetSeries, infer_input_dim


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


def build_retail_data(config: dict[str, Any]) -> RetailDataBundle:
    data_cfg = config["dataset"]
    if data_cfg["name"] in {"synthetic_retail", "structured_residual_retail"}:
        dataset_cls = SyntheticRetailSeries if data_cfg["name"] == "synthetic_retail" else StructuredResidualRetailSeries
        kwargs = {}
        if data_cfg["name"] == "structured_residual_retail":
            kwargs["residual_scale"] = float(data_cfg.get("residual_scale", 1.0))
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
        return RetailDataBundle(
            train=train,
            valid=valid,
            test=test,
            input_dim=10,
            days=int(data_cfg["days"]),
            hours=int(data_cfg.get("hours", 24)),
            forecast_days=int(data_cfg["forecast_days"]),
        )

    if data_cfg["name"] == "freshretailnet":
        merged = dict(data_cfg)
        merged["series_days"] = int(data_cfg["history_days"]) + int(data_cfg["forecast_days"])
        merged["forecast_windows"] = int(data_cfg["forecast_days"])
        train = FreshRetailNetSeries.from_config(merged, merged["train_split"], max_series=merged["max_train_series"])
        valid = FreshRetailNetSeries.from_config(merged, merged["eval_split"], max_series=merged["max_eval_series"])
        test = valid
        return RetailDataBundle(
            train=train,
            valid=valid,
            test=test,
            input_dim=infer_input_dim(merged),
            days=int(data_cfg["history_days"]),
            hours=24,
            forecast_days=int(data_cfg["forecast_days"]),
        )

    raise ValueError(f"unknown dataset name: {data_cfg['name']}")
