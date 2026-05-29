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


def build_retail_data(config: dict[str, Any]) -> RetailDataBundle:
    data_cfg = config["dataset"]
    if data_cfg["name"] == "synthetic_retail":
        full = SyntheticRetailSeries(
            n_series=int(data_cfg["n_series"]),
            days=int(data_cfg["days"]),
            hours=int(data_cfg.get("hours", 24)),
            forecast_days=int(data_cfg["forecast_days"]),
            seed=int(config["seed"]),
            noise_std=float(data_cfg.get("noise_std", 0.15)),
            stockout_rate=float(data_cfg.get("stockout_rate", 0.06)),
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
