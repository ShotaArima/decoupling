from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SeriesExample:
    x: np.ndarray
    mask: np.ndarray
    static_ids: np.ndarray
    target: float
    subgroup: int


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_freshretailnet_frame(cfg: dict[str, Any], split: str) -> pd.DataFrame:
    data_dir = Path(cfg["data_dir"])
    candidates = [
        data_dir / f"{split}.parquet",
        data_dir / f"{split}.csv",
        data_dir / split / "data.parquet",
        data_dir / split / "data.csv",
    ]
    for path in candidates:
        if path.exists():
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Dataset files were not found locally and `datasets` is not installed. "
            "Run `uv sync` or place train/eval parquet files under data/freshretailnet."
        ) from exc
    return load_dataset(cfg["hf_name"], split=split).to_pandas()


def _as_hours(value: Any) -> list[float]:
    if isinstance(value, str):
        parsed = json.loads(value)
        return [float(v) for v in parsed]
    if isinstance(value, Iterable):
        return [float(v) for v in value]
    raise ValueError(f"cannot parse hourly list: {value!r}")


def _time_features(day_index: int) -> np.ndarray:
    hours = np.arange(24, dtype=np.float32)
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)
    dow = day_index % 7
    dow_sin = np.full(24, np.sin(2 * np.pi * dow / 7.0), dtype=np.float32)
    dow_cos = np.full(24, np.cos(2 * np.pi * dow / 7.0), dtype=np.float32)
    return np.stack([hour_sin, hour_cos, dow_sin, dow_cos], axis=0)


def frame_to_examples(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    max_series: int | None = None,
    target_df: pd.DataFrame | None = None,
) -> list[SeriesExample]:
    sample_cols = cfg["sample_id_columns"]
    static_cols = cfg["static_id_columns"]
    numeric_cols = cfg["daily_numeric_columns"]
    sales_col = cfg["hourly_sales_column"]
    stock_col = cfg["hourly_stock_column"]
    stockout_value = float(cfg["stockout_value"])
    series_days = int(cfg["series_days"])
    forecast_windows = int(cfg.get("forecast_windows", 2))
    history_days = series_days - forecast_windows

    examples: list[SeriesExample] = []
    grouped = df.sort_values(sample_cols + ["dt"]).groupby(sample_cols, sort=False)
    target_groups = None
    if target_df is not None:
        target_groups = {
            key: group.sort_values("dt")
            for key, group in target_df.groupby(sample_cols, sort=False)
            if len(group) >= forecast_windows
        }

    for key, group in grouped:
        needed_days = history_days if target_groups is None else history_days
        if len(group) < needed_days:
            continue
        if target_groups is None:
            if len(group) < series_days:
                continue
            source_group = group.iloc[:series_days]
            history_group = source_group.iloc[:history_days]
            target_group = source_group.iloc[history_days:series_days]
        else:
            lookup_key = key if isinstance(key, tuple) else (key,)
            target_group = target_groups.get(lookup_key)
            if target_group is None:
                continue
            history_group = group.iloc[-history_days:]
            target_group = target_group.iloc[:forecast_windows]
        channels = []
        masks = []
        sale = np.stack([_as_hours(v) for v in history_group[sales_col]], axis=0).astype(np.float32)
        stock = np.stack([_as_hours(v) for v in history_group[stock_col]], axis=0).astype(np.float32)
        sale_obs = (stock != stockout_value).astype(np.float32)
        channels.append(sale.reshape(-1))
        masks.append(sale_obs.reshape(-1))
        channels.append(stock.reshape(-1))
        masks.append(np.ones(history_days * 24, dtype=np.float32))

        for col in numeric_cols:
            repeated = np.repeat(history_group[col].astype(np.float32).to_numpy(), 24)
            channels.append(repeated)
            masks.append(np.ones(history_days * 24, dtype=np.float32))

        time_feats = [_time_features(i) for i in range(history_days)]
        for feat in np.concatenate(time_feats, axis=1):
            channels.append(feat.astype(np.float32))
            masks.append(np.ones(history_days * 24, dtype=np.float32))

        x = np.stack(channels, axis=0)
        mask = np.stack(masks, axis=0)
        static_ids = history_group.iloc[0][static_cols].astype(np.int64).to_numpy()
        target_sale = np.stack([_as_hours(v) for v in target_group[sales_col]], axis=0).astype(np.float32)
        target = target_sale.sum(dtype=np.float32)
        subgroup = int(history_group.iloc[0][cfg.get("subgroup_target", "city_id")])
        examples.append(
            SeriesExample(
                x=x,
                mask=mask,
                static_ids=static_ids,
                target=float(target),
                subgroup=subgroup,
            )
        )
        if max_series is not None and len(examples) >= max_series:
            break
    return examples


class FreshRetailNetSeries(Dataset):
    def __init__(self, examples: list[SeriesExample]):
        self.examples = examples

    @classmethod
    def from_config(cls, cfg: dict[str, Any], split: str, max_series: int | None = None):
        merged = dict(cfg)
        merged["subgroup_target"] = cfg.get("subgroup_target", "city_id")
        if split == cfg.get("eval_split", "eval"):
            history = load_freshretailnet_frame(cfg, cfg.get("train_split", "train"))
            target = load_freshretailnet_frame(cfg, split)
            return cls(frame_to_examples(history, merged, max_series=max_series, target_df=target))
        df = load_freshretailnet_frame(cfg, split)
        return cls(frame_to_examples(df, merged, max_series=max_series))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "x": torch.from_numpy(ex.x),
            "mask": torch.from_numpy(ex.mask),
            "static_ids": torch.from_numpy(ex.static_ids),
            "target": torch.tensor(ex.target, dtype=torch.float32),
            "subgroup": torch.tensor(ex.subgroup, dtype=torch.long),
        }


def infer_input_dim(cfg: dict[str, Any]) -> int:
    return 2 + len(cfg["daily_numeric_columns"]) + 4
