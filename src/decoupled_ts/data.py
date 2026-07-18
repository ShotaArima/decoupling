"""Data loading and tensorisation for FreshRetailNet-style hourly retail data.

Terminology for stockout handling (paper <-> code):

    paper            code                                     meaning
    ---------------  ---------------------------------------  ------------------------------------
    m_{i,d,h} = 1    ``observed = 1`` (``mask`` channel 0)    observed cell (contributes to loss)
    m_{i,d,h} = 0    ``observed = 0`` /                       stockout cell (excluded from loss)
                     ``stock_status == stockout_value``

The binary observation indicator is called ``observed`` throughout the code
base and equals the paper's observation mask ``m_{i,d,h}``.  ``mask`` refers
to the multi-channel mask tensor whose channel 0 (the sales channel) is
``observed``; the remaining channels are constant 1.  ``stock_status`` is the
raw stock flag and ``stockout_value`` (=1 for FreshRetailNet) is the value
that marks a stockout.  Use :func:`observed_from_stock` to derive
``observed`` from the raw flag instead of comparing against
``stockout_value`` inline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


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
            LOGGER.info("Loading FreshRetailNet split=%s from %s", split, path)
            if path.suffix == ".parquet":
                frame = pd.read_parquet(path)
            else:
                frame = pd.read_csv(path)
            LOGGER.info("Loaded split=%s rows=%d columns=%d", split, len(frame), len(frame.columns))
            return frame

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Dataset files were not found locally and `datasets` is not installed. "
            "Run `uv sync` or place train/eval parquet files under data/freshretailnet."
        ) from exc
    LOGGER.info("Loading FreshRetailNet split=%s from Hugging Face dataset=%s", split, cfg["hf_name"])
    frame = load_dataset(cfg["hf_name"], split=split).to_pandas()
    LOGGER.info("Loaded split=%s rows=%d columns=%d", split, len(frame), len(frame.columns))
    return frame


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


def observed_from_stock(stock: np.ndarray, stockout_value: float) -> np.ndarray:
    """Derive the observation indicator ``observed`` from the raw stock flag.

    Beware the counter-intuitive polarity: ``stock_status == stockout_value``
    (=1 for FreshRetailNet) marks a *stockout* hour, i.e. the sale is NOT
    observable.  The returned array is therefore 1.0 for observed cells and
    0.0 for stockout cells, and corresponds one-to-one to the paper's
    observation mask ``m_{i,d,h}`` (m=1: observed, contributes to the loss;
    m=0: stockout, excluded from the loss).
    """
    return (np.asarray(stock) != stockout_value).astype(np.float32)


def _aggregate_duplicate_days(group: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if not group["dt"].duplicated().any():
        return group.sort_values("dt")
    sales_col = cfg["hourly_sales_column"]
    stock_col = cfg["hourly_stock_column"]
    stockout_value = float(cfg["stockout_value"])
    numeric_cols = cfg["daily_numeric_columns"]
    rows = []
    for _, day_group in group.sort_values("dt").groupby("dt", sort=False):
        row = day_group.iloc[0].copy()
        sales = np.stack([_as_hours(v) for v in day_group[sales_col]], axis=0).astype(np.float32)
        stock = np.stack([_as_hours(v) for v in day_group[stock_col]], axis=0).astype(np.float32)
        row[sales_col] = sales.sum(axis=0).astype(float).tolist()
        # Aggregated demand is observable at an hour if at least one member series is observable.
        observed_any = observed_from_stock(stock, stockout_value).astype(bool).any(axis=0)
        row[stock_col] = np.where(observed_any, 0.0, stockout_value).astype(float).tolist()
        for col in numeric_cols:
            if col in day_group:
                row[col] = float(day_group[col].astype(np.float32).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dt")


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
    series_start_offset = int(cfg.get("series_start_offset", 0))

    examples: list[SeriesExample] = []
    skipped_examples = 0
    grouped = df.sort_values(sample_cols + ["dt"]).groupby(sample_cols, sort=False)
    LOGGER.info(
        "Converting rows to tensors: groups=%d history_days=%d forecast_days=%d target_from_eval=%s start_offset=%d",
        grouped.ngroups,
        history_days,
        forecast_windows,
        target_df is not None,
        series_start_offset,
    )
    target_groups = None
    if target_df is not None:
        LOGGER.info("Indexing eval targets by %s", ",".join(sample_cols))
        target_groups = {}
        for key, group in target_df.groupby(sample_cols, sort=False):
            lookup_key = key if isinstance(key, tuple) else (key,)
            if len(group) >= forecast_windows:
                target_groups[lookup_key] = _aggregate_duplicate_days(group, cfg)
        LOGGER.info("Indexed eval target groups=%d", len(target_groups))

    iterator = tqdm(grouped, total=grouped.ngroups, desc="build-series", unit="series")
    for key, group in iterator:
        group = _aggregate_duplicate_days(group, cfg)
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
        observed = observed_from_stock(stock, stockout_value)
        channels.append(sale.reshape(-1))
        # Mask channel 0 (sales channel) is the paper's observation mask
        # m_{i,d,h}; all remaining mask channels are constant 1.
        masks.append(observed.reshape(-1))
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
        if skipped_examples < series_start_offset:
            skipped_examples += 1
            continue
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
        if len(examples) and len(examples) % 1000 == 0:
            iterator.set_postfix(examples=len(examples))
    LOGGER.info("Built series examples=%d", len(examples))
    return examples


class FreshRetailNetSeries(Dataset):
    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.tensors = tensors

    @classmethod
    def from_config(cls, cfg: dict[str, Any], split: str, max_series: int | None = None):
        merged = dict(cfg)
        merged["subgroup_target"] = cfg.get("subgroup_target", "city_id")
        cache_path = _cache_path(merged, split, max_series)
        if bool(merged.get("use_cache", True)) and cache_path.exists():
            LOGGER.info("Loading preprocessed tensor cache from %s", cache_path)
            return cls(torch.load(cache_path, map_location="cpu", weights_only=False))
        if split == cfg.get("eval_split", "eval"):
            LOGGER.info("Building eval dataset from train history plus eval target split")
            history = load_freshretailnet_frame(cfg, cfg.get("train_split", "train"))
            target = load_freshretailnet_frame(cfg, split)
            examples = frame_to_examples(history, merged, max_series=max_series, target_df=target)
        else:
            df = load_freshretailnet_frame(cfg, split)
            examples = frame_to_examples(df, merged, max_series=max_series)
        tensors = _examples_to_tensors(examples)
        if bool(merged.get("use_cache", True)):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Saving preprocessed tensor cache to %s", cache_path)
            torch.save(tensors, cache_path)
        return cls(tensors)

    def __len__(self) -> int:
        return int(self.tensors["x"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.tensors["x"][idx],
            "mask": self.tensors["mask"][idx],
            "static_ids": self.tensors["static_ids"][idx],
            "target": self.tensors["target"][idx],
            "subgroup": self.tensors["subgroup"][idx],
        }


def infer_input_dim(cfg: dict[str, Any]) -> int:
    return 2 + len(cfg["daily_numeric_columns"]) + 4


def _examples_to_tensors(examples: list[SeriesExample]) -> dict[str, torch.Tensor]:
    if not examples:
        raise RuntimeError("No FreshRetailNet series examples were built. Check split names and series_days.")
    return {
        "x": torch.from_numpy(np.stack([e.x for e in examples]).astype(np.float32)),
        "mask": torch.from_numpy(np.stack([e.mask for e in examples]).astype(np.float32)),
        "static_ids": torch.from_numpy(np.stack([e.static_ids for e in examples]).astype(np.int64)),
        "target": torch.tensor([e.target for e in examples], dtype=torch.float32),
        "subgroup": torch.tensor([e.subgroup for e in examples], dtype=torch.long),
    }


def _cache_path(cfg: dict[str, Any], split: str, max_series: int | None) -> Path:
    cache_dir = Path(cfg.get("cache_dir") or Path(cfg["data_dir"]) / "cache")
    target_mode = "eval-target" if split == cfg.get("eval_split", "eval") else "self-target"
    limit = "all" if max_series is None else str(max_series)
    days = f"d{cfg['series_days']}_w{cfg['window_size']}_f{cfg.get('forecast_windows', 2)}"
    subgroup = cfg.get("subgroup_target", "city_id")
    sample = "-".join(str(col) for col in cfg.get("sample_id_columns", ["sample"]))
    offset = int(cfg.get("series_start_offset", 0))
    offset_tag = "" if offset == 0 else f"_o{offset}"
    return cache_dir / f"{split}_{target_mode}_{days}_{subgroup}_{sample}_{limit}{offset_tag}.pt"
