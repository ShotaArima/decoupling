from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .data import load_config
from .experiment_logging import append_jsonl, setup_run_logger, write_json
from .metrics import regression_metrics
from .retail_data import build_retail_data
from .retail_models import FlattenMLPForecast, RetailMultiGrainModel, covariance_penalty, flatten_to_grid
from .train import optimize_torch_runtime, resolve_device


NAIVE_VARIANTS = {"naive_last_day", "naive_recent_mean", "naive_same_hour_recent_mean"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model(config: dict[str, Any], variant: dict[str, Any], input_dim: int, days: int, hours: int) -> nn.Module:
    model_cfg = config["model"]
    if variant["type"] == "flatten_mlp":
        return FlattenMLPForecast(
            input_dim=input_dim,
            days=days,
            hours=hours,
            hidden_dim=int(model_cfg["hidden_dim"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if variant["type"] == "multigrain":
        return RetailMultiGrainModel(
            input_dim=input_dim,
            days=days,
            hours=hours,
            hidden_dim=int(model_cfg["hidden_dim"]),
            global_dim=int(model_cfg["global_dim"]),
            day_dim=int(model_cfg["day_dim"]),
            hour_dim=int(model_cfg["hour_dim"]),
            interaction_dim=int(model_cfg["interaction_dim"]),
            use_day=bool(variant.get("use_day", True)),
            use_hour=bool(variant.get("use_hour", True)),
            use_interaction=bool(variant.get("use_interaction", False)),
            forecast_days=int(config["dataset"]["forecast_days"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    raise ValueError(f"unknown variant type: {variant['type']}")


def loss_for_batch(model: nn.Module, batch: dict[str, torch.Tensor], out: dict[str, torch.Tensor], config: dict) -> dict[str, torch.Tensor]:
    target = batch["target"]
    forecast_loss = nn.functional.huber_loss(out["prediction"], target, delta=float(config["train"].get("huber_delta", 25.0)))
    total = forecast_loss
    values = {"loss_forecast": forecast_loss.detach()}
    if "reconstruction" in out:
        grid = flatten_to_grid(batch["x"])
        mask = flatten_to_grid(batch["mask"])
        recon_loss = ((out["reconstruction"] - grid).abs() * mask).sum() / mask.sum().clamp_min(1.0)
        sales_loss = ((out["sales_grid"] - grid[..., 0]).abs() * mask[..., 0]).sum() / mask[..., 0].sum().clamp_min(1.0)
        decouple = covariance_penalty([out["z_global"], out["z_day"], out["z_hour"]])
        total = (
            total
            + float(config["loss"]["reconstruction_weight"]) * recon_loss
            + float(config["loss"]["history_sales_weight"]) * sales_loss
            + float(config["loss"]["decouple_weight"]) * decouple
        )
        values.update(
            {
                "loss_reconstruction": recon_loss.detach(),
                "loss_history_sales": sales_loss.detach(),
                "loss_decouple": decouple.detach(),
            }
        )
    values["loss"] = total
    return values


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    config: dict,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    progress = tqdm(loader, desc=desc, unit="batch")
    for step, batch in enumerate(progress, start=1):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        out = model(batch["x"], batch["mask"])
        losses = loss_for_batch(model, batch, out, config)
        if training:
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"].get("grad_clip", 1.0)))
            optimizer.step()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        progress.set_postfix(loss=f"{totals['loss'] / step:.4f}")
    return {key: value / max(len(loader), 1) for key, value in totals.items()}


@torch.no_grad()
def predict_model(model: nn.Module, loader: DataLoader, forecast_days: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for batch in tqdm(loader, desc="test", unit="batch"):
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        if hasattr(model, "predict_future_sum"):
            pred = model.predict_future_sum(x, mask, forecast_days=forecast_days)
        else:
            pred = model(x, mask)["prediction"]
        preds.append(pred.detach().cpu().numpy())
        trues.append(batch["target"].numpy())
    return np.concatenate(trues), np.concatenate(preds)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, forecast_days: int, device: torch.device) -> dict[str, float]:
    y_true, y_pred = predict_model(model, loader, forecast_days, device)
    return regression_metrics(y_true, y_pred)


def _observed_sales_grid(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    grid = flatten_to_grid(batch["x"])
    mask = flatten_to_grid(batch["mask"])
    return grid[..., 0], mask[..., 0]


def predict_naive_batch(batch: dict[str, torch.Tensor], method: str, forecast_days: int, recent_days: int) -> torch.Tensor:
    sales, sales_mask = _observed_sales_grid(batch)
    observed = sales * sales_mask
    if method == "naive_last_day":
        daily = observed[:, -1, :].sum(dim=-1)
        return daily * forecast_days
    if method == "naive_recent_mean":
        tail_sales = observed[:, -recent_days:, :].sum(dim=(1, 2))
        tail_counts = sales_mask[:, -recent_days:, :].sum(dim=(1, 2)).clamp_min(1.0)
        daily_mean = tail_sales / tail_counts * sales.shape[2]
        return daily_mean * forecast_days
    if method == "naive_same_hour_recent_mean":
        tail_sales = observed[:, -recent_days:, :]
        tail_counts = sales_mask[:, -recent_days:, :].sum(dim=1).clamp_min(1.0)
        hourly_mean = tail_sales.sum(dim=1) / tail_counts
        return hourly_mean.sum(dim=-1) * forecast_days
    raise ValueError(f"unknown naive method: {method}")


@torch.no_grad()
def evaluate_naive(loader: DataLoader, method: str, forecast_days: int, recent_days: int, out_dir: Path) -> dict[str, float]:
    preds, trues = [], []
    for batch in tqdm(loader, desc=f"test {method}", unit="batch"):
        preds.append(predict_naive_batch(batch, method, forecast_days, recent_days).cpu().numpy())
        trues.append(batch["target"].numpy())
    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)
    write_predictions_csv(out_dir / "predictions.csv", y_true, y_pred)
    return regression_metrics(y_true, y_pred)


def write_predictions_csv(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "y_true", "y_pred", "error", "abs_error"])
        writer.writeheader()
        for idx, (true, pred) in enumerate(zip(y_true.tolist(), y_pred.tolist())):
            error = float(pred - true)
            writer.writerow(
                {
                    "index": idx,
                    "y_true": float(true),
                    "y_pred": float(pred),
                    "error": error,
                    "abs_error": abs(error),
                }
            )


@torch.no_grad()
def collect_latent_arrays(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    if not hasattr(model, "encode"):
        return {}
    model.eval()
    parts: dict[str, list[np.ndarray]] = {"z_global": [], "z_day": [], "z_hour": []}
    labels = []
    for batch in tqdm(loader, desc="latents", unit="batch"):
        enc = model.encode(batch["x"].to(device), batch["mask"].to(device))
        for key in parts:
            parts[key].append(enc[key].detach().cpu().numpy())
        labels.append(batch["subgroup"].numpy())
    return {key: np.concatenate(values, axis=0) for key, values in parts.items()} | {"subgroup": np.concatenate(labels, axis=0)}


def save_latent_arrays(latents: dict[str, np.ndarray], out_dir: Path) -> None:
    for key, value in latents.items():
        np.save(out_dir / f"{key}.npy", value)


def run_representation_probes(train_latents: dict[str, np.ndarray], test_latents: dict[str, np.ndarray]) -> dict[str, float]:
    if not train_latents or len(np.unique(train_latents["subgroup"])) < 2:
        return {}
    train_labels = np.unique(train_latents["subgroup"])
    test_labels = np.unique(test_latents["subgroup"])
    overlap = np.intersect1d(train_labels, test_labels)
    majority = np.bincount(train_latents["subgroup"].astype(int)).argmax()
    majority_acc = float(np.mean(test_latents["subgroup"] == majority))
    metrics = {
        "probe_subgroup_train_classes": float(len(train_labels)),
        "probe_subgroup_test_classes": float(len(test_labels)),
        "probe_subgroup_overlap_classes": float(len(overlap)),
        "probe_subgroup_majority_accuracy": majority_acc,
    }
    if len(overlap) < 2:
        return metrics
    train_zg = train_latents["z_global"].reshape(train_latents["z_global"].shape[0], -1)
    test_zg = test_latents["z_global"].reshape(test_latents["z_global"].shape[0], -1)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(train_zg, train_latents["subgroup"])
    pred = clf.predict(test_zg)
    metrics["probe_z_global_subgroup_accuracy"] = float(accuracy_score(test_latents["subgroup"], pred))
    return metrics


def audit_dataset(bundle, out_dir: Path) -> dict[str, float]:
    loader = DataLoader(bundle.train, batch_size=256, shuffle=False)
    n = 0
    sales_values = []
    target_values = []
    sales_mask_sum = 0.0
    sales_mask_count = 0.0
    subgroups = []
    for batch in loader:
        sales, sales_mask = _observed_sales_grid(batch)
        sales_values.append(sales.reshape(-1).numpy())
        target_values.append(batch["target"].numpy())
        sales_mask_sum += float(sales_mask.sum())
        sales_mask_count += float(sales_mask.numel())
        subgroups.append(batch["subgroup"].numpy())
        n += int(batch["x"].shape[0])
    sales_all = np.concatenate(sales_values)
    target_all = np.concatenate(target_values)
    subgroup_all = np.concatenate(subgroups)
    audit = {
        "train_examples": float(len(bundle.train)),
        "valid_examples": float(len(bundle.valid)),
        "test_examples": float(len(bundle.test)),
        "input_dim": float(bundle.input_dim),
        "days": float(bundle.days),
        "hours": float(bundle.hours),
        "forecast_days": float(bundle.forecast_days),
        "sales_zero_rate": float(np.mean(sales_all == 0.0)),
        "sales_observed_rate": float(sales_mask_sum / max(sales_mask_count, 1.0)),
        "target_mean": float(np.mean(target_all)),
        "target_std": float(np.std(target_all)),
        "target_p50": float(np.percentile(target_all, 50)),
        "target_p90": float(np.percentile(target_all, 90)),
        "subgroup_classes": float(len(np.unique(subgroup_all))),
    }
    write_json(out_dir / "data_audit.json", audit)
    return audit


def run_variant(config: dict[str, Any], variant: dict[str, Any], bundle, device: torch.device, root_out: Path) -> dict[str, Any]:
    variant_config = dict(config)
    if "loss_overrides" in variant:
        variant_config["loss"] = {**config["loss"], **variant["loss_overrides"]}
    name = variant["name"]
    out_dir = root_out / name
    logger = setup_run_logger(out_dir, name=f"retail.{name}")
    logger.info("Variant start: %s", name)
    logger.info("Variant config: %s", variant)
    write_json(out_dir / "config.json", {"config": variant_config, "variant": variant})

    train_cfg = config["train"]
    loader_kwargs = {
        "batch_size": int(train_cfg["batch_size"]),
        "num_workers": int(train_cfg.get("num_workers", 0)),
        "pin_memory": bool(train_cfg.get("pin_memory", False)) and device.type == "cuda",
    }
    train_loader = DataLoader(bundle.train, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(bundle.valid, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(bundle.test, shuffle=False, **loader_kwargs)
    if variant["type"] in NAIVE_VARIANTS:
        metrics = evaluate_naive(
            test_loader,
            variant["type"],
            bundle.forecast_days,
            recent_days=int(variant.get("recent_days", config["dataset"].get("recent_days", 7))),
            out_dir=out_dir,
        )
        write_json(out_dir / "metrics.json", metrics)
        logger.info("Naive variant complete: metrics=%s", metrics)
        return {"name": name, "best_valid_loss": None, **metrics}

    model = make_model(variant_config, variant, bundle.input_dim, bundle.days, bundle.hours).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    best = float("inf")
    best_path = out_dir / "best.pt"
    selection_metric = str(config["train"].get("selection_metric", "valid_loss"))
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_losses = run_epoch(model, train_loader, variant_config, device, optimizer, f"{name} train {epoch}")
        with torch.inference_mode():
            valid_losses = run_epoch(model, valid_loader, variant_config, device, None, f"{name} valid {epoch}")
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_losses.items()}, **{f"valid_{k}": v for k, v in valid_losses.items()}}
        append_jsonl(out_dir / "history.jsonl", row)
        logger.info("Epoch %d complete: %s", epoch, row)
        score = row.get(selection_metric)
        if score is None:
            raise KeyError(f"selection_metric={selection_metric!r} is not available in epoch row keys={sorted(row)}")
        if float(score) < best:
            best = float(score)
            torch.save({"model": model.state_dict(), "config": variant_config, "variant": variant}, best_path)
            logger.info("Saved best checkpoint: %s=%.5f path=%s", selection_metric, best, best_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    y_true, y_pred = predict_model(model, test_loader, bundle.forecast_days, device)
    write_predictions_csv(out_dir / "predictions.csv", y_true, y_pred)
    metrics = regression_metrics(y_true, y_pred)
    if hasattr(model, "encode"):
        train_latents = collect_latent_arrays(model, train_loader, device)
        test_latents = collect_latent_arrays(model, test_loader, device)
        save_latent_arrays(test_latents, out_dir)
        probe_metrics = run_representation_probes(train_latents, test_latents)
        metrics.update(probe_metrics)
    write_json(out_dir / "metrics.json", metrics)
    logger.info("Variant complete: metrics=%s", metrics)
    return {"name": name, "best_valid_loss": best, **metrics}


def run_retail_experiments(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    device = resolve_device(config["train"]["device"])
    optimize_torch_runtime(device)
    root_out = Path(config["train"]["output_dir"])
    root_out.mkdir(parents=True, exist_ok=True)
    logger = setup_run_logger(root_out, name="retail")
    logger.info("Retail experiment start config=%s device=%s", config_path, device)
    write_json(root_out / "resolved_config.json", config)

    bundle = build_retail_data(config)
    logger.info(
        "Data ready: train=%d valid=%d test=%d input_dim=%d days=%d hours=%d forecast_days=%d",
        len(bundle.train),
        len(bundle.valid),
        len(bundle.test),
        bundle.input_dim,
        bundle.days,
        bundle.hours,
        bundle.forecast_days,
    )
    audit = audit_dataset(bundle, root_out)
    logger.info("Data audit: %s", audit)
    results = [run_variant(config, variant, bundle, device, root_out) for variant in config["experiments"]]
    summary = {"results": results}
    write_json(root_out / "summary.json", summary)
    with (root_out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in results for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Retail experiment complete: summary=%s", summary)
    return summary
