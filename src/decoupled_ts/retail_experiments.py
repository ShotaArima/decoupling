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
from sklearn.linear_model import LogisticRegression, Ridge
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
            forecast_activation=str(variant.get("forecast_activation", model_cfg.get("forecast_activation", "softplus"))),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    raise ValueError(f"unknown variant type: {variant['type']}")


def loss_for_batch(model: nn.Module, batch: dict[str, torch.Tensor], out: dict[str, torch.Tensor], config: dict) -> dict[str, torch.Tensor]:
    target = batch["target"]
    pred = final_prediction(out, batch, config)
    loss_target = training_target(batch, config)
    loss_pred = out["prediction"] if prediction_mode(config) == "residual_target" else pred
    forecast_loss = nn.functional.huber_loss(loss_pred, loss_target, delta=float(config["train"].get("huber_delta", 25.0)))
    total = forecast_loss
    values = {"loss_forecast": forecast_loss.detach()}
    if "reconstruction" in out:
        grid = flatten_to_grid(batch["x"])
        mask = flatten_to_grid(batch["mask"])
        weights = reconstruction_weights(mask, config)
        recon_loss = ((out["reconstruction"] - grid).abs() * weights).sum() / weights.sum().clamp_min(1.0)
        sales_weights = weights[..., 0]
        sales_loss = ((out["sales_grid"] - grid[..., 0]).abs() * sales_weights).sum() / sales_weights.sum().clamp_min(1.0)
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


def prediction_mode(config: dict) -> str:
    return str(config.get("prediction", {}).get("mode", "direct"))


def residual_baseline(batch: dict[str, torch.Tensor], config: dict) -> torch.Tensor:
    pred_cfg = config.get("prediction", {})
    method = str(pred_cfg.get("baseline_method", config.get("dataset", {}).get("baseline_method", "naive_same_hour_recent_mean")))
    recent_days = int(pred_cfg.get("recent_days", config.get("dataset", {}).get("recent_days", 7)))
    forecast_days = int(config["dataset"]["forecast_days"])
    return predict_naive_batch(batch, method, forecast_days, recent_days).to(batch["target"].device)


def final_prediction(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], config: dict) -> torch.Tensor:
    mode = prediction_mode(config)
    if mode == "direct":
        pred = out["prediction"]
    elif mode in {"residual_additive", "residual_target"}:
        pred = residual_baseline(batch, config) + out["prediction"]
    else:
        raise ValueError(f"unknown prediction mode: {mode}")
    if bool(config.get("prediction", {}).get("nonnegative", True)):
        pred = pred.clamp_min(0.0)
    return pred


def training_target(batch: dict[str, torch.Tensor], config: dict) -> torch.Tensor:
    if prediction_mode(config) == "residual_target":
        return batch["target"] - residual_baseline(batch, config)
    return batch["target"]


def reconstruction_weights(mask: torch.Tensor, config: dict) -> torch.Tensor:
    """Weight censored stockout sales in reconstruction-style auxiliary losses."""
    mode = str(config.get("loss", {}).get("stockout_weighting", "mask"))
    if mode == "mask":
        return mask
    if mode == "uniform":
        return torch.ones_like(mask)
    if mode == "soft":
        stockout_weight = float(config.get("loss", {}).get("stockout_weight", 0.1))
        return torch.where(mask > 0.0, torch.ones_like(mask), torch.full_like(mask, stockout_weight))
    raise ValueError(f"unknown stockout_weighting mode: {mode}")


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
def predict_model(model: nn.Module, loader: DataLoader, forecast_days: int, device: torch.device, config: dict) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for batch in tqdm(loader, desc="test", unit="batch"):
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        if hasattr(model, "predict_future_sum"):
            raw_pred = model.predict_future_sum(x, mask, forecast_days=forecast_days)
            pred = final_prediction({"prediction": raw_pred}, batch | {"x": x, "mask": mask, "target": batch["target"].to(device)}, config)
        else:
            out = model(x, mask)
            pred = final_prediction(out, batch | {"x": x, "mask": mask, "target": batch["target"].to(device)}, config)
        preds.append(pred.detach().cpu().numpy())
        trues.append(batch["target"].numpy())
    return np.concatenate(trues), np.concatenate(preds)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, forecast_days: int, device: torch.device, config: dict) -> dict[str, float]:
    y_true, y_pred = predict_model(model, loader, forecast_days, device, config)
    return regression_metrics(y_true, y_pred)


@torch.no_grad()
def validation_prediction_metrics(model: nn.Module, loader: DataLoader, forecast_days: int, device: torch.device, config: dict) -> dict[str, float]:
    y_true, y_pred = predict_model(model, loader, forecast_days, device, config)
    return {f"valid_{key}": value for key, value in regression_metrics(y_true, y_pred).items()}


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
def collect_latent_arrays(model: nn.Module, loader: DataLoader, device: torch.device, config: dict | None = None) -> dict[str, np.ndarray]:
    if not hasattr(model, "encode"):
        return {}
    model.eval()
    parts: dict[str, list[np.ndarray]] = {"z_global": [], "z_day": [], "z_hour": []}
    labels = []
    day_weekday = []
    day_holiday = []
    day_discount = []
    for batch in tqdm(loader, desc="latents", unit="batch"):
        x = batch["x"].to(device)
        enc = model.encode(x, batch["mask"].to(device))
        for key in parts:
            parts[key].append(enc[key].detach().cpu().numpy())
        labels.append(batch["subgroup"].numpy())
        if config is not None:
            label_dict = extract_day_probe_labels(x.detach().cpu(), config)
            day_weekday.append(label_dict["weekday"])
            day_holiday.append(label_dict["holiday"])
            day_discount.append(label_dict["discount"])
    arrays = {key: np.concatenate(values, axis=0) for key, values in parts.items()} | {"subgroup": np.concatenate(labels, axis=0)}
    if day_weekday:
        arrays["day_weekday"] = np.concatenate(day_weekday, axis=0)
        arrays["day_holiday"] = np.concatenate(day_holiday, axis=0)
        arrays["day_discount"] = np.concatenate(day_discount, axis=0)
    return arrays


def extract_day_probe_labels(x: torch.Tensor, config: dict) -> dict[str, np.ndarray]:
    grid = flatten_to_grid(x)
    data_cfg = config["dataset"]
    if data_cfg["name"] == "synthetic_retail":
        discount_idx = 2
        holiday_idx = 3
        dow_sin_idx = 7
        dow_cos_idx = 8
    else:
        numeric_cols = list(data_cfg["daily_numeric_columns"])
        discount_idx = 2 + numeric_cols.index("discount") if "discount" in numeric_cols else None
        holiday_idx = 2 + numeric_cols.index("holiday_flag") if "holiday_flag" in numeric_cols else None
        dow_sin_idx = 2 + len(numeric_cols) + 2
        dow_cos_idx = 2 + len(numeric_cols) + 3
    day_grid = grid[:, :, 0, :]
    dow_angle = torch.atan2(day_grid[..., dow_sin_idx], day_grid[..., dow_cos_idx])
    weekday = torch.round((dow_angle.remainder(2 * torch.pi)) / (2 * torch.pi) * 7).long().remainder(7)
    holiday = torch.zeros_like(weekday) if holiday_idx is None else (day_grid[..., holiday_idx] > 0.5).long()
    discount = torch.zeros_like(day_grid[..., 0]) if discount_idx is None else day_grid[..., discount_idx].float()
    return {
        "weekday": weekday.numpy(),
        "holiday": holiday.numpy(),
        "discount": discount.numpy(),
    }


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
    metrics.update(run_day_probes(train_latents, test_latents))
    return metrics


def run_day_probes(train_latents: dict[str, np.ndarray], test_latents: dict[str, np.ndarray]) -> dict[str, float]:
    needed = {"z_day", "day_weekday", "day_holiday", "day_discount"}
    if not needed.issubset(train_latents) or not needed.issubset(test_latents):
        return {}
    train_z = train_latents["z_day"].reshape(-1, train_latents["z_day"].shape[-1])
    test_z = test_latents["z_day"].reshape(-1, test_latents["z_day"].shape[-1])
    train_weekday = train_latents["day_weekday"].reshape(-1)
    test_weekday = test_latents["day_weekday"].reshape(-1)
    train_holiday = train_latents["day_holiday"].reshape(-1)
    test_holiday = test_latents["day_holiday"].reshape(-1)
    train_discount = train_latents["day_discount"].reshape(-1)
    test_discount = test_latents["day_discount"].reshape(-1)
    metrics: dict[str, float] = {}
    if len(np.unique(train_weekday)) > 1:
        weekday_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        weekday_clf.fit(train_z, train_weekday)
        metrics["probe_z_day_weekday_accuracy"] = float(accuracy_score(test_weekday, weekday_clf.predict(test_z)))
    if len(np.unique(train_holiday)) > 1:
        holiday_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        holiday_clf.fit(train_z, train_holiday)
        metrics["probe_z_day_holiday_accuracy"] = float(accuracy_score(test_holiday, holiday_clf.predict(test_z)))
    if float(np.std(train_discount)) > 1e-6:
        reg = Ridge(alpha=1.0)
        reg.fit(train_z, train_discount)
        pred = reg.predict(test_z)
        metrics["probe_z_day_discount_mae"] = float(np.mean(np.abs(pred - test_discount)))
    return metrics


def save_latent_diagnostics(latents: dict[str, np.ndarray], out_dir: Path) -> None:
    if "z_hour" in latents:
        z_hour = latents["z_hour"]
        with (out_dir / "z_hour_heatmap.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["hour", *[f"dim_{idx}" for idx in range(z_hour.shape[-1])]])
            mean_by_hour = z_hour.mean(axis=0)
            for hour, row in enumerate(mean_by_hour):
                writer.writerow([hour, *[float(v) for v in row]])
    if {"z_day", "day_weekday", "day_holiday", "day_discount"}.issubset(latents):
        z_day = latents["z_day"]
        day_summary = {
            "z_day_mean_norm": float(np.linalg.norm(z_day.mean(axis=(0, 1)))),
            "weekday_classes": float(len(np.unique(latents["day_weekday"]))),
            "holiday_rate": float(np.mean(latents["day_holiday"])),
            "discount_mean": float(np.mean(latents["day_discount"])),
        }
        write_json(out_dir / "latent_diagnostics.json", day_summary)


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
    patience = int(config["train"].get("early_stopping_patience", 0))
    min_delta = float(config["train"].get("early_stopping_min_delta", 0.0))
    epochs_without_improvement = 0
    best_epoch = 0
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_losses = run_epoch(model, train_loader, variant_config, device, optimizer, f"{name} train {epoch}")
        with torch.inference_mode():
            valid_losses = run_epoch(model, valid_loader, variant_config, device, None, f"{name} valid {epoch}")
            valid_pred_metrics = validation_prediction_metrics(model, valid_loader, bundle.forecast_days, device, variant_config)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_losses.items()}, **{f"valid_{k}": v for k, v in valid_losses.items()}}
        row.update(valid_pred_metrics)
        append_jsonl(out_dir / "history.jsonl", row)
        logger.info("Epoch %d complete: %s", epoch, row)
        score = row.get(selection_metric)
        if score is None:
            raise KeyError(f"selection_metric={selection_metric!r} is not available in epoch row keys={sorted(row)}")
        if float(score) < best - min_delta:
            best = float(score)
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"model": model.state_dict(), "config": variant_config, "variant": variant}, best_path)
            logger.info("Saved best checkpoint: %s=%.5f path=%s", selection_metric, best, best_path)
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping at epoch=%d best_epoch=%d best_%s=%.5f patience=%d",
                    epoch,
                    best_epoch,
                    selection_metric,
                    best,
                    patience,
                )
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    y_true, y_pred = predict_model(model, test_loader, bundle.forecast_days, device, variant_config)
    write_predictions_csv(out_dir / "predictions.csv", y_true, y_pred)
    metrics = regression_metrics(y_true, y_pred)
    if hasattr(model, "encode"):
        train_latents = collect_latent_arrays(model, train_loader, device, variant_config)
        test_latents = collect_latent_arrays(model, test_loader, device, variant_config)
        save_latent_arrays(test_latents, out_dir)
        save_latent_diagnostics(test_latents, out_dir)
        probe_metrics = run_representation_probes(train_latents, test_latents)
        metrics.update(probe_metrics)
    write_json(out_dir / "metrics.json", metrics)
    logger.info("Variant complete: metrics=%s", metrics)
    return {"name": name, "best_validation_score": best, "best_epoch": best_epoch, "selection_metric": selection_metric, **metrics}


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
