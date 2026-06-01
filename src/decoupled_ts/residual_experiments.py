from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import load_config
from .experiment_logging import append_jsonl, setup_run_logger, write_json
from .metrics import regression_metrics
from .residual_diagnostics import _labels_from_grid
from .residual_models import ResidualFlattenAE, ResidualMultiGrainAE, residual_decouple_penalty
from .retail_data import build_retail_data
from .retail_models import flatten_to_grid
from .train import optimize_torch_runtime, resolve_device
from .retail_experiments import seed_everything


def _same_hour_recent_mean_torch(sales: torch.Tensor, observed: torch.Tensor, recent_days: int) -> torch.Tensor:
    series_sum = (sales * observed).sum(dim=(1, 2), keepdim=True)
    series_count = observed.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    series_mean = series_sum / series_count
    hour_sum = (sales * observed).sum(dim=1)
    hour_count = observed.sum(dim=1).clamp_min(1.0)
    hour_mean = hour_sum / hour_count
    baseline = torch.empty_like(sales)
    for day in range(sales.shape[1]):
        start = max(0, day - recent_days)
        if start == day:
            baseline[:, day, :] = hour_mean
        else:
            window_sales = sales[:, start:day, :] * observed[:, start:day, :]
            window_count = observed[:, start:day, :].sum(dim=1)
            day_mean = window_sales.sum(dim=1) / window_count.clamp_min(1.0)
            baseline[:, day, :] = torch.where(window_count > 0, day_mean, series_mean[:, 0, 0][:, None])
    return baseline


def residual_batch(batch: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    grid = flatten_to_grid(batch["x"])
    mask_grid = flatten_to_grid(batch["mask"])
    sales = grid[..., 0]
    observed = mask_grid[..., 0]
    method = str(config.get("residual", {}).get("baseline_method", "same_hour_recent_mean"))
    if method != "same_hour_recent_mean":
        raise ValueError("residual experiments currently support baseline_method='same_hour_recent_mean'")
    baseline = _same_hour_recent_mean_torch(sales, observed, int(config["dataset"].get("recent_days", 7)))
    residual = sales - baseline
    x_residual = batch["x"].clone()
    x_residual[:, 0, :] = residual.reshape(residual.shape[0], -1)
    return {
        "x": x_residual,
        "mask": batch["mask"],
        "sales": sales,
        "observed": observed,
        "baseline": baseline,
        "residual": residual,
    }


def make_residual_model(config: dict[str, Any], variant: dict[str, Any], input_dim: int, days: int, hours: int) -> nn.Module:
    model_cfg = config["model"]
    if variant["type"] == "flatten_ae":
        return ResidualFlattenAE(
            input_dim=input_dim,
            days=days,
            hours=hours,
            hidden_dim=int(model_cfg["hidden_dim"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if variant["type"] == "multigrain_ae":
        return ResidualMultiGrainAE(
            input_dim=input_dim,
            days=days,
            hours=hours,
            hidden_dim=int(model_cfg["hidden_dim"]),
            global_dim=int(model_cfg["global_dim"]),
            local_dim=int(model_cfg.get("local_dim", model_cfg["day_dim"])),
            day_dim=int(model_cfg["day_dim"]),
            hour_dim=int(model_cfg["hour_dim"]),
            interaction_dim=int(model_cfg["interaction_dim"]),
            use_global=bool(variant.get("use_global", True)),
            use_local=bool(variant.get("use_local", False)),
            use_day=bool(variant.get("use_day", True)),
            use_hour=bool(variant.get("use_hour", True)),
            use_interaction=bool(variant.get("use_interaction", False)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    raise ValueError(f"unknown residual variant type: {variant['type']}")


def residual_loss(out: dict[str, torch.Tensor], rb: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    observed = rb["observed"]
    residual = rb["residual"]
    abs_loss = (torch.abs(out["residual_hat"] - residual) * observed).sum() / observed.sum().clamp_min(1.0)
    total = abs_loss
    values = {"loss_residual_reconstruction": abs_loss.detach()}
    if "z_global" in out:
        decouple = residual_decouple_penalty(out)
        total = total + float(config["loss"].get("decouple_weight", 0.0)) * decouple
        values["loss_decouple"] = decouple.detach()
    values["loss"] = total
    return values


def _sample_derangement(batch: int, device: torch.device) -> torch.Tensor:
    if batch <= 1:
        return torch.arange(batch, device=device)
    return torch.roll(torch.randperm(batch, device=device), shifts=1)


def _center_by_mean(x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    return x - x.mean(dim=dims, keepdim=True)


def swap_regularization_loss(model: nn.Module, out: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    if not isinstance(model, ResidualMultiGrainAE):
        return {}
    swap_cfg = config.get("swap_regularization", {})
    if not bool(swap_cfg.get("enabled", False)):
        return {}
    if out["residual_hat"].shape[0] <= 1:
        return {}
    perm = _sample_derangement(out["residual_hat"].shape[0], out["residual_hat"].device)
    base = out["residual_hat"]
    losses: dict[str, torch.Tensor] = {}
    if model.use_global and "z_global" in out:
        swapped = model.decode_from_parts(out, z_global=out["z_global"][perm])
        # Swapping series identity should mainly change the whole-cell level, not day/hour shape after centering.
        losses["loss_swap_global_shape"] = torch.mean(torch.abs(_center_by_mean(swapped, (1, 2)) - _center_by_mean(base, (1, 2))))
    if model.use_day and "z_day" in out:
        swapped = model.decode_from_parts(out, z_day=out["z_day"][perm])
        # Day swaps should preserve hour profile after averaging over days.
        losses["loss_swap_day_hour_invariance"] = torch.mean(torch.abs(swapped.mean(dim=1) - base.mean(dim=1)))
    if model.use_hour and "z_hour" in out:
        swapped = model.decode_from_parts(out, z_hour=out["z_hour"][perm])
        # Hour swaps should preserve day profile after averaging over hours.
        losses["loss_swap_hour_day_invariance"] = torch.mean(torch.abs(swapped.mean(dim=2) - base.mean(dim=2)))
    if model.use_interaction and "z_interaction" in out:
        swapped = model.decode_from_parts(out, z_interaction=out["z_interaction"][perm])
        # Interaction swaps should have little additive main effect; emphasize cell-level deviations.
        main_effect = swapped.mean(dim=1, keepdim=True) + swapped.mean(dim=2, keepdim=True) - swapped.mean(dim=(1, 2), keepdim=True)
        losses["loss_swap_interaction_main_effect"] = torch.mean(torch.abs(main_effect - swapped.mean(dim=(1, 2), keepdim=True)))
    return losses


def run_epoch(model: nn.Module, loader: DataLoader, config: dict[str, Any], device: torch.device, optimizer, desc: str) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    progress = tqdm(loader, desc=desc, unit="batch")
    for step, batch in enumerate(progress, start=1):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        rb = residual_batch(batch, config)
        if training:
            optimizer.zero_grad(set_to_none=True)
        out = model(rb["x"], rb["mask"])
        losses = residual_loss(out, rb, config)
        swap_losses = swap_regularization_loss(model, out, config)
        if swap_losses:
            total_swap = torch.tensor(0.0, device=losses["loss"].device)
            weights = config.get("swap_regularization", {})
            for key, value in swap_losses.items():
                weight_key = key.replace("loss_swap_", "") + "_weight"
                total_swap = total_swap + float(weights.get(weight_key, weights.get("weight", 0.0))) * value
                losses[key] = value.detach()
            losses["loss_swap"] = total_swap.detach()
            losses["loss"] = losses["loss"] + total_swap
        if training:
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"].get("grad_clip", 1.0)))
            optimizer.step()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        progress.set_postfix(loss=f"{totals['loss'] / step:.4f}")
    return {key: value / max(len(loader), 1) for key, value in totals.items()}


@torch.no_grad()
def predict_residuals(model: nn.Module, loader: DataLoader, config: dict[str, Any], device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    parts = {"sales": [], "baseline": [], "residual": [], "residual_hat": [], "observed": [], "grid": [], "subgroup": []}
    latents: dict[str, list[np.ndarray]] = {"z_global": [], "z_local": [], "z_day": [], "z_hour": []}
    for batch in tqdm(loader, desc="residual test", unit="batch"):
        batch_dev = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        rb = residual_batch(batch_dev, config)
        out = model(rb["x"], rb["mask"])
        for key in ("sales", "baseline", "residual", "observed"):
            parts[key].append(rb[key].detach().cpu().numpy())
        parts["residual_hat"].append(out["residual_hat"].detach().cpu().numpy())
        parts["grid"].append(flatten_to_grid(batch["x"]).numpy())
        parts["subgroup"].append(batch["subgroup"].numpy())
        for key in latents:
            if key in out:
                latents[key].append(out[key].detach().cpu().numpy())
    result = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    result.update({key: np.concatenate(value, axis=0) for key, value in latents.items() if value})
    return result


def residual_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    keep = arrays["observed"] > 0
    residual = arrays["residual"][keep]
    pred = arrays["residual_hat"][keep]
    sales = arrays["sales"][keep]
    baseline = arrays["baseline"][keep]
    corrected = baseline + pred
    metrics = {
        "residual_mae": float(np.mean(np.abs(pred - residual))),
        "residual_rmse": float(np.sqrt(np.mean(np.square(pred - residual)))),
        "residual_r2": float(r2_score(residual, pred)) if residual.size > 1 else 0.0,
        "residual_sign_accuracy": float(np.mean(np.sign(pred) == np.sign(residual))),
    }
    metrics.update({f"baseline_cell_{key}": value for key, value in regression_metrics(sales, baseline).items()})
    metrics.update({f"corrected_cell_{key}": value for key, value in regression_metrics(sales, corrected).items()})
    abs_residual = np.abs(residual)
    threshold = np.quantile(abs_residual, 0.9) if abs_residual.size else 0.0
    high = abs_residual >= threshold
    if high.any():
        metrics["high_residual_top10_baseline_mae"] = float(np.mean(np.abs(baseline[high] - sales[high])))
        metrics["high_residual_top10_corrected_mae"] = float(np.mean(np.abs(corrected[high] - sales[high])))
    return metrics


@torch.no_grad()
def evaluate_swap_diagnostics(model: nn.Module, loader: DataLoader, config: dict[str, Any], device: torch.device) -> dict[str, float]:
    if not isinstance(model, ResidualMultiGrainAE):
        return {}
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in tqdm(loader, desc="swap diagnostics", unit="batch"):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        rb = residual_batch(batch, config)
        out = model(rb["x"], rb["mask"])
        if out["residual_hat"].shape[0] <= 1:
            continue
        perm = _sample_derangement(out["residual_hat"].shape[0], out["residual_hat"].device)
        base = out["residual_hat"]
        batch_metrics: dict[str, torch.Tensor] = {}
        if model.use_global and "z_global" in out:
            swapped = model.decode_from_parts(out, z_global=out["z_global"][perm])
            batch_metrics["swap_global_total_delta"] = torch.mean(torch.abs(swapped - base))
            batch_metrics["swap_global_shape_delta"] = torch.mean(torch.abs(_center_by_mean(swapped, (1, 2)) - _center_by_mean(base, (1, 2))))
        if model.use_day and "z_day" in out:
            swapped = model.decode_from_parts(out, z_day=out["z_day"][perm])
            batch_metrics["swap_day_total_delta"] = torch.mean(torch.abs(swapped - base))
            batch_metrics["swap_day_hour_profile_delta"] = torch.mean(torch.abs(swapped.mean(dim=1) - base.mean(dim=1)))
        if model.use_hour and "z_hour" in out:
            swapped = model.decode_from_parts(out, z_hour=out["z_hour"][perm])
            batch_metrics["swap_hour_total_delta"] = torch.mean(torch.abs(swapped - base))
            batch_metrics["swap_hour_day_profile_delta"] = torch.mean(torch.abs(swapped.mean(dim=2) - base.mean(dim=2)))
        if model.use_interaction and "z_interaction" in out:
            swapped = model.decode_from_parts(out, z_interaction=out["z_interaction"][perm])
            main_effect = swapped.mean(dim=1, keepdim=True) + swapped.mean(dim=2, keepdim=True) - swapped.mean(dim=(1, 2), keepdim=True)
            batch_metrics["swap_interaction_total_delta"] = torch.mean(torch.abs(swapped - base))
            batch_metrics["swap_interaction_main_effect_delta"] = torch.mean(torch.abs(main_effect - swapped.mean(dim=(1, 2), keepdim=True)))
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_residual_predictions(path: Path, arrays: dict[str, np.ndarray]) -> None:
    keep = arrays["observed"] > 0
    sales = arrays["sales"][keep]
    baseline = arrays["baseline"][keep]
    residual = arrays["residual"][keep]
    pred = arrays["residual_hat"][keep]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "sales", "baseline", "residual", "residual_hat", "corrected", "baseline_abs_error", "corrected_abs_error"])
        writer.writeheader()
        for idx, row in enumerate(zip(sales, baseline, residual, pred)):
            y, b, r, r_hat = [float(v) for v in row]
            writer.writerow(
                {
                    "index": idx,
                    "sales": y,
                    "baseline": b,
                    "residual": r,
                    "residual_hat": r_hat,
                    "corrected": b + r_hat,
                    "baseline_abs_error": abs(b - y),
                    "corrected_abs_error": abs(b + r_hat - y),
                }
            )


def _probe_day_labels(grid: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray]:
    labels = _labels_from_grid(grid, config)
    return {
        "weekday": labels["weekday"][:, :, 0],
        "holiday": (labels["holiday"][:, :, 0] > 0.5).astype(np.int64),
        "discount": labels["discount"][:, :, 0],
    }


def run_latent_probes(train_arrays: dict[str, np.ndarray], test_arrays: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if "z_global" in train_arrays and len(np.unique(train_arrays["subgroup"])) > 1:
        train_z = train_arrays["z_global"].reshape(train_arrays["z_global"].shape[0], -1)
        test_z = test_arrays["z_global"].reshape(test_arrays["z_global"].shape[0], -1)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(train_z, train_arrays["subgroup"])
        metrics["probe_z_global_subgroup_accuracy"] = float(accuracy_score(test_arrays["subgroup"], clf.predict(test_z)))
    if "z_day" in train_arrays:
        train_labels = _probe_day_labels(train_arrays["grid"], config)
        test_labels = _probe_day_labels(test_arrays["grid"], config)
        train_z = train_arrays["z_day"].reshape(-1, train_arrays["z_day"].shape[-1])
        test_z = test_arrays["z_day"].reshape(-1, test_arrays["z_day"].shape[-1])
        train_weekday = train_labels["weekday"].reshape(-1)
        test_weekday = test_labels["weekday"].reshape(-1)
        if len(np.unique(train_weekday)) > 1:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(train_z, train_weekday)
            metrics["probe_z_day_weekday_accuracy"] = float(accuracy_score(test_weekday, clf.predict(test_z)))
        train_discount = train_labels["discount"].reshape(-1)
        test_discount = test_labels["discount"].reshape(-1)
        if float(np.std(train_discount)) > 1e-6:
            reg = Ridge(alpha=1.0)
            reg.fit(train_z, train_discount)
            metrics["probe_z_day_discount_mae"] = float(np.mean(np.abs(reg.predict(test_z) - test_discount)))
    if "z_hour" in train_arrays:
        train_z = train_arrays["z_hour"].reshape(-1, train_arrays["z_hour"].shape[-1])
        test_z = test_arrays["z_hour"].reshape(-1, test_arrays["z_hour"].shape[-1])
        train_hour = np.tile(np.arange(train_arrays["z_hour"].shape[1]), train_arrays["z_hour"].shape[0])
        test_hour = np.tile(np.arange(test_arrays["z_hour"].shape[1]), test_arrays["z_hour"].shape[0])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(train_z, train_hour)
        metrics["probe_z_hour_hour_accuracy"] = float(accuracy_score(test_hour, clf.predict(test_z)))
    return metrics


def save_latent_outputs(arrays: dict[str, np.ndarray], out_dir: Path) -> None:
    for key in ("z_global", "z_local", "z_day", "z_hour", "subgroup"):
        if key in arrays:
            np.save(out_dir / f"{key}.npy", arrays[key])
    if "z_hour" in arrays:
        with (out_dir / "z_hour_heatmap.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["hour", *[f"dim_{i}" for i in range(arrays["z_hour"].shape[-1])]])
            for hour, row in enumerate(arrays["z_hour"].mean(axis=0)):
                writer.writerow([hour, *[float(v) for v in row]])


def run_variant(config: dict[str, Any], variant: dict[str, Any], bundle, device: torch.device, root_out: Path) -> dict[str, Any]:
    variant_config = dict(config)
    if "loss_overrides" in variant:
        variant_config["loss"] = {**config["loss"], **variant["loss_overrides"]}
    if "swap_regularization_overrides" in variant:
        variant_config["swap_regularization"] = {
            **config.get("swap_regularization", {}),
            **variant["swap_regularization_overrides"],
        }
    name = variant["name"]
    out_dir = root_out / name
    logger = setup_run_logger(out_dir, name=f"residual.{name}")
    for stale in (
        "z_global.npy",
        "z_local.npy",
        "z_day.npy",
        "z_hour.npy",
        "subgroup.npy",
        "z_hour_heatmap.csv",
        "residual_predictions.csv",
        "metrics.json",
        "history.jsonl",
        "best.pt",
    ):
        path = out_dir / stale
        if path.exists():
            path.unlink()
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
    model = make_residual_model(variant_config, variant, bundle.input_dim, bundle.days, bundle.hours).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    best = float("inf")
    best_path = out_dir / "best.pt"
    best_epoch = 0
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_losses = run_epoch(model, train_loader, variant_config, device, optimizer, f"{name} train {epoch}")
        valid_losses = run_epoch(model, valid_loader, variant_config, device, None, f"{name} valid {epoch}")
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_losses.items()}, **{f"valid_{k}": v for k, v in valid_losses.items()}}
        append_jsonl(out_dir / "history.jsonl", row)
        score = row[str(train_cfg.get("selection_metric", "valid_loss_residual_reconstruction"))]
        logger.info("Epoch %d complete: %s", epoch, row)
        if score < best:
            best = float(score)
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "config": variant_config, "variant": variant}, best_path)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    train_arrays = predict_residuals(model, train_loader, variant_config, device)
    test_arrays = predict_residuals(model, test_loader, variant_config, device)
    metrics = residual_metrics(test_arrays)
    metrics.update(run_latent_probes(train_arrays, test_arrays, variant_config))
    metrics.update(evaluate_swap_diagnostics(model, test_loader, variant_config, device))
    save_residual_predictions(out_dir / "residual_predictions.csv", test_arrays)
    save_latent_outputs(test_arrays, out_dir)
    write_json(out_dir / "metrics.json", metrics)
    return {"name": name, "best_validation_score": best, "best_epoch": best_epoch, **metrics}


def run_residual_experiments(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    device = resolve_device(config["train"]["device"])
    optimize_torch_runtime(device)
    root_out = Path(config["train"]["output_dir"])
    root_out.mkdir(parents=True, exist_ok=True)
    logger = setup_run_logger(root_out, name="residual_experiment")
    logger.info("Residual representation experiment start config=%s device=%s", config_path, device)
    write_json(root_out / "resolved_config.json", config)
    bundle = build_retail_data(config)
    results = [run_variant(config, variant, bundle, device, root_out) for variant in config["experiments"]]
    summary = {"results": results}
    write_json(root_out / "summary.json", summary)
    with (root_out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in results for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Residual representation experiment complete: %s", summary)
    return summary
