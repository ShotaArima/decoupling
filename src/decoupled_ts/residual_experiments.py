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
from .residual_models import OutputDecompositionResidualModel, ResidualFlattenAE, ResidualMultiGrainAE, residual_decouple_penalty
from .retail_data import build_retail_data
from .retail_models import flatten_to_grid
from .train import optimize_torch_runtime, resolve_device
from .retail_experiments import seed_everything


def _series_mean_torch(sales: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    series_sum = (sales * observed).sum(dim=(1, 2), keepdim=True)
    series_count = observed.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    return torch.broadcast_to(series_sum / series_count, sales.shape)


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


def _weekday_same_hour_mean_torch(sales: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    series_mean = _series_mean_torch(sales, observed)
    baseline = torch.empty_like(sales)
    days = sales.shape[1]
    weekday_ids = torch.arange(days, device=sales.device) % 7
    for weekday in range(7):
        day_sel = weekday_ids == weekday
        if not bool(day_sel.any()):
            continue
        wk_sales = sales[:, day_sel, :] * observed[:, day_sel, :]
        wk_count = observed[:, day_sel, :].sum(dim=1)
        wk_mean = wk_sales.sum(dim=1) / wk_count.clamp_min(1.0)
        fill = torch.where(wk_count > 0, wk_mean, series_mean[:, 0, :])
        baseline[:, day_sel, :] = fill[:, None, :]
    return baseline


def residual_batch(batch: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    grid = flatten_to_grid(batch["x"])
    mask_grid = flatten_to_grid(batch["mask"])
    sales = grid[..., 0]
    observed = mask_grid[..., 0]
    method = str(config.get("residual", {}).get("baseline_method", "same_hour_recent_mean"))
    target = str(config.get("residual", {}).get("target", "baseline_residual"))
    if target in {"true_residual", "noisy_true_residual"} and target in batch:
        residual = batch[target].to(sales.device)
        baseline = sales - residual
        extra: dict[str, torch.Tensor] = {}
    elif method == "series_mean":
        baseline = _series_mean_torch(sales, observed)
        residual = sales - baseline
        extra = {}
    elif method == "weekday_same_hour_mean":
        baseline = _weekday_same_hour_mean_torch(sales, observed)
        residual = sales - baseline
        extra = {}
    elif method == "same_hour_recent_mean":
        recent_days = int(config.get("residual", {}).get("recent_days", config["dataset"].get("recent_days", 7)))
        baseline = _same_hour_recent_mean_torch(sales, observed, recent_days)
        residual = sales - baseline
        extra = {}
    elif method == "log1p_same_hour_recent_mean":
        recent_days = int(config.get("residual", {}).get("recent_days", config["dataset"].get("recent_days", 7)))
        log_sales = torch.log1p(torch.clamp_min(sales, 0.0))
        baseline_log = _same_hour_recent_mean_torch(log_sales, observed, recent_days)
        baseline = torch.expm1(baseline_log).clamp_min(0.0)
        residual = log_sales - baseline_log
        extra = {"baseline_log": baseline_log}
    else:
        raise ValueError(
            "residual experiments support baseline_method in "
            "{'series_mean', 'weekday_same_hour_mean', 'same_hour_recent_mean', 'log1p_same_hour_recent_mean'}"
        )
    x_residual = batch["x"].clone()
    x_residual[:, 0, :] = residual.reshape(residual.shape[0], -1)
    result = {
        "x": x_residual,
        "mask": batch["mask"],
        "sales": sales,
        "observed": observed,
        "baseline": baseline,
        "residual": residual,
    }
    result.update(extra)
    return result


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
    if variant["type"] == "output_decomposition":
        return OutputDecompositionResidualModel(
            input_dim=input_dim,
            days=days,
            hours=hours,
            hidden_dim=int(model_cfg["hidden_dim"]),
            global_dim=int(model_cfg["global_dim"]),
            day_dim=int(model_cfg["day_dim"]),
            hour_dim=int(model_cfg["hour_dim"]),
            interaction_dim=int(model_cfg["interaction_dim"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
            center_components=bool(variant.get("center_components", True)),
            use_interaction=bool(variant.get("use_interaction", True)),
        )
    raise ValueError(f"unknown residual variant type: {variant['type']}")


def component_constraint_loss(out: dict[str, torch.Tensor]) -> torch.Tensor:
    penalties = []
    if "day_component" in out:
        penalties.append(torch.mean(torch.square(out["day_component"].mean(dim=1))))
    if "hour_component" in out:
        penalties.append(torch.mean(torch.square(out["hour_component"].mean(dim=2))))
    if "interaction_component" in out:
        penalties.append(torch.mean(torch.square(out["interaction_component"].mean(dim=1))))
        penalties.append(torch.mean(torch.square(out["interaction_component"].mean(dim=2))))
    if not penalties:
        return torch.tensor(0.0, device=out["residual_hat"].device)
    return sum(penalties)


def residual_loss(out: dict[str, torch.Tensor], rb: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    observed = rb["observed"]
    residual = rb["residual"]
    error = out["residual_hat"] - residual
    abs_loss = (torch.abs(error) * observed).sum() / observed.sum().clamp_min(1.0)
    total = abs_loss
    values = {"loss_residual_reconstruction": abs_loss.detach()}
    residual_bias_weight = float(config["loss"].get("residual_bias_weight", 0.0))
    if residual_bias_weight > 0.0:
        residual_bias = torch.square((error * observed).sum() / observed.sum().clamp_min(1.0))
        total = total + residual_bias_weight * residual_bias
        values["loss_residual_bias"] = residual_bias.detach()
    series_residual_bias_weight = float(config["loss"].get("series_residual_bias_weight", 0.0))
    if series_residual_bias_weight > 0.0:
        series_count = observed.sum(dim=(1, 2)).clamp_min(1.0)
        series_bias = (error * observed).sum(dim=(1, 2)) / series_count
        series_bias_penalty = torch.mean(torch.square(series_bias))
        total = total + series_residual_bias_weight * series_bias_penalty
        values["loss_series_residual_bias"] = series_bias_penalty.detach()
    if "z_global" in out:
        decouple = residual_decouple_penalty(out)
        total = total + float(config["loss"].get("decouple_weight", 0.0)) * decouple
        values["loss_decouple"] = decouple.detach()
    if any(key in out for key in ("day_component", "hour_component", "interaction_component")):
        component_penalty = component_constraint_loss(out)
        total = total + float(config["loss"].get("component_constraint_weight", 0.0)) * component_penalty
        values["loss_component_constraint"] = component_penalty.detach()
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
    parts = {"sales": [], "baseline": [], "baseline_log": [], "residual": [], "residual_hat": [], "observed": [], "grid": [], "subgroup": []}
    latents: dict[str, list[np.ndarray]] = {"z_global": [], "z_local": [], "z_day": [], "z_hour": [], "z_interaction": []}
    components: dict[str, list[np.ndarray]] = {
        "global_component": [],
        "day_component": [],
        "hour_component": [],
        "interaction_component": [],
        "true_global": [],
        "true_day": [],
        "true_hour": [],
        "true_interaction": [],
        "true_residual": [],
        "noisy_true_residual": [],
    }
    for batch in tqdm(loader, desc="residual test", unit="batch"):
        batch_dev = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        rb = residual_batch(batch_dev, config)
        out = model(rb["x"], rb["mask"])
        for key in ("sales", "baseline", "residual", "observed"):
            parts[key].append(rb[key].detach().cpu().numpy())
        if "baseline_log" in rb:
            parts["baseline_log"].append(rb["baseline_log"].detach().cpu().numpy())
        parts["residual_hat"].append(out["residual_hat"].detach().cpu().numpy())
        parts["grid"].append(flatten_to_grid(batch["x"]).numpy())
        parts["subgroup"].append(batch["subgroup"].numpy())
        for key in latents:
            if key in out:
                latents[key].append(out[key].detach().cpu().numpy())
        for key in ("global_component", "day_component", "hour_component", "interaction_component"):
            if key in out:
                components[key].append(out[key].detach().cpu().numpy())
        for key in ("true_global", "true_day", "true_hour", "true_interaction", "true_residual", "noisy_true_residual"):
            if key in batch:
                components[key].append(batch[key].numpy())
    result = {key: np.concatenate(value, axis=0) for key, value in parts.items() if value}
    result.update({key: np.concatenate(value, axis=0) for key, value in latents.items() if value})
    result.update({key: np.concatenate(value, axis=0) for key, value in components.items() if value})
    return result


def residual_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    keep = arrays["observed"] > 0
    residual = arrays["residual"][keep]
    pred = arrays["residual_hat"][keep]
    sales = arrays["sales"][keep]
    baseline = arrays["baseline"][keep]
    if "baseline_log" in arrays:
        baseline_log = arrays["baseline_log"][keep]
        corrected = np.expm1(baseline_log + pred)
        corrected = np.clip(corrected, 0.0, None)
    else:
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


def _observed_mean(values: np.ndarray, observed: np.ndarray) -> float:
    keep = observed > 0
    if not keep.any():
        return 0.0
    return float(np.mean(values[keep]))


def residual_prediction_bias(arrays: dict[str, np.ndarray]) -> float:
    return _observed_mean(arrays["residual_hat"] - arrays["residual"], arrays["observed"])


def _observed_pair(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    keep = arrays["observed"] > 0
    return arrays["residual_hat"][keep], arrays["residual"][keep]


def _calibration_bias(residual_hat: np.ndarray, residual: np.ndarray) -> float:
    if residual_hat.size == 0:
        return 0.0
    return float(np.mean(residual_hat - residual))


def _calibration_candidate_metrics(
    pred_valid: np.ndarray,
    residual_valid: np.ndarray,
    alpha: float,
    bias: float,
) -> dict[str, float]:
    if residual_valid.size == 0:
        return {"mae": 0.0, "bias": 0.0, "high_residual_top10_mae": 0.0}
    pred = alpha * pred_valid + bias
    abs_error = np.abs(pred - residual_valid)
    abs_residual = np.abs(residual_valid)
    threshold = float(np.quantile(abs_residual, 0.9)) if abs_residual.size else 0.0
    high = abs_residual >= threshold
    return {
        "mae": float(np.mean(abs_error)),
        "bias": _calibration_bias(pred, residual_valid),
        "high_residual_top10_mae": float(np.mean(abs_error[high])) if high.any() else 0.0,
    }


def _estimate_calibration_bias(residual_error: np.ndarray, estimator: str) -> float:
    if residual_error.size == 0:
        return 0.0
    if estimator == "median":
        return float(np.median(residual_error))
    return float(np.mean(residual_error))


def calibrated_residual_metrics(
    valid_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
    calibration_cfg: dict[str, Any],
) -> dict[str, float]:
    if not bool(calibration_cfg.get("enabled", False)):
        return {}
    mode = str(calibration_cfg.get("mode", "validation_residual_bias"))
    adjusted = dict(test_arrays)
    bias = 0.0
    alpha = 1.0
    validation_mae = 0.0
    validation_bias = 0.0
    validation_high_residual_top10_mae = 0.0
    constraint_satisfied: float | None = None
    if mode == "validation_residual_bias":
        bias = residual_prediction_bias(valid_arrays)
        adjusted["residual_hat"] = test_arrays["residual_hat"] - bias
        alpha = 1.0
        pred_valid, residual_valid = _observed_pair({**valid_arrays, "residual_hat": valid_arrays["residual_hat"] - bias})
        candidate = _calibration_candidate_metrics(pred_valid, residual_valid, 1.0, 0.0)
        validation_mae = candidate["mae"]
        validation_bias = candidate["bias"]
        validation_high_residual_top10_mae = candidate["high_residual_top10_mae"]
    elif mode == "validation_affine_residual":
        pred_valid, residual_valid = _observed_pair(valid_arrays)
        pred_mean = float(np.mean(pred_valid)) if pred_valid.size else 0.0
        residual_mean = float(np.mean(residual_valid)) if residual_valid.size else 0.0
        denom = float(np.mean(np.square(pred_valid - pred_mean))) if pred_valid.size else 0.0
        if denom > 1e-12:
            alpha = float(np.mean((pred_valid - pred_mean) * (residual_valid - residual_mean)) / denom)
        else:
            alpha = 0.0
        alpha = float(np.clip(alpha, float(calibration_cfg.get("min_alpha", 0.0)), float(calibration_cfg.get("max_alpha", 1.5))))
        bias = residual_mean - alpha * pred_mean
        adjusted["residual_hat"] = alpha * test_arrays["residual_hat"] + bias
        candidate = _calibration_candidate_metrics(pred_valid, residual_valid, alpha, bias)
        validation_mae = candidate["mae"]
        validation_bias = candidate["bias"]
        validation_high_residual_top10_mae = candidate["high_residual_top10_mae"]
    elif mode == "validation_mae_grid":
        pred_valid, residual_valid = _observed_pair(valid_arrays)
        alpha_grid = calibration_cfg.get("alpha_grid")
        if alpha_grid is None:
            alpha_grid = [round(value * 0.05, 2) for value in range(0, 31)]
        best_score = float("inf")
        best_alpha = 1.0
        best_bias = 0.0
        estimator = str(calibration_cfg.get("bias_estimator", "median"))
        for alpha_candidate in [float(value) for value in alpha_grid]:
            residual_error = residual_valid - alpha_candidate * pred_valid
            bias_candidate = _estimate_calibration_bias(residual_error, estimator)
            score = _calibration_candidate_metrics(pred_valid, residual_valid, alpha_candidate, bias_candidate)["mae"]
            if score < best_score:
                best_score = score
                best_alpha = alpha_candidate
                best_bias = bias_candidate
        alpha = best_alpha
        bias = best_bias
        validation_mae = best_score
        adjusted["residual_hat"] = alpha * test_arrays["residual_hat"] + bias
        candidate = _calibration_candidate_metrics(pred_valid, residual_valid, alpha, bias)
        validation_bias = candidate["bias"]
        validation_high_residual_top10_mae = candidate["high_residual_top10_mae"]
    elif mode == "validation_bias_constrained_mae_grid":
        pred_valid, residual_valid = _observed_pair(valid_arrays)
        alpha_grid = calibration_cfg.get("alpha_grid")
        if alpha_grid is None:
            alpha_grid = [round(value * 0.05, 2) for value in range(0, 31)]
        estimator = str(calibration_cfg.get("bias_estimator", "median"))
        max_abs_bias = float(calibration_cfg.get("max_abs_validation_bias", 0.02))
        best_tuple: tuple[float, float, float, float, float] | None = None
        best_fallback: tuple[float, float, float, float, float] | None = None
        for alpha_candidate in [float(value) for value in alpha_grid]:
            residual_error = residual_valid - alpha_candidate * pred_valid
            bias_candidate = _estimate_calibration_bias(residual_error, estimator)
            candidate = _calibration_candidate_metrics(pred_valid, residual_valid, alpha_candidate, bias_candidate)
            cand_tuple = (
                candidate["mae"],
                abs(candidate["bias"]),
                candidate["high_residual_top10_mae"],
                alpha_candidate,
                bias_candidate,
            )
            fallback_tuple = (
                abs(candidate["bias"]),
                candidate["mae"],
                candidate["high_residual_top10_mae"],
                alpha_candidate,
                bias_candidate,
            )
            if abs(candidate["bias"]) <= max_abs_bias and (best_tuple is None or cand_tuple < best_tuple):
                best_tuple = cand_tuple
            if best_fallback is None or fallback_tuple < best_fallback:
                best_fallback = fallback_tuple
        constraint_satisfied = 1.0 if best_tuple is not None else 0.0
        chosen = best_tuple if best_tuple is not None else best_fallback
        if chosen is None:
            alpha = 1.0
            bias = 0.0
            validation_mae = 0.0
        else:
            if best_tuple is not None:
                validation_mae, _, validation_high_residual_top10_mae, alpha, bias = chosen
            else:
                _, validation_mae, validation_high_residual_top10_mae, alpha, bias = chosen
            candidate = _calibration_candidate_metrics(pred_valid, residual_valid, alpha, bias)
            validation_bias = candidate["bias"]
            validation_high_residual_top10_mae = candidate["high_residual_top10_mae"]
        adjusted["residual_hat"] = alpha * test_arrays["residual_hat"] + bias
    elif mode == "validation_weighted_mae_bias_grid":
        pred_valid, residual_valid = _observed_pair(valid_arrays)
        alpha_grid = calibration_cfg.get("alpha_grid")
        if alpha_grid is None:
            alpha_grid = [round(value * 0.05, 2) for value in range(0, 31)]
        estimator = str(calibration_cfg.get("bias_estimator", "median"))
        bias_weight = float(calibration_cfg.get("bias_weight", 1.0))
        high_residual_weight = float(calibration_cfg.get("high_residual_weight", 0.0))
        best_score = float("inf")
        for alpha_candidate in [float(value) for value in alpha_grid]:
            residual_error = residual_valid - alpha_candidate * pred_valid
            bias_candidate = _estimate_calibration_bias(residual_error, estimator)
            candidate = _calibration_candidate_metrics(pred_valid, residual_valid, alpha_candidate, bias_candidate)
            score = candidate["mae"] + bias_weight * abs(candidate["bias"]) + high_residual_weight * candidate["high_residual_top10_mae"]
            if score < best_score:
                best_score = score
                alpha = alpha_candidate
                bias = bias_candidate
                validation_mae = candidate["mae"]
                validation_bias = candidate["bias"]
                validation_high_residual_top10_mae = candidate["high_residual_top10_mae"]
        adjusted["residual_hat"] = alpha * test_arrays["residual_hat"] + bias
    elif mode == "test_oracle_residual_bias":
        bias = residual_prediction_bias(test_arrays)
        adjusted["residual_hat"] = test_arrays["residual_hat"] - bias
        alpha = 1.0
        pred_valid, residual_valid = _observed_pair(valid_arrays)
        candidate = _calibration_candidate_metrics(pred_valid, residual_valid, alpha, -bias)
        validation_mae = candidate["mae"]
        validation_bias = candidate["bias"]
        validation_high_residual_top10_mae = candidate["high_residual_top10_mae"]
    elif mode == "none":
        adjusted["residual_hat"] = test_arrays["residual_hat"].copy()
        pred_valid, residual_valid = _observed_pair(valid_arrays)
        candidate = _calibration_candidate_metrics(pred_valid, residual_valid, alpha, bias)
        validation_mae = candidate["mae"]
        validation_bias = candidate["bias"]
        validation_high_residual_top10_mae = candidate["high_residual_top10_mae"]
    else:
        raise ValueError(f"unknown calibration mode: {mode}")
    clip_quantile = calibration_cfg.get("clip_residual_quantile")
    if clip_quantile is not None:
        keep = valid_arrays["observed"] > 0
        q = float(clip_quantile)
        limit = float(np.quantile(np.abs(valid_arrays["residual"][keep]), q)) if keep.any() else 0.0
        adjusted["residual_hat"] = np.clip(adjusted["residual_hat"], -limit, limit)
    metrics = residual_metrics(adjusted)
    metrics.update(component_recovery_metrics(adjusted))
    metrics.update(component_ablation_metrics(adjusted))
    metrics.update(hour_profile_metrics(adjusted))
    metrics["calibration_residual_bias"] = bias
    metrics["calibration_alpha"] = alpha
    metrics["calibration_validation_mae"] = validation_mae
    metrics["calibration_validation_prediction_bias"] = validation_bias
    metrics["calibration_validation_high_residual_top10_mae"] = validation_high_residual_top10_mae
    if constraint_satisfied is not None:
        metrics["calibration_constraint_satisfied"] = constraint_satisfied
    return {f"calibrated_{key}": value for key, value in metrics.items()}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1)
    y = b.reshape(-1)
    if x.size < 2 or float(np.std(x)) <= 1e-8 or float(np.std(y)) <= 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def component_recovery_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    pairs = {
        "global": ("true_global", "global_component"),
        "day": ("true_day", "day_component"),
        "hour": ("true_hour", "hour_component"),
        "interaction": ("true_interaction", "interaction_component"),
    }
    metrics: dict[str, float] = {}
    keep = arrays["observed"] > 0
    for name, (true_key, pred_key) in pairs.items():
        if true_key not in arrays or pred_key not in arrays:
            continue
        true = arrays[true_key][keep]
        pred = arrays[pred_key][keep]
        metrics[f"component_{name}_mae"] = float(np.mean(np.abs(pred - true)))
        metrics[f"component_{name}_rmse"] = float(np.sqrt(np.mean(np.square(pred - true))))
        metrics[f"component_{name}_corr"] = _corr(arrays[true_key][keep], arrays[pred_key][keep])
    if "true_residual" in arrays:
        true = arrays["true_residual"][keep]
        pred = arrays["residual_hat"][keep]
        metrics["component_total_true_residual_mae"] = float(np.mean(np.abs(pred - true)))
        metrics["component_total_true_residual_rmse"] = float(np.sqrt(np.mean(np.square(pred - true))))
    if all(key in arrays for key in ("day_component", "hour_component", "interaction_component")):
        metrics["component_day_mean_abs"] = float(np.mean(np.abs(arrays["day_component"].mean(axis=1))))
        metrics["component_hour_mean_abs"] = float(np.mean(np.abs(arrays["hour_component"].mean(axis=2))))
        metrics["component_interaction_day_mean_abs"] = float(np.mean(np.abs(arrays["interaction_component"].mean(axis=1))))
        metrics["component_interaction_hour_mean_abs"] = float(np.mean(np.abs(arrays["interaction_component"].mean(axis=2))))
    return metrics


def component_ablation_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    if not all(key in arrays for key in ("global_component", "day_component", "hour_component", "interaction_component")):
        return {}
    keep = arrays["observed"] > 0
    target = arrays["true_residual"] if "true_residual" in arrays else arrays["residual"]
    full = arrays["residual_hat"]
    full_mae = float(np.mean(np.abs(full[keep] - target[keep])))
    metrics = {"component_ablation_full_mae": full_mae}
    for name, key in (
        ("global", "global_component"),
        ("day", "day_component"),
        ("hour", "hour_component"),
        ("interaction", "interaction_component"),
    ):
        pred = full - arrays[key]
        mae = float(np.mean(np.abs(pred[keep] - target[keep])))
        metrics[f"component_ablation_without_{name}_mae"] = mae
        metrics[f"component_ablation_without_{name}_mae_delta"] = mae - full_mae
    return metrics


def hour_profile_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    keep = arrays["observed"] > 0
    residual = arrays["residual"]
    pred = arrays["residual_hat"]
    observed = arrays["observed"]
    residual_profile = np.sum(residual * observed, axis=(0, 1)) / np.clip(np.sum(observed, axis=(0, 1)), 1.0, None)
    pred_profile = np.sum(pred * observed, axis=(0, 1)) / np.clip(np.sum(observed, axis=(0, 1)), 1.0, None)
    metrics = {
        "residual_hour_profile_corr": _corr(residual_profile, pred_profile),
        "residual_hour_profile_mae": float(np.mean(np.abs(residual_profile - pred_profile))),
    }
    if "hour_component" in arrays:
        hour_component = arrays["hour_component"]
        component_profile = np.sum(hour_component * observed, axis=(0, 1)) / np.clip(np.sum(observed, axis=(0, 1)), 1.0, None)
        metrics["hour_component_residual_profile_corr"] = _corr(residual_profile, component_profile)
        metrics["hour_component_profile_mean_abs"] = float(np.mean(np.abs(component_profile)))
    if keep.any() and "hour_component" in arrays:
        metrics["hour_component_cell_abs_mean"] = float(np.mean(np.abs(arrays["hour_component"][keep])))
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
    if "baseline_log" in arrays:
        corrected = np.clip(np.expm1(arrays["baseline_log"][keep] + pred), 0.0, None)
    else:
        corrected = baseline + pred
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "sales", "baseline", "residual", "residual_hat", "corrected", "baseline_abs_error", "corrected_abs_error"])
        writer.writeheader()
        for idx, row in enumerate(zip(sales, baseline, residual, pred, corrected)):
            y, b, r, r_hat, y_hat = [float(v) for v in row]
            writer.writerow(
                {
                    "index": idx,
                    "sales": y,
                    "baseline": b,
                    "residual": r,
                    "residual_hat": r_hat,
                    "corrected": y_hat,
                    "baseline_abs_error": abs(b - y),
                    "corrected_abs_error": abs(y_hat - y),
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


def save_latent_outputs(arrays: dict[str, np.ndarray], out_dir: Path, output_cfg: dict[str, Any]) -> None:
    if bool(output_cfg.get("save_latent_arrays", True)):
        for key in (
            "z_global",
            "z_local",
            "z_day",
            "z_hour",
            "z_interaction",
            "global_component",
            "day_component",
            "hour_component",
            "interaction_component",
            "true_global",
            "true_day",
            "true_hour",
            "true_interaction",
            "true_residual",
            "noisy_true_residual",
            "subgroup",
        ):
            if key in arrays:
                np.save(out_dir / f"{key}.npy", arrays[key])
    if bool(output_cfg.get("save_hour_heatmap", True)) and "z_hour" in arrays:
        with (out_dir / "z_hour_heatmap.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["hour", *[f"dim_{i}" for i in range(arrays["z_hour"].shape[-1])]])
            for hour, row in enumerate(arrays["z_hour"].mean(axis=0)):
                writer.writerow([hour, *[float(v) for v in row]])


def _masked_profile(values: np.ndarray, observed: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    numerator = np.sum(values * observed, axis=axis)
    denominator = np.clip(np.sum(observed, axis=axis), 1.0, None)
    return numerator / denominator


def _series_mae(error: np.ndarray, observed: np.ndarray) -> np.ndarray:
    numerator = np.sum(np.abs(error) * observed, axis=(1, 2))
    denominator = np.clip(np.sum(observed, axis=(1, 2)), 1.0, None)
    return numerator / denominator


def save_visualization_outputs(arrays: dict[str, np.ndarray], out_dir: Path, output_cfg: dict[str, Any]) -> None:
    if not bool(output_cfg.get("save_visualization_tables", False)):
        return
    viz_dir = out_dir / "visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)
    observed = arrays["observed"]
    sales = arrays["sales"]
    baseline = arrays["baseline"]
    residual = arrays["residual"]
    residual_hat = arrays["residual_hat"]
    corrected = baseline + residual_hat
    profile_sources = {
        "residual": residual,
        "residual_hat": residual_hat,
        "baseline_abs_error": np.abs(baseline - sales),
        "corrected_abs_error": np.abs(corrected - sales),
    }
    for key in ("global_component", "day_component", "hour_component", "interaction_component"):
        if key in arrays:
            profile_sources[key] = arrays[key]
    with (viz_dir / "profiles_by_hour.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "hour", "value"])
        for source, values in profile_sources.items():
            profile = _masked_profile(values, observed, axis=(0, 1))
            for hour, value in enumerate(profile):
                writer.writerow([source, hour, float(value)])
    with (viz_dir / "profiles_by_day.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "day", "value"])
        for source, values in profile_sources.items():
            profile = _masked_profile(values, observed, axis=(0, 2))
            for day, value in enumerate(profile):
                writer.writerow([source, day, float(value)])

    baseline_mae = _series_mae(baseline - sales, observed)
    corrected_mae = _series_mae(corrected - sales, observed)
    residual_mae = _series_mae(residual_hat - residual, observed)
    mean_abs_residual = _series_mae(residual, observed)
    improvement = baseline_mae - corrected_mae
    rows = []
    for idx in range(sales.shape[0]):
        rows.append(
            {
                "series_index": idx,
                "subgroup": int(arrays["subgroup"][idx]) if "subgroup" in arrays else 0,
                "observed_cells": float(np.sum(observed[idx])),
                "baseline_mae": float(baseline_mae[idx]),
                "corrected_mae": float(corrected_mae[idx]),
                "improvement": float(improvement[idx]),
                "residual_mae": float(residual_mae[idx]),
                "mean_abs_residual": float(mean_abs_residual[idx]),
            }
        )
    with (viz_dir / "series_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["series_index"])
        writer.writeheader()
        writer.writerows(rows)

    top_k = int(output_cfg.get("visualization_top_k", 3))
    order_best = np.argsort(-improvement)[:top_k]
    order_worst = np.argsort(improvement)[:top_k]
    for group_name, indices in (("best", order_best), ("worst", order_worst)):
        for rank, idx in enumerate(indices, start=1):
            path = viz_dir / f"series_{group_name}_{rank:02d}.csv"
            fieldnames = [
                "day",
                "hour",
                "observed",
                "sales",
                "baseline",
                "corrected",
                "residual",
                "residual_hat",
                "baseline_abs_error",
                "corrected_abs_error",
            ]
            for key in ("global_component", "day_component", "hour_component", "interaction_component"):
                if key in arrays:
                    fieldnames.append(key)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for day in range(sales.shape[1]):
                    for hour in range(sales.shape[2]):
                        row = {
                            "day": day,
                            "hour": hour,
                            "observed": float(observed[idx, day, hour]),
                            "sales": float(sales[idx, day, hour]),
                            "baseline": float(baseline[idx, day, hour]),
                            "corrected": float(corrected[idx, day, hour]),
                            "residual": float(residual[idx, day, hour]),
                            "residual_hat": float(residual_hat[idx, day, hour]),
                            "baseline_abs_error": float(abs(baseline[idx, day, hour] - sales[idx, day, hour])),
                            "corrected_abs_error": float(abs(corrected[idx, day, hour] - sales[idx, day, hour])),
                        }
                        for key in ("global_component", "day_component", "hour_component", "interaction_component"):
                            if key in arrays:
                                row[key] = float(arrays[key][idx, day, hour])
                        writer.writerow(row)


def run_variant(config: dict[str, Any], variant: dict[str, Any], bundle, device: torch.device, root_out: Path) -> dict[str, Any]:
    variant_config = dict(config)
    if "loss_overrides" in variant:
        variant_config["loss"] = {**config["loss"], **variant["loss_overrides"]}
    if "swap_regularization_overrides" in variant:
        variant_config["swap_regularization"] = {
            **config.get("swap_regularization", {}),
            **variant["swap_regularization_overrides"],
        }
    if "calibration_overrides" in variant:
        variant_config["calibration"] = {
            **config.get("calibration", {}),
            **variant["calibration_overrides"],
        }
    output_cfg = {**config.get("output", {}), **variant.get("output_overrides", {})}
    name = variant["name"]
    out_dir = root_out / name
    logger = setup_run_logger(out_dir, name=f"residual.{name}")
    for stale in (
        "z_global.npy",
        "z_local.npy",
        "z_day.npy",
        "z_hour.npy",
        "z_interaction.npy",
        "global_component.npy",
        "day_component.npy",
        "hour_component.npy",
        "interaction_component.npy",
        "true_global.npy",
        "true_day.npy",
        "true_hour.npy",
        "true_interaction.npy",
        "true_residual.npy",
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
    valid_arrays = predict_residuals(model, valid_loader, variant_config, device)
    test_arrays = predict_residuals(model, test_loader, variant_config, device)
    metrics = residual_metrics(test_arrays)
    metrics.update(component_recovery_metrics(test_arrays))
    metrics.update(component_ablation_metrics(test_arrays))
    metrics.update(hour_profile_metrics(test_arrays))
    metrics.update(calibrated_residual_metrics(valid_arrays, test_arrays, variant_config.get("calibration", {})))
    metrics.update(run_latent_probes(train_arrays, test_arrays, variant_config))
    metrics.update(evaluate_swap_diagnostics(model, test_loader, variant_config, device))
    if bool(output_cfg.get("save_residual_predictions", True)):
        save_residual_predictions(out_dir / "residual_predictions.csv", test_arrays)
    save_latent_outputs(test_arrays, out_dir, output_cfg)
    save_visualization_outputs(test_arrays, out_dir, output_cfg)
    if not bool(output_cfg.get("save_checkpoints", True)) and best_path.exists():
        best_path.unlink()
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
