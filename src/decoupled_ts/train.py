from __future__ import annotations

import random
import logging
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import FreshRetailNetSeries, infer_input_dim, load_config
from .model import GLRModel


LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def optimize_torch_runtime(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def make_loader(dataset, config: dict, *, shuffle: bool) -> DataLoader:
    train_cfg = config["train"]
    workers = int(train_cfg.get("num_workers", 0))
    kwargs = {
        "batch_size": int(train_cfg["batch_size"]),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": bool(train_cfg.get("pin_memory", False)) and torch.cuda.is_available(),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


def checkpoint_state_dict(model: torch.nn.Module) -> dict:
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def build_model(config: dict) -> GLRModel:
    data_cfg = config["dataset"]
    model_cfg = dict(config["model"])
    model_cfg["input_dim"] = infer_input_dim(data_cfg)
    return GLRModel(
        input_dim=model_cfg["input_dim"],
        window_size=data_cfg["window_size"],
        global_dim=model_cfg["global_dim"],
        local_dim=model_cfg["local_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        beta=model_cfg["beta"],
        lambda_counterfactual=model_cfg["lambda_counterfactual"],
        kernel_names=model_cfg["kernel_names"],
        kernel_scales=model_cfg["kernel_scales"],
        decoder_std=model_cfg["decoder_std"],
    )


def train_glr(config_path: str) -> Path:
    LOGGER.info("Starting GLR training config=%s", config_path)
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    device = resolve_device(config["train"]["device"])
    optimize_torch_runtime(device)
    out_dir = Path(config["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Using device=%s seed=%s output_dir=%s", device, config["seed"], out_dir)

    data_cfg = dict(config["dataset"])
    data_cfg["subgroup_target"] = config["evaluation"]["subgroup_target"]
    LOGGER.info("Step 1/5: building training dataset split=%s", data_cfg["train_split"])
    train_ds = FreshRetailNetSeries.from_config(
        data_cfg, data_cfg["train_split"], max_series=data_cfg["max_train_series"]
    )
    LOGGER.info("Training examples=%d", len(train_ds))
    LOGGER.info("Step 2/5: building validation dataset split=%s", data_cfg["eval_split"])
    valid_ds = FreshRetailNetSeries.from_config(
        data_cfg, data_cfg["eval_split"], max_series=data_cfg["max_eval_series"]
    )
    LOGGER.info("Validation examples=%d", len(valid_ds))
    LOGGER.info("Step 3/5: initializing model")
    model = build_model(config).to(device)
    if bool(config["train"].get("compile", False)) and hasattr(torch, "compile"):
        LOGGER.info("Compiling model with torch.compile")
        model = torch.compile(model)
    opt = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
    train_loader = make_loader(train_ds, config, shuffle=True)
    valid_loader = make_loader(valid_ds, config, shuffle=False)
    amp_enabled = bool(config["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    LOGGER.info(
        "Model/data ready: input_dim=%d local_dim=%d global_dim=%d train_batches=%d valid_batches=%d amp=%s",
        getattr(model, "input_dim", config["model"]["input_dim"]),
        config["model"]["local_dim"],
        config["model"]["global_dim"],
        len(train_loader),
        len(valid_loader),
        amp_enabled,
    )

    best = float("inf")
    epochs = int(config["train"]["epochs"])
    LOGGER.info("Step 4/5: training epochs=%d", epochs)
    for epoch in range(1, epochs + 1):
        LOGGER.info("Epoch %d/%d: train start", epoch, epochs)
        model.train()
        running = 0.0
        running_recon = 0.0
        running_kl_local = 0.0
        running_kl_global = 0.0
        running_cf = 0.0
        progress = tqdm(train_loader, desc=f"train epoch {epoch}/{epochs}", unit="batch")
        for step, batch in enumerate(progress, start=1):
            x = batch["x"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                out = model(x, mask)
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += float(out["loss"].detach())
            running_recon += float(out["reconstruction"])
            running_kl_local += float(out["kl_local"])
            running_kl_global += float(out["kl_global"])
            running_cf += float(out["counterfactual"])
            progress.set_postfix(
                loss=f"{running / step:.4f}",
                recon=f"{running_recon / step:.4f}",
                kl_l=f"{running_kl_local / step:.4f}",
                kl_g=f"{running_kl_global / step:.4f}",
                cf=f"{running_cf / step:.4f}",
            )

        LOGGER.info("Epoch %d/%d: validation start", epoch, epochs)
        model.eval()
        val = 0.0
        valid_progress = tqdm(valid_loader, desc=f"valid epoch {epoch}/{epochs}", unit="batch")
        with torch.inference_mode():
            for step, batch in enumerate(valid_progress, start=1):
                with autocast_context(device, amp_enabled):
                    out = model(batch["x"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True))
                val += float(out["loss"])
                valid_progress.set_postfix(loss=f"{val / step:.4f}")
        val /= max(len(valid_loader), 1)
        train_loss = running / max(len(train_loader), 1)
        LOGGER.info("Epoch %d/%d complete: train_loss=%.5f val_loss=%.5f", epoch, epochs, train_loss, val)
        last_path = out_dir / "last.pt"
        LOGGER.info("Saving latest checkpoint to %s", last_path)
        torch.save({"model": checkpoint_state_dict(model), "config": config}, last_path)
        if val < best:
            best = val
            best_path = out_dir / "best.pt"
            LOGGER.info("New best val_loss=%.5f; saving %s", best, best_path)
            torch.save({"model": checkpoint_state_dict(model), "config": config}, best_path)
    LOGGER.info("Step 5/5: training complete best_val_loss=%.5f", best)
    return out_dir / "best.pt"
