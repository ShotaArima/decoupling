from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import FreshRetailNetSeries, infer_input_dim, load_config
from .model import GLRModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    device = resolve_device(config["train"]["device"])
    out_dir = Path(config["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = dict(config["dataset"])
    data_cfg["subgroup_target"] = config["evaluation"]["subgroup_target"]
    train_ds = FreshRetailNetSeries.from_config(
        data_cfg, data_cfg["train_split"], max_series=data_cfg["max_train_series"]
    )
    valid_ds = FreshRetailNetSeries.from_config(
        data_cfg, data_cfg["eval_split"], max_series=data_cfg["max_eval_series"]
    )
    model = build_model(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"]["num_workers"]),
    )
    valid_loader = DataLoader(valid_ds, batch_size=int(config["train"]["batch_size"]), shuffle=False)

    best = float("inf")
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x, mask)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(out["loss"].detach())

        model.eval()
        val = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                out = model(batch["x"].to(device), batch["mask"].to(device))
                val += float(out["loss"])
        val /= max(len(valid_loader), 1)
        print(f"epoch={epoch} train_loss={running / max(len(train_loader), 1):.5f} val_loss={val:.5f}")
        last_path = out_dir / "last.pt"
        torch.save({"model": model.state_dict(), "config": config}, last_path)
        if val < best:
            best = val
            torch.save({"model": model.state_dict(), "config": config}, out_dir / "best.pt")
    return out_dir / "best.pt"
