from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data import FreshRetailNetSeries
from .train import autocast_context, build_model, make_loader, optimize_torch_runtime, resolve_device


LOGGER = logging.getLogger(__name__)


def evaluation_batch_size(config: dict) -> int:
    return int(config.get("evaluation", {}).get("batch_size") or config["train"]["batch_size"])


def release_cuda_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


class RepresentationRegressor(nn.Module):
    def __init__(self, local_dim: int, global_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.rnn = nn.GRU(local_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, zl: torch.Tensor, zg: torch.Tensor) -> torch.Tensor:
        _, h = self.rnn(zl)
        return self.head(torch.cat([h[-1], zg], dim=-1)).squeeze(-1)


class RepresentationClassifier(nn.Module):
    def __init__(self, global_dim: int, classes: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, zg: torch.Tensor) -> torch.Tensor:
        return self.net(zg)


@torch.no_grad()
def collect_representations(model: nn.Module, loader: DataLoader, device: torch.device, name: str):
    zls, zgs, ys, groups = [], [], [], []
    model.eval()
    amp_enabled = device.type == "cuda"
    progress = tqdm(loader, desc=f"collect-repr {name}", unit="batch")
    for batch in progress:
        x = batch["x"].to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            enc = model.encode(x)
        zls.append(enc["zl_mean"].float().cpu())
        zgs.append(enc["zg_mean"].float().cpu())
        ys.append(batch["target"].float())
        groups.append(batch["subgroup"].long())
    return (
        torch.cat(zls),
        torch.cat(zgs),
        torch.cat(ys),
        torch.cat(groups),
    )


def train_downstream_regression(train_rep, eval_rep, config: dict, device: torch.device) -> dict[str, float]:
    LOGGER.info("Evaluation step 4/6: downstream prediction training start")
    train_zl, train_zg, train_y = [v.float() for v in train_rep[:3]]
    eval_zl, eval_zg, eval_y = [v.float() for v in eval_rep[:3]]
    model = RepresentationRegressor(
        local_dim=config["model"]["local_dim"],
        global_dim=config["model"]["global_dim"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(config["evaluation"]["lr"]))
    ds = TensorDataset(train_zl, train_zg, train_y)
    batch_size = evaluation_batch_size(config)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    epochs = int(config["evaluation"]["epochs"])
    for epoch in range(1, epochs + 1):
        running = 0.0
        progress = tqdm(loader, desc=f"downstream epoch {epoch}/{epochs}", unit="batch")
        for step, (zl, zg, y) in enumerate(progress, start=1):
            zl = zl.to(device, non_blocking=True)
            zg = zg.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(zl, zg)
            loss = nn.functional.l1_loss(pred, y)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            progress.set_postfix(mae_loss=f"{running / step:.4f}")
        LOGGER.info("Downstream epoch %d/%d train_mae_loss=%.5f", epoch, epochs, running / max(len(loader), 1))
    pred_parts = []
    eval_loader = DataLoader(
        TensorDataset(eval_zl, eval_zg, eval_y),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    with torch.inference_mode():
        for zl, zg, _ in tqdm(eval_loader, desc="downstream eval", unit="batch"):
            pred_parts.append(model(zl.to(device, non_blocking=True), zg.to(device, non_blocking=True)).cpu())
    pred = torch.cat(pred_parts).numpy()
    y = eval_y.cpu().numpy()
    metrics = {"mae": float(mean_absolute_error(y, pred)), "mse": float(mean_squared_error(y, pred))}
    LOGGER.info("Downstream prediction metrics: %s", metrics)
    release_cuda_memory(device)
    return metrics


def train_subgroup_classifier(train_rep, eval_rep, config: dict, device: torch.device) -> dict[str, float]:
    LOGGER.info("Evaluation step 5/6: subgroup classifier training start")
    train_zg = train_rep[1].float()
    eval_zg = eval_rep[1].float()
    labels = torch.cat([train_rep[3], eval_rep[3]])
    classes = {int(v): i for i, v in enumerate(sorted(labels.unique().tolist()))}
    train_y = torch.tensor([classes[int(v)] for v in train_rep[3]], dtype=torch.long)
    eval_y = torch.tensor([classes[int(v)] for v in eval_rep[3]], dtype=torch.long)
    model = RepresentationClassifier(config["model"]["global_dim"], len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(config["evaluation"]["lr"]))
    batch_size = evaluation_batch_size(config)
    loader = DataLoader(TensorDataset(train_zg, train_y), batch_size=batch_size, shuffle=True)
    LOGGER.info("Subgroup classes=%d", len(classes))
    epochs = int(config["evaluation"]["epochs"])
    for epoch in range(1, epochs + 1):
        running = 0.0
        progress = tqdm(loader, desc=f"subgroup epoch {epoch}/{epochs}", unit="batch")
        for step, (zg, y) in enumerate(progress, start=1):
            zg = zg.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(zg), y)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            progress.set_postfix(ce_loss=f"{running / step:.4f}")
        LOGGER.info("Subgroup epoch %d/%d train_ce_loss=%.5f", epoch, epochs, running / max(len(loader), 1))
    pred_parts = []
    eval_loader = DataLoader(TensorDataset(eval_zg, eval_y), batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.inference_mode():
        for zg, _ in tqdm(eval_loader, desc="subgroup eval", unit="batch"):
            pred_parts.append(model(zg.to(device, non_blocking=True)).argmax(dim=-1).cpu())
    pred = torch.cat(pred_parts).numpy()
    metrics = {"accuracy": float(accuracy_score(eval_y.cpu().numpy(), pred))}
    LOGGER.info("Subgroup identification metrics: %s", metrics)
    release_cuda_memory(device)
    return metrics


def evaluate_forecast(model: nn.Module, eval_loader: DataLoader, config: dict, device: torch.device) -> dict[str, float]:
    LOGGER.info("Evaluation step 6/6: GP latent forecasting start")
    forecast_windows = int(config["dataset"]["forecast_windows"])
    preds, trues = [], []
    model.eval()
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        progress = tqdm(eval_loader, desc="forecast", unit="batch")
        for batch in progress:
            x = batch["x"].to(device, non_blocking=True)
            with autocast_context(device, amp_enabled):
                pred_windows = model.forecast(x, forecast_windows=forecast_windows)
            sale_pred = pred_windows[:, :, 0, :].reshape(x.shape[0], -1).sum(dim=1).cpu().numpy()
            preds.append(sale_pred)
            trues.append(batch["target"].numpy())
    pred = np.concatenate(preds)
    y = np.concatenate(trues)
    metrics = {"mae": float(mean_absolute_error(y, pred)), "mse": float(mean_squared_error(y, pred))}
    LOGGER.info("Forecasting metrics: %s", metrics)
    release_cuda_memory(device)
    return metrics


def run_evaluation(checkpoint_path: str) -> dict[str, dict[str, float]]:
    LOGGER.info("Evaluation step 1/6: loading checkpoint=%s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = resolve_device(config["train"]["device"])
    optimize_torch_runtime(device)
    LOGGER.info("Using device=%s", device)
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    data_cfg = dict(config["dataset"])
    data_cfg["subgroup_target"] = config["evaluation"]["subgroup_target"]
    LOGGER.info("Evaluation step 2/6: building datasets")
    train_ds = FreshRetailNetSeries.from_config(
        data_cfg, data_cfg["train_split"], max_series=data_cfg["max_train_series"]
    )
    eval_ds = FreshRetailNetSeries.from_config(
        data_cfg, data_cfg["eval_split"], max_series=data_cfg["max_eval_series"]
    )
    LOGGER.info("Evaluation datasets ready: train_examples=%d eval_examples=%d", len(train_ds), len(eval_ds))
    train_loader = make_loader(train_ds, config, shuffle=False)
    eval_loader = make_loader(eval_ds, config, shuffle=False)
    LOGGER.info("Evaluation step 3/6: collecting learned representations")
    train_rep = collect_representations(model, train_loader, device, "train")
    eval_rep = collect_representations(model, eval_loader, device, "eval")
    LOGGER.info(
        "Representations ready: train_zl=%s train_zg=%s eval_zl=%s eval_zg=%s",
        tuple(train_rep[0].shape),
        tuple(train_rep[1].shape),
        tuple(eval_rep[0].shape),
        tuple(eval_rep[1].shape),
    )
    results = {
        "downstream_prediction": train_downstream_regression(train_rep, eval_rep, config, device),
        "subgroup_identification": train_subgroup_classifier(train_rep, eval_rep, config, device),
        "forecasting": evaluate_forecast(model, eval_loader, config, device),
    }
    out_path = Path(config["train"]["output_dir"]) / "metrics.json"
    out_path.write_text(__import__("json").dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Evaluation complete; metrics written to %s", out_path)
    return results
