from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import FreshRetailNetSeries
from .train import build_model, resolve_device


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
def collect_representations(model: nn.Module, loader: DataLoader, device: torch.device):
    zls, zgs, ys, groups, xs, masks = [], [], [], [], [], []
    model.eval()
    for batch in loader:
        x = batch["x"].to(device)
        enc = model.encode(x)
        zls.append(enc["zl_mean"].cpu())
        zgs.append(enc["zg_mean"].cpu())
        ys.append(batch["target"].float())
        groups.append(batch["subgroup"].long())
        xs.append(batch["x"])
        masks.append(batch["mask"])
    return (
        torch.cat(zls),
        torch.cat(zgs),
        torch.cat(ys),
        torch.cat(groups),
        torch.cat(xs),
        torch.cat(masks),
    )


def train_downstream_regression(train_rep, eval_rep, config: dict, device: torch.device) -> dict[str, float]:
    train_zl, train_zg, train_y = [v.to(device) for v in train_rep[:3]]
    eval_zl, eval_zg, eval_y = [v.to(device) for v in eval_rep[:3]]
    model = RepresentationRegressor(
        local_dim=config["model"]["local_dim"],
        global_dim=config["model"]["global_dim"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(config["evaluation"]["lr"]))
    ds = TensorDataset(train_zl, train_zg, train_y)
    loader = DataLoader(ds, batch_size=int(config["train"]["batch_size"]), shuffle=True)
    for _ in range(int(config["evaluation"]["epochs"])):
        for zl, zg, y in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(zl, zg)
            loss = nn.functional.l1_loss(pred, y)
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = model(eval_zl, eval_zg).cpu().numpy()
    y = eval_y.cpu().numpy()
    return {"mae": float(mean_absolute_error(y, pred)), "mse": float(mean_squared_error(y, pred))}


def train_subgroup_classifier(train_rep, eval_rep, config: dict, device: torch.device) -> dict[str, float]:
    train_zg = train_rep[1].to(device)
    eval_zg = eval_rep[1].to(device)
    labels = torch.cat([train_rep[3], eval_rep[3]])
    classes = {int(v): i for i, v in enumerate(sorted(labels.unique().tolist()))}
    train_y = torch.tensor([classes[int(v)] for v in train_rep[3]], dtype=torch.long, device=device)
    eval_y = torch.tensor([classes[int(v)] for v in eval_rep[3]], dtype=torch.long, device=device)
    model = RepresentationClassifier(config["model"]["global_dim"], len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(config["evaluation"]["lr"]))
    loader = DataLoader(TensorDataset(train_zg, train_y), batch_size=int(config["train"]["batch_size"]), shuffle=True)
    for _ in range(int(config["evaluation"]["epochs"])):
        for zg, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(zg), y)
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = model(eval_zg).argmax(dim=-1).cpu().numpy()
    return {"accuracy": float(accuracy_score(eval_y.cpu().numpy(), pred))}


def evaluate_forecast(model: nn.Module, eval_loader: DataLoader, config: dict, device: torch.device) -> dict[str, float]:
    forecast_windows = int(config["dataset"]["forecast_windows"])
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            x = batch["x"].to(device)
            pred_windows = model.forecast(x, forecast_windows=forecast_windows)
            sale_pred = pred_windows[:, :, 0, :].reshape(x.shape[0], -1).sum(dim=1).cpu().numpy()
            preds.append(sale_pred)
            trues.append(batch["target"].numpy())
    pred = np.concatenate(preds)
    y = np.concatenate(trues)
    return {"mae": float(mean_absolute_error(y, pred)), "mse": float(mean_squared_error(y, pred))}


def run_evaluation(checkpoint_path: str) -> dict[str, dict[str, float]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    device = resolve_device(config["train"]["device"])
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    data_cfg = dict(config["dataset"])
    data_cfg["subgroup_target"] = config["evaluation"]["subgroup_target"]
    train_ds = FreshRetailNetSeries.from_config(
        data_cfg, data_cfg["train_split"], max_series=data_cfg["max_train_series"]
    )
    eval_ds = FreshRetailNetSeries.from_config(
        data_cfg, data_cfg["eval_split"], max_series=data_cfg["max_eval_series"]
    )
    train_loader = DataLoader(train_ds, batch_size=int(config["train"]["batch_size"]), shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=int(config["train"]["batch_size"]), shuffle=False)
    train_rep = collect_representations(model, train_loader, device)
    eval_rep = collect_representations(model, eval_loader, device)
    results = {
        "downstream_prediction": train_downstream_regression(train_rep, eval_rep, config, device),
        "subgroup_identification": train_subgroup_classifier(train_rep, eval_rep, config, device),
        "forecasting": evaluate_forecast(model, eval_loader, config, device),
    }
    out_path = Path(config["train"]["output_dir"]) / "metrics.json"
    out_path.write_text(__import__("json").dumps(results, indent=2), encoding="utf-8")
    return results
