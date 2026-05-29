from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RetailBatch:
    x: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor
    subgroup: torch.Tensor | None = None


class FlattenMLPForecast(nn.Module):
    """Simple baseline that ignores the D x H structure."""

    def __init__(self, input_dim: int, days: int, hours: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * days * hours, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        grid = flatten_to_grid(x)
        pred = self.net(grid.reshape(grid.shape[0], -1)).squeeze(-1)
        return {"prediction": pred}


class GlobalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, global_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, global_dim),
        )

    def forward(self, grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        observed = grid * mask
        denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
        mean = observed.sum(dim=(1, 2)) / denom
        var = ((grid - mean[:, None, None, :]).pow(2) * mask).sum(dim=(1, 2)) / denom
        return self.net(torch.cat([mean, torch.sqrt(var + 1e-6)], dim=-1))


class DayEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, day_dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim * 2, hidden_dim), nn.ReLU())
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, day_dim)

    def forward(self, grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=2).clamp_min(1.0)
        mean = (grid * mask).sum(dim=2) / denom
        var = ((grid - mean[:, :, None, :]).pow(2) * mask).sum(dim=2) / denom
        h = self.proj(torch.cat([mean, torch.sqrt(var + 1e-6)], dim=-1))
        out, _ = self.rnn(h)
        return self.head(out)


class HourEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hour_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hour_dim),
        )

    def forward(self, grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=1).clamp_min(1.0)
        mean = (grid * mask).sum(dim=1) / denom
        var = ((grid - mean[:, None, :, :]).pow(2) * mask).sum(dim=1) / denom
        return self.net(torch.cat([mean, torch.sqrt(var + 1e-6)], dim=-1))


class InteractionEncoder(nn.Module):
    def __init__(self, day_dim: int, hour_dim: int, hidden_dim: int, interaction_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(day_dim + hour_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, interaction_dim),
        )

    def forward(self, z_day: torch.Tensor, z_hour: torch.Tensor) -> torch.Tensor:
        days = z_day.shape[1]
        hours = z_hour.shape[1]
        zd = z_day[:, :, None, :].expand(-1, days, hours, -1)
        zh = z_hour[:, None, :, :].expand(-1, days, hours, -1)
        return self.net(torch.cat([zd, zh], dim=-1))


class RetailMultiGrainModel(nn.Module):
    """Retail model with global, day, hour, and optional day-hour interaction latents."""

    def __init__(
        self,
        input_dim: int,
        days: int,
        hours: int,
        hidden_dim: int,
        global_dim: int,
        day_dim: int,
        hour_dim: int,
        interaction_dim: int,
        use_day: bool = True,
        use_hour: bool = True,
        use_interaction: bool = False,
        forecast_days: int = 1,
        forecast_activation: str = "softplus",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.days = days
        self.hours = hours
        self.global_dim = global_dim
        self.day_dim = day_dim
        self.hour_dim = hour_dim
        self.interaction_dim = interaction_dim if use_interaction else 0
        self.use_day = use_day
        self.use_hour = use_hour
        self.use_interaction = use_interaction
        self.forecast_days = forecast_days
        self.forecast_activation = forecast_activation

        self.global_encoder = GlobalEncoder(input_dim, hidden_dim, global_dim)
        self.day_encoder = DayEncoder(input_dim, hidden_dim, day_dim)
        self.hour_encoder = HourEncoder(input_dim, hidden_dim, hour_dim)
        self.interaction_encoder = InteractionEncoder(day_dim, hour_dim, hidden_dim, interaction_dim)
        latent_dim = global_dim
        latent_dim += day_dim if use_day else 0
        latent_dim += hour_dim if use_hour else 0
        latent_dim += interaction_dim if use_interaction else 0
        self.forecast_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        grid = flatten_to_grid(x)
        mask_grid = flatten_to_grid(mask)
        z_global = self.global_encoder(grid, mask_grid)
        z_day = self.day_encoder(grid, mask_grid)
        z_hour = self.hour_encoder(grid, mask_grid)
        z_interaction = self.interaction_encoder(z_day, z_hour)
        return {
            "grid": grid,
            "mask_grid": mask_grid,
            "z_global": z_global,
            "z_day": z_day,
            "z_hour": z_hour,
            "z_interaction": z_interaction,
        }

    def _latent_grid(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        batch = encoded["grid"].shape[0]
        days = encoded["grid"].shape[1]
        hours = encoded["grid"].shape[2]
        parts = [encoded["z_global"][:, None, None, :].expand(batch, days, hours, -1)]
        if self.use_day:
            parts.append(encoded["z_day"][:, :, None, :].expand(batch, days, hours, -1))
        if self.use_hour:
            parts.append(encoded["z_hour"][:, None, :, :].expand(batch, days, hours, -1))
        if self.use_interaction:
            parts.append(encoded["z_interaction"])
        return torch.cat(parts, dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode(x, mask)
        latent = self._latent_grid(encoded)
        sales_grid = F.softplus(self.forecast_head(latent).squeeze(-1))
        recon_grid = self.reconstruction_head(latent)
        target_sum = self._predict_future_from_encoded(encoded, self.forecast_days)
        return {
            "prediction": target_sum,
            "sales_grid": sales_grid,
            "reconstruction": recon_grid,
            **encoded,
        }

    @torch.no_grad()
    def predict_future_sum(self, x: torch.Tensor, mask: torch.Tensor, forecast_days: int) -> torch.Tensor:
        encoded = self.encode(x, mask)
        return self._predict_future_from_encoded(encoded, forecast_days)

    def _predict_future_from_encoded(self, encoded: dict[str, torch.Tensor], forecast_days: int) -> torch.Tensor:
        z_day = encoded["z_day"]
        recent = z_day[:, -min(7, z_day.shape[1]) :, :].mean(dim=1)
        future_day = recent[:, None, :].expand(-1, forecast_days, -1)
        z_hour = encoded["z_hour"]
        batch, hours = z_hour.shape[:2]
        parts = [encoded["z_global"][:, None, None, :].expand(batch, forecast_days, hours, -1)]
        if self.use_day:
            parts.append(future_day[:, :, None, :].expand(batch, forecast_days, hours, -1))
        if self.use_hour:
            parts.append(z_hour[:, None, :, :].expand(batch, forecast_days, hours, -1))
        if self.use_interaction:
            parts.append(self.interaction_encoder(future_day, z_hour))
        latent = torch.cat(parts, dim=-1)
        pred = self.forecast_head(latent).squeeze(-1)
        if self.forecast_activation == "softplus":
            pred = F.softplus(pred)
        elif self.forecast_activation == "relu":
            pred = F.relu(pred)
        elif self.forecast_activation == "linear":
            pass
        else:
            raise ValueError(f"unknown forecast_activation: {self.forecast_activation}")
        return pred.sum(dim=(1, 2))


def flatten_to_grid(x: torch.Tensor) -> torch.Tensor:
    """Convert [B, F, D*H] to [B, D, H, F]."""
    if x.ndim != 3:
        raise ValueError(f"expected [batch, features, time], got shape={tuple(x.shape)}")
    batch, features, time = x.shape
    if time % 24 != 0:
        raise ValueError("time dimension must be divisible by 24")
    days = time // 24
    return x.reshape(batch, features, days, 24).permute(0, 2, 3, 1).contiguous()


def covariance_penalty(parts: list[torch.Tensor]) -> torch.Tensor:
    flattened = [p.reshape(p.shape[0], -1, p.shape[-1]).mean(dim=1) for p in parts if p is not None]
    if len(flattened) < 2:
        return torch.tensor(0.0, device=flattened[0].device if flattened else "cpu")
    penalty = torch.tensor(0.0, device=flattened[0].device)
    for i in range(len(flattened)):
        for j in range(i + 1, len(flattened)):
            a = flattened[i] - flattened[i].mean(dim=0, keepdim=True)
            b = flattened[j] - flattened[j].mean(dim=0, keepdim=True)
            cov = a.T @ b / max(a.shape[0] - 1, 1)
            penalty = penalty + cov.pow(2).mean()
    return penalty
