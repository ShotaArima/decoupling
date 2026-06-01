from __future__ import annotations

from torch import nn
import torch

from .retail_models import DayEncoder, GlobalEncoder, HourEncoder, InteractionEncoder, covariance_penalty, flatten_to_grid


class ResidualFlattenAE(nn.Module):
    def __init__(self, input_dim: int, days: int, hours: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.days = days
        self.hours = hours
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * days * hours, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(hidden_dim, days * hours)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        grid = flatten_to_grid(x)
        z = self.encoder(grid.reshape(grid.shape[0], -1))
        residual_hat = self.decoder(z).reshape(grid.shape[0], self.days, self.hours)
        return {"residual_hat": residual_hat, "z_global": z}


class CellLocalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, local_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, local_dim),
        )

    def forward(self, grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([grid, mask], dim=-1))


class ResidualMultiGrainAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        days: int,
        hours: int,
        hidden_dim: int,
        global_dim: int,
        local_dim: int,
        day_dim: int,
        hour_dim: int,
        interaction_dim: int,
        use_global: bool = True,
        use_local: bool = False,
        use_day: bool = True,
        use_hour: bool = True,
        use_interaction: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.days = days
        self.hours = hours
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.day_dim = day_dim
        self.hour_dim = hour_dim
        self.interaction_dim = interaction_dim if use_interaction else 0
        self.use_global = use_global
        self.use_local = use_local
        self.use_day = use_day
        self.use_hour = use_hour
        self.use_interaction = use_interaction

        self.global_encoder = GlobalEncoder(input_dim, hidden_dim, global_dim)
        self.local_encoder = CellLocalEncoder(input_dim, hidden_dim, local_dim)
        self.day_encoder = DayEncoder(input_dim, hidden_dim, day_dim)
        self.hour_encoder = HourEncoder(input_dim, hidden_dim, hour_dim)
        self.interaction_encoder = InteractionEncoder(day_dim, hour_dim, hidden_dim, interaction_dim)
        latent_dim = 0
        latent_dim += global_dim if use_global else 0
        latent_dim += local_dim if use_local else 0
        latent_dim += day_dim if use_day else 0
        latent_dim += hour_dim if use_hour else 0
        latent_dim += interaction_dim if use_interaction else 0
        if latent_dim == 0:
            raise ValueError("at least one latent component must be enabled")
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        grid = flatten_to_grid(x)
        mask_grid = flatten_to_grid(mask)
        z_global = self.global_encoder(grid, mask_grid)
        z_local = self.local_encoder(grid, mask_grid)
        z_day = self.day_encoder(grid, mask_grid)
        z_hour = self.hour_encoder(grid, mask_grid)
        z_interaction = self.interaction_encoder(z_day, z_hour)
        return {
            "grid": grid,
            "mask_grid": mask_grid,
            "z_global": z_global,
            "z_local": z_local,
            "z_day": z_day,
            "z_hour": z_hour,
            "z_interaction": z_interaction,
        }

    def _latent_grid(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        batch, days, hours = encoded["grid"].shape[:3]
        parts = []
        if self.use_global:
            parts.append(encoded["z_global"][:, None, None, :].expand(batch, days, hours, -1))
        if self.use_local:
            parts.append(encoded["z_local"])
        if self.use_day:
            parts.append(encoded["z_day"][:, :, None, :].expand(batch, days, hours, -1))
        if self.use_hour:
            parts.append(encoded["z_hour"][:, None, :, :].expand(batch, days, hours, -1))
        if self.use_interaction:
            parts.append(encoded["z_interaction"])
        return torch.cat(parts, dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode(x, mask)
        residual_hat = self.decode_from_encoded(encoded)
        out = {
            "residual_hat": residual_hat,
            "grid": encoded["grid"],
            "mask_grid": encoded["mask_grid"],
        }
        if self.use_global:
            out["z_global"] = encoded["z_global"]
        if self.use_local:
            out["z_local"] = encoded["z_local"]
        if self.use_day:
            out["z_day"] = encoded["z_day"]
        if self.use_hour:
            out["z_hour"] = encoded["z_hour"]
        if self.use_interaction:
            out["z_interaction"] = encoded["z_interaction"]
        return out

    def decode_from_encoded(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.decoder(self._latent_grid(encoded)).squeeze(-1)

    def decode_from_parts(
        self,
        reference: dict[str, torch.Tensor],
        z_global: torch.Tensor | None = None,
        z_local: torch.Tensor | None = None,
        z_day: torch.Tensor | None = None,
        z_hour: torch.Tensor | None = None,
        z_interaction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = dict(reference)
        if z_global is not None:
            encoded["z_global"] = z_global
        if z_local is not None:
            encoded["z_local"] = z_local
        if z_day is not None:
            encoded["z_day"] = z_day
        if z_hour is not None:
            encoded["z_hour"] = z_hour
        if z_interaction is not None:
            encoded["z_interaction"] = z_interaction
        if self.use_interaction and z_interaction is None and (z_day is not None or z_hour is not None):
            encoded["z_interaction"] = self.interaction_encoder(encoded["z_day"], encoded["z_hour"])
        return self.decode_from_encoded(encoded)


def residual_decouple_penalty(out: dict[str, torch.Tensor]) -> torch.Tensor:
    parts = [out.get("z_global"), out.get("z_local"), out.get("z_day"), out.get("z_hour")]
    return covariance_penalty([part for part in parts if part is not None])
