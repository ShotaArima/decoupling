from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .kernels import gp_posterior_mean, kl_mvn_diag_to_gp


@dataclass
class GLRBatch:
    x: torch.Tensor
    mask: torch.Tensor
    static_ids: torch.Tensor | None = None
    target: torch.Tensor | None = None
    subgroup: torch.Tensor | None = None


class WindowMLP(nn.Module):
    def __init__(self, input_dim: int, window_size: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * window_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        batch, steps, features, width = windows.shape
        return self.net(windows.reshape(batch, steps, features * width))


class LocalEncoder(nn.Module):
    def __init__(self, input_dim: int, window_size: int, hidden_dim: int, local_dim: int):
        super().__init__()
        self.local_dim = local_dim
        self.encoder = WindowMLP(input_dim, window_size, hidden_dim, local_dim * 2)

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.encoder(windows)
        mean, raw_logvar = params.chunk(2, dim=-1)
        logvar = raw_logvar.clamp(-6.0, 3.0)
        return mean, logvar


class GlobalEncoder(nn.Module):
    def __init__(self, input_dim: int, window_size: int, hidden_dim: int, global_dim: int):
        super().__init__()
        self.window = WindowMLP(input_dim, window_size, hidden_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, global_dim * 2)

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.window(windows)
        _, last = self.rnn(h)
        mean, raw_logvar = self.head(last[-1]).chunk(2, dim=-1)
        return mean, raw_logvar.clamp(-6.0, 3.0)


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        window_size: int,
        hidden_dim: int,
        local_dim: int,
        global_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.net = nn.Sequential(
            nn.Linear(local_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * window_size),
        )

    def forward(self, z_local: torch.Tensor, z_global: torch.Tensor) -> torch.Tensor:
        steps = z_local.shape[1]
        zg = z_global[:, None, :].expand(-1, steps, -1)
        out = self.net(torch.cat([z_local, zg], dim=-1))
        return out.reshape(z_local.shape[0], steps, self.input_dim, self.window_size)


class GLRModel(nn.Module):
    """VAE with decoupled local GP latents and one global latent per series."""

    def __init__(
        self,
        input_dim: int,
        window_size: int,
        global_dim: int,
        local_dim: int,
        hidden_dim: int,
        beta: float,
        lambda_counterfactual: float,
        kernel_names: list[str],
        kernel_scales: list[float],
        decoder_std: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.beta = beta
        self.lambda_counterfactual = lambda_counterfactual
        self.kernel_names = kernel_names
        self.kernel_scales = kernel_scales
        self.decoder_std = decoder_std
        self.local_encoder = LocalEncoder(input_dim, window_size, hidden_dim, local_dim)
        self.global_encoder = GlobalEncoder(input_dim, window_size, hidden_dim, global_dim)
        self.decoder = Decoder(input_dim, window_size, hidden_dim, local_dim, global_dim)

    def _window(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self.window_size != 0:
            raise ValueError("time dimension must be divisible by window_size")
        steps = x.shape[-1] // self.window_size
        return x.reshape(x.shape[0], x.shape[1], steps, self.window_size).permute(0, 2, 1, 3)

    @staticmethod
    def _sample(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    @staticmethod
    def _kl_standard_normal(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(torch.sum(1.0 + logvar - mean.pow(2) - logvar.exp(), dim=-1))

    @staticmethod
    def _log_prob_diag(value: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum((value - mean).pow(2) / logvar.exp() + logvar + math.log(2.0 * math.pi), dim=-1)

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        windows = self._window(x)
        zl_mean, zl_logvar = self.local_encoder(windows)
        zg_mean, zg_logvar = self.global_encoder(windows)
        return {
            "windows": windows,
            "zl_mean": zl_mean,
            "zl_logvar": zl_logvar,
            "zg_mean": zg_mean,
            "zg_logvar": zg_logvar,
        }

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode(x)
        z_local = self._sample(encoded["zl_mean"], encoded["zl_logvar"])
        z_global = self._sample(encoded["zg_mean"], encoded["zg_logvar"])
        recon = self.decoder(z_local, z_global)
        mask_w = self._window(mask)
        nll = 0.5 * ((recon - encoded["windows"]) / self.decoder_std).pow(2)
        reconstruction = (nll * mask_w).sum() / mask_w.sum().clamp_min(1.0)
        kl_local = kl_mvn_diag_to_gp(
            encoded["zl_mean"], encoded["zl_logvar"], self.kernel_names, self.kernel_scales
        )
        kl_global = self._kl_standard_normal(encoded["zg_mean"], encoded["zg_logvar"])
        cf_global = torch.randn_like(z_global)
        counterfactual = self.decoder(z_local, cf_global)
        cf_mean, cf_logvar = self.global_encoder(counterfactual)
        log_q_original = self._log_prob_diag(z_global.detach(), cf_mean, cf_logvar)
        log_q_cf = self._log_prob_diag(cf_global.detach(), cf_mean, cf_logvar)
        cf_loss = F.softplus(log_q_original - log_q_cf).mean()
        loss = reconstruction + self.beta * (kl_local + kl_global) + self.lambda_counterfactual * cf_loss
        return {
            "loss": loss,
            "reconstruction": reconstruction.detach(),
            "kl_local": kl_local.detach(),
            "kl_global": kl_global.detach(),
            "counterfactual": cf_loss.detach(),
            "recon": recon,
            **encoded,
        }

    @torch.no_grad()
    def forecast(self, x: torch.Tensor, forecast_windows: int) -> torch.Tensor:
        encoded = self.encode(x)
        future_z = gp_posterior_mean(
            encoded["zl_mean"], forecast_windows, self.kernel_names, self.kernel_scales
        )
        return self.decoder(future_z, encoded["zg_mean"])
