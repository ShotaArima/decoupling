from __future__ import annotations

import torch


_KERNEL_CACHE: dict[tuple[str, str, int, str, float], tuple[torch.Tensor, torch.Tensor]] = {}


def kernel_matrix(name: str, steps: int, scale: float, device: torch.device) -> torch.Tensor:
    """Build a zero-mean GP covariance matrix over window indices."""
    t = torch.arange(steps, device=device, dtype=torch.float32)
    dist = (t[:, None] - t[None, :]).abs()
    scale = max(float(scale), 1e-4)
    if name == "rbf":
        cov = torch.exp(-0.5 * (dist / scale) ** 2)
    elif name == "cauchy":
        cov = 1.0 / (1.0 + (dist / scale) ** 2)
    elif name == "periodic":
        cov = torch.exp(-2.0 * torch.sin(torch.pi * dist / scale).pow(2))
    else:
        raise ValueError(f"unknown kernel: {name}")
    eye = torch.eye(steps, device=device)
    return cov + 1e-4 * eye


def precision_and_logdet(
    name: str, steps: int, scale: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device.type, str(device.index), steps, name, float(scale))
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    k = kernel_matrix(name, steps, scale, device)
    precision = torch.linalg.inv(k)
    logdet = torch.linalg.slogdet(k).logabsdet
    _KERNEL_CACHE[key] = (precision, logdet)
    return precision, logdet


def kl_mvn_diag_to_gp(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    kernel_names: list[str],
    kernel_scales: list[float],
) -> torch.Tensor:
    """KL(q=N(mean, diag(var)) || p=N(0, K)) for each local latent dimension."""
    batch, steps, dims = mean.shape
    if len(kernel_names) < dims or len(kernel_scales) < dims:
        raise ValueError("kernel_names and kernel_scales must cover local_dim")

    total = mean.new_zeros(())
    for j in range(dims):
        precision, logdet_k = precision_and_logdet(kernel_names[j], steps, kernel_scales[j], mean.device)
        var = logvar[:, :, j].exp()
        mu = mean[:, :, j]
        trace_term = torch.einsum("st,bt->b", precision, var)
        quad_term = torch.einsum("bs,st,bt->b", mu, precision, mu)
        logdet_q = logvar[:, :, j].sum(dim=1)
        total = total + 0.5 * (trace_term + quad_term - steps + logdet_k - logdet_q).mean()
    return total / dims


def gp_posterior_mean(
    observed_z: torch.Tensor,
    forecast_steps: int,
    kernel_names: list[str],
    kernel_scales: list[float],
) -> torch.Tensor:
    """Forecast future local latents using the GP conditional mean in the paper."""
    observed_z = observed_z.float()
    batch, steps, dims = observed_z.shape
    device = observed_z.device
    all_steps = steps + forecast_steps
    pred = []
    for j in range(dims):
        k_all = kernel_matrix(kernel_names[j], all_steps, kernel_scales[j], device)
        k_oo = k_all[:steps, :steps]
        k_fo = k_all[steps:, :steps]
        alpha = torch.linalg.solve(k_oo, observed_z[:, :, j].T).T
        pred.append(torch.matmul(alpha, k_fo.T))
    return torch.stack(pred, dim=-1)
