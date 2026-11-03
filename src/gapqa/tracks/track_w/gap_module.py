"""Magnitude-bearing gap token (protocol v1.1, Section 6; no tau term in v1).

    e_gap(g) = e_[GAP] + W_g * phi(u),  u = log(1 + g),
    phi(u)   = [u / u_max, sin(2^k u), cos(2^k u)]_{k=0..K-1},  u_max = log(1 + 2048)   (2K + 1 dims)

The linear term lets the magnitude extrapolate beyond the training range of g; periodic features
alone alias there (Tancik et al., 2020).

The vocabulary is not resized (Qwen2.5 ties input embeddings and lm_head). A spare
special token (<|fim_pad|>) is used as the placeholder id; its row is replaced in the
inputs_embeds path by this module. W_g is zero-initialised, so arm G starts exactly as G0.
"""
from __future__ import annotations

import math

import torch
from torch import nn


U_MAX = math.log1p(2048.0)


def fourier_features(u: torch.Tensor, k: int) -> torch.Tensor:
    """u: [...] -> [..., 2k + 1]: a linear term u / U_MAX, then sin and cos at frequencies 2^0 .. 2^(k-1)."""
    freqs = 2.0 ** torch.arange(k, device=u.device, dtype=torch.float32)
    x = u.float()[..., None] * freqs
    return torch.cat([u.float()[..., None] / U_MAX, torch.sin(x), torch.cos(x)], dim=-1)


class GapEmbedding(nn.Module):
    def __init__(self, hidden_size: int, k_phi: int = 8, use_magnitude: bool = True,
                 init_vector: torch.Tensor | None = None):
        super().__init__()
        self.k_phi = k_phi
        self.use_magnitude = use_magnitude
        init = init_vector.detach().clone().float() if init_vector is not None else torch.zeros(hidden_size)
        self.e_gap = nn.Parameter(init)
        self.w_g = nn.Linear(2 * k_phi + 1, hidden_size)
        nn.init.zeros_(self.w_g.weight)
        nn.init.zeros_(self.w_g.bias)

    @staticmethod
    def init_from_embeddings(weight: torch.Tensor, noise: float = 0.02, seed: int = 0) -> torch.Tensor:
        """Mean embedding plus Gaussian noise with std = noise x (std of all embedding entries) (protocol 6).
        The seed is the run seed, so the initial gap vector differs across seeds as pre-registered."""
        g = torch.Generator().manual_seed(seed)
        w = weight.detach().float().cpu()
        mean = w.mean(0)
        return mean + noise * w.std() * torch.randn(mean.shape, generator=g)

    def gap_vectors(self, g: torch.Tensor) -> torch.Tensor:
        """g: [n] long -> [n, hidden]."""
        v = self.e_gap.expand(g.shape[0], -1)
        if self.use_magnitude:
            v = v + self.w_g(fourier_features(torch.log1p(g.float()), self.k_phi))
        return v

    def forward(self, emb: torch.Tensor, is_gap: torch.Tensor, gap_g: torch.Tensor) -> torch.Tensor:
        """Replace the rows of `emb` [B, T, H] where is_gap [B, T] is True."""
        if not bool(is_gap.any()):
            return emb
        out = emb.clone()
        vec = self.gap_vectors(gap_g[is_gap]).to(emb.dtype)
        out[is_gap] = vec
        return out
