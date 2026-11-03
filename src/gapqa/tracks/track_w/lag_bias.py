"""Gamma-lag attention bias (DeepSEM-GLA Study 2, protocol v2.3, Section 3.2).

Each layer l and head h adds to its attention logits, for query i and key j <= i,

    b_h^(l)(D) = a_h^(l) * ln(1 + D) - c_h^(l) * D / LAMBDA,     LAMBDA = 1000,   a = c = 0 at initialisation,

where D is a *measured* token distance: D_true = p_true_i - p_true_j (true source positions, so a hidden span
of g tokens counts g) for arms GL and LIN, or D_vis = p_vis_i - p_vis_j (contiguous positions) for arm GLR.
With alpha = 1 + a and beta = c / LAMBDA, exp(b) is proportional to (1 + D)^(alpha - 1) exp(-beta D): a gamma
log-prior over lags (Almon 1965; Koyck 1954), ALiBi-type linear decay when a = 0 (Press et al. 2022).

Mechanics. A custom attention function is registered with transformers' AttentionInterface under the name
ATTN_NAME. It receives the layer's attention module (which carries `layer_idx` and a reference to the
GammaLagBias instance, attached by `attach`), computes this layer's [1, H, q, kv] bias from (a, c) and the
positions set by `begin`/`extend`, folds it into a float additive mask together with the causal (or given)
mask, casts the mask to the attention dtype and calls transformers' own sdpa function with it. The kernel is
unchanged; only its mask argument changes. Every Study-2 arm, B1 and M0 included, runs through this path with
a zero bias, so that a = c = 0 reproduces B1 bit-for-bit on the same hardware (KAT i) and the arms differ in
nothing but the bias values. Rotary embeddings never see true positions (they get position_ids as before).

Parameters are plain nn.Parameters without a positivity transform (zero must be reachable exactly; grade C).
LIN fixes a at zero (a buffer); B1/M0 fix both. Gradients reach (a, c) through the mask argument of
scaled_dot_product_attention (math and memory-efficient backends support a differentiable mask; KAT iv checks
on CPU, the pilot checks on the GPU that ||a||, ||c|| move).
"""
from __future__ import annotations

import math

import torch
from torch import nn

LAMBDA = 1000.0
ATTN_NAME = "gla_lag_sdpa"
_ATTR = "_gla_lag_bias"          # attribute on each attention module pointing at the GammaLagBias instance


def gamma_lag_bias(delta: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """delta: [..., q, kv] (any dtype, negative values are clamped to 0 and masked by the caller);
    a, c: [H] -> bias [..., H, q, kv] in float32."""
    d = delta.clamp_min(0).float()
    d = d.unsqueeze(-3)                                        # [..., 1, q, kv]
    av = a.float().view(*([1] * (d.dim() - 3)), -1, 1, 1)
    cv = c.float().view(*([1] * (d.dim() - 3)), -1, 1, 1)
    return av * torch.log1p(d) - cv * (d / LAMBDA)


def causal_allowed(q_len: int, kv_len: int, device) -> torch.Tensor:
    """[q, kv] bool: query i (the (kv_len - q_len + i)-th key) may attend to keys j <= kv_len - q_len + i."""
    qi = torch.arange(q_len, device=device)[:, None] + (kv_len - q_len)
    kj = torch.arange(kv_len, device=device)[None, :]
    return kj <= qi


def combine_mask(bias: torch.Tensor, attention_mask: torch.Tensor | None, q_len: int, kv_len: int,
                 dtype: torch.dtype) -> torch.Tensor:
    """Additive float mask [B, H, q, kv] in `dtype`: bias where attention is allowed, finfo.min elsewhere.
    `attention_mask` is transformers' 4-D mask for this layer (bool, or float with finfo.min at masked
    entries), or None (then plain causal)."""
    if attention_mask is None:
        allowed = causal_allowed(q_len, kv_len, bias.device).view(1, 1, q_len, kv_len)
    else:
        m = attention_mask[:, :, -q_len:, :kv_len]
        allowed = m if m.dtype == torch.bool else (m > torch.finfo(m.dtype).min / 2)
    neg = torch.finfo(dtype).min
    out = torch.where(allowed, bias.to(dtype), torch.full((), neg, dtype=dtype, device=bias.device))
    return out


class GammaLagBias(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, distance: str = "true", learn_a: bool = True,
                 learn_c: bool = True):
        super().__init__()
        if distance not in ("true", "vis"):
            raise ValueError(distance)
        self.n_layers, self.n_heads, self.distance = n_layers, n_heads, distance
        self.learn_a, self.learn_c = learn_a, learn_c
        z = torch.zeros(n_layers, n_heads)
        if learn_a:
            self.a = nn.Parameter(z.clone())
        else:
            self.register_buffer("a", z.clone())
        if learn_c:
            self.c = nn.Parameter(z.clone())
        else:
            self.register_buffer("c", z.clone())
        self._pos: torch.Tensor | None = None       # [B, N] positions of the distance in use (true or vis)
        self.calls = 0                              # attention calls since the last begin (diagnostics)

    # ------------------------------------------------------------------ positions
    def begin(self, true_pos: torch.Tensor, vis_pos: torch.Tensor) -> None:
        """Positions of the full prompt, [B, T] long, before a forward pass."""
        self._pos = (true_pos if self.distance == "true" else vis_pos).to(torch.long)
        self.calls = 0

    def extend(self, true_next: int, vis_next: int) -> None:
        """Append the position of the next generated token (cached decoding)."""
        if self._pos is None:
            raise RuntimeError("extend() before begin()")
        p = true_next if self.distance == "true" else vis_next
        self._pos = torch.cat([self._pos, self._pos.new_full((self._pos.shape[0], 1), p)], dim=1)

    def end(self) -> None:
        self._pos = None

    @property
    def is_zero(self) -> bool:
        return not (self.learn_a or self.learn_c)

    # ------------------------------------------------------------------ bias
    def layer_bias(self, layer_idx: int, q_len: int, kv_len: int) -> torch.Tensor:
        """[B, H, q, kv] float32 bias for one layer from the stored positions."""
        if self._pos is None:
            raise RuntimeError("positions not set: call begin() before the forward pass")
        if self._pos.shape[1] < kv_len:
            raise RuntimeError(f"positions cover {self._pos.shape[1]} tokens, attention sees {kv_len} keys")
        if self._pos.device != self.a.device:          # positions follow the parameters (defensive)
            self._pos = self._pos.to(self.a.device)
        pk = self._pos[:, :kv_len]
        pq = self._pos[:, kv_len - q_len: kv_len]
        delta = pq[:, :, None] - pk[:, None, :]                     # [B, q, kv]
        return gamma_lag_bias(delta, self.a[layer_idx], self.c[layer_idx])

    # ------------------------------------------------------------------ readouts (E1)
    def alpha_beta(self) -> tuple[torch.Tensor, torch.Tensor]:
        return 1.0 + self.a.detach().float(), self.c.detach().float() / LAMBDA

    def modes(self) -> torch.Tensor:
        """Delta* = (alpha - 1) / beta - 1 per layer and head; NaN where the gamma reading does not hold
        (alpha <= 1 or beta <= 0)."""
        alpha, beta = self.alpha_beta()
        ok = (alpha > 1) & (beta > 0)
        m = (alpha - 1) / beta.clamp_min(1e-12) - 1
        return torch.where(ok, m, torch.full_like(m, float("nan")))

    def summary(self) -> dict:
        alpha, beta = self.alpha_beta()
        m = self.modes()
        return {"norm_a": float(self.a.detach().float().norm()), "norm_c": float(self.c.detach().float().norm()),
                "max_abs_a": float(self.a.detach().abs().max()), "max_abs_c": float(self.c.detach().abs().max()),
                "share_gamma_reading": float(((alpha > 0) & (beta > 0)).float().mean()),
                "share_with_mode": float((~torch.isnan(m)).float().mean()),
                "n_heads": int(m.numel())}


# ---------------------------------------------------------------------- attention function
def _lag_sdpa(module, query, key, value, attention_mask, **kw):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    bias_mod: GammaLagBias | None = getattr(module, _ATTR, None)
    if bias_mod is None:                                   # a model without the bias attached: plain sdpa
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
    q_len, kv_len = query.shape[2], key.shape[2]
    bias = bias_mod.layer_bias(module.layer_idx, q_len, kv_len)
    mask = combine_mask(bias, attention_mask, q_len, kv_len, query.dtype)
    bias_mod.calls += 1
    kw["is_causal"] = False
    return sdpa_attention_forward(module, query, key, value, mask, **kw)


_registered = False


def register() -> str:
    """Register the attention function once; returns its name for config._attn_implementation."""
    global _registered
    if not _registered:
        from transformers import AttentionInterface
        AttentionInterface.register(ATTN_NAME, _lag_sdpa)
        _registered = True
    return ATTN_NAME


def attention_modules(base: nn.Module) -> list[nn.Module]:
    mods = [m for m in base.modules() if hasattr(m, "layer_idx") and type(m).__name__.endswith("Attention")]
    if not mods:
        raise RuntimeError("no attention modules with layer_idx found")
    return mods


def attach(base: nn.Module, bias: GammaLagBias) -> None:
    """Point every attention module at `bias` and switch the model's attention implementation to the lag path."""
    name = register()
    mods = attention_modules(base)
    if len(mods) != bias.n_layers:
        raise ValueError(f"model has {len(mods)} attention layers, bias has {bias.n_layers}")
    for m in mods:
        object.__setattr__(m, _ATTR, bias)      # a plain attribute, NOT a registered submodule: the bias must not
        #                                          appear in the base model's parameters, state dict or LoRA count
    cfg = base.config
    cfg._attn_implementation = name
    # PEFT wrappers and some models keep a second config reference; set them all
    for m in base.modules():
        c = getattr(m, "config", None)
        if c is not None and c is not cfg and hasattr(c, "_attn_implementation"):
            try:
                c._attn_implementation = name
            except Exception:
                pass


def detach(base: nn.Module, attn_implementation: str = "sdpa") -> None:
    for m in attention_modules(base):
        if _ATTR in m.__dict__:
            object.__delattr__(m, _ATTR)
    base.config._attn_implementation = attn_implementation


def heads_of(base: nn.Module) -> tuple[int, int]:
    cfg = base.config
    cfg = getattr(cfg, "text_config", cfg)
    return int(cfg.num_hidden_layers), int(cfg.num_attention_heads)


def mode_of_gamma(k: float, theta: float) -> float:
    """Mode of a Gamma(k, theta) inter-arrival law, (k - 1) theta for k > 1 (used by E1)."""
    return (k - 1) * theta if k > 1 else math.nan
