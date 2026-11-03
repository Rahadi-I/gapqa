"""DeepSEM-GLA v1 wrapper around a Hugging Face causal LM (Qwen2.5-1.5B in practice).

Four touch points (plan section 3.1), nothing else in the base model changes:
  1. true positions are passed as position_ids;
  2. gap tokens: the placeholder rows of inputs_embeds are replaced by GapEmbedding;
  3. logits only for the last n_target + 1 positions (logits_to_keep), loss computed here;
  4. (eval) attention mass on gap/marker keys, see eval/attention_mass.py.

Study 2 adds a fifth touch point: a gamma-lag bias on the attention logits computed from true positions
(gla/lag_bias.py). Arms GL/GLR/LIN carry learnable (a, c); with `GLAConfig.lag_attention=True` the arms
without a bias (B1, M0) run through the same masked attention path with a zero bias.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ...arms import ARMS

from .gap_module import GapEmbedding
from .lag_bias import _ATTR, GammaLagBias, attach, attention_modules, detach, heads_of


@dataclass
class GLAConfig:
    arm: str = "G"
    k_phi: int = 8
    gap_token: str = "<|fim_pad|>"
    init_seed: int = 0          # run seed: sets the noise in the initial gap vector
    lag_attention: bool = False  # Study 2: route every arm through the masked attention path (zero bias if none)


@torch.no_grad()
def embedding_table(base: nn.Module, chunk: int = 16384) -> torch.Tensor:
    """The embedding module's OUTPUT for every vocabulary id (float32, CPU). Gemma 3 scales its embeddings by
    sqrt(hidden) inside the module, so the raw weight matrix is not what the first layer sees; the gap vector must
    live on the same scale as the rows it replaces (KAT-2 checks this bit-exactly)."""
    mod = base.get_input_embeddings()
    n = mod.weight.shape[0]
    dev = mod.weight.device
    out = [mod(torch.arange(i, min(i + chunk, n), device=dev)).detach().float().cpu() for i in range(0, n, chunk)]
    return torch.cat(out, 0)


class GLAModel(nn.Module):
    def __init__(self, base: nn.Module, gap_token_id: int, cfg: GLAConfig = GLAConfig()):
        super().__init__()
        self.base = base
        self.cfg = cfg
        self.spec = ARMS[cfg.arm]
        self.gap_token_id = gap_token_id
        self.gap = None
        if self.spec.marker == "gap":
            emb = embedding_table(base)
            self.gap = GapEmbedding(emb.shape[1], cfg.k_phi, use_magnitude=self.spec.use_magnitude,
                                    init_vector=GapEmbedding.init_from_embeddings(emb, seed=cfg.init_seed))
        self.lag_bias = None
        self._attn_mods = attention_modules(base)
        self._native_attn = base.config._attn_implementation
        if self._native_attn == "gla_lag_sdpa":                      # base shared with an earlier wrapper
            self._native_attn = "sdpa"
        if self.spec.bias != "none" or cfg.lag_attention:
            L, H = heads_of(base)
            self.lag_bias = GammaLagBias(L, H, distance=self.spec.distance, learn_a=self.spec.bias == "gamma",
                                         learn_c=self.spec.bias in ("gamma", "linear"))
            # create the bias where the base model already lives (a wrapper built around a base that is already on
            # the GPU is not moved by a later .to(); fp32 regardless of the base dtype)
            dev = next(base.parameters()).device
            self.lag_bias.to(device=dev, dtype=torch.float32)
            attach(base, self.lag_bias)

    def _activate(self) -> None:
        """Several wrappers may share one base model (KAT scripts): make sure its attention modules point at THIS
        wrapper's bias (or at none, for a wrapper without the lag path) before every forward pass."""
        cur = self._attn_mods[0].__dict__.get(_ATTR)
        if self.lag_bias is None:
            if cur is not None:
                detach(self.base, self._native_attn)
        elif cur is not self.lag_bias:
            attach(self.base, self.lag_bias)

    def _begin_positions(self, position_ids, true_pos) -> None:
        self._activate()
        if self.lag_bias is not None:
            if true_pos is None:
                if self.spec.distance == "true" and not self.lag_bias.is_zero:
                    raise ValueError(f"arm {self.spec.name} needs true positions in the batch")
                true_pos = position_ids
            self.lag_bias.begin(true_pos, position_ids)

    def embed(self, input_ids: torch.Tensor, is_gap: torch.Tensor, gap_g: torch.Tensor) -> torch.Tensor:
        emb = self.base.get_input_embeddings()(input_ids)
        if self.gap is not None:
            emb = self.gap(emb, is_gap, gap_g)
        return emb

    def forward(self, input_ids, position_ids, is_gap, gap_g, logits_to_keep: int = 0, attention_mask=None,
                true_pos=None, **kw):
        emb = self.embed(input_ids, is_gap, gap_g)
        self._begin_positions(position_ids, true_pos)      # positions stay set for checkpoint recomputation
        # IMPORTANT (KAT-3d): transformers >= 4.5x treats non-contiguous position_ids as *packed sequences*
        # when attention_mask is None and there is no cache, i.e. every jump would start a new sequence and
        # tokens after a gap could not attend to tokens before it. An explicit all-ones mask disables that.
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return self.base(inputs_embeds=emb, position_ids=position_ids, attention_mask=attention_mask,
                         logits_to_keep=logits_to_keep, use_cache=False, **kw)

    def target_loss(self, batch: dict) -> torch.Tensor:
        """Cross-entropy on the target tokens only. Batch size 1 (no padding) in v1."""
        labels = batch["labels"]
        n_target = int((labels[0] != -100).sum())
        out = self(batch["input_ids"], batch["position_ids"], batch["is_gap"], batch["gap_g"],
                   logits_to_keep=n_target + 1, true_pos=batch.get("true_pos"))
        logits = out.logits[:, :-1].float()            # predict tokens t-n_target .. t-1
        tgt = labels[:, -n_target:]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def decision_score(self, batch: dict, answer_id: int, abstain_id: int) -> float:
        """s = p(ANSWER) / (p(ANSWER) + p(ABSTAIN)) at the first assistant position (one forward pass)."""
        out = self(batch["input_ids"], batch["position_ids"], batch["is_gap"], batch["gap_g"], logits_to_keep=1,
                   true_pos=batch.get("true_pos"))
        lp = out.logits[0, -1].float().log_softmax(-1)
        return float(torch.sigmoid(lp[answer_id] - lp[abstain_id]))

    @torch.no_grad()
    def greedy_decode(self, batch: dict, max_new: int, stop_ids: set[int], banned_ids: set[int]) -> list[int]:
        """Greedy continuation with a KV cache. Positions continue from the last prompt position; the explicit
        attention mask is extended each step (KAT-3d). Banned ids (the gap placeholder) get -inf logits."""
        ids, pos = batch["input_ids"], batch["position_ids"]
        tpos = batch.get("true_pos")
        emb = self.embed(ids, batch["is_gap"], batch["gap_g"])
        self._begin_positions(pos, tpos)
        mask = torch.ones_like(ids)
        out = self.base(inputs_embeds=emb, position_ids=pos, attention_mask=mask, use_cache=True, logits_to_keep=1)
        past, nxt_pos, gen = out.past_key_values, int(pos[0, -1]) + 1, []
        nxt_true = (int(tpos[0, -1]) if tpos is not None else int(pos[0, -1])) + 1
        logits = out.logits[0, -1].float()
        for _ in range(max_new):
            for b in banned_ids:
                logits[b] = float("-inf")
            t = int(logits.argmax())
            if t in stop_ids:
                break
            gen.append(t)
            if self.lag_bias is not None:
                self.lag_bias.extend(nxt_true, nxt_pos)
            nxt_true += 1
            mask = torch.cat([mask, mask.new_ones((1, 1))], dim=1)
            out = self.base(input_ids=torch.tensor([[t]], device=ids.device),
                            position_ids=torch.tensor([[nxt_pos]], device=ids.device),
                            attention_mask=mask, past_key_values=past, use_cache=True, logits_to_keep=1)
            past, nxt_pos = out.past_key_values, nxt_pos + 1
            logits = out.logits[0, -1].float()
        return gen

    def trainable_extra(self) -> list[nn.Parameter]:
        out = list(self.gap.parameters()) if self.gap is not None else []
        if self.lag_bias is not None:
            out += list(self.lag_bias.parameters())
        return out


def to_batch(seq, device="cpu", with_labels: bool = True) -> dict:
    """One Sequence -> tensors with batch dimension 1. For scoring, pass a sequence built without target."""
    t = lambda x, dt=torch.long: torch.tensor([x], dtype=dt, device=device)  # noqa: E731
    b = {"input_ids": t(seq.input_ids), "position_ids": t(seq.position_ids),
         "is_gap": t(seq.is_gap, torch.bool), "gap_g": t(seq.gap_g)}
    if getattr(seq, "true_pos", None) is not None:
        b["true_pos"] = t(seq.true_pos)
    if with_labels:
        b["labels"] = t(seq.labels)
    return b
