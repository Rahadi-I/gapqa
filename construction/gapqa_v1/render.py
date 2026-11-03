"""Render a v1 item for one model and one arm (same six arms as v0, see gapqa.render.ARMS).

Observed pieces are tokenized separately with the model's tokenizer; each deleted piece is counted in the
same tokenizer (g_model), which sets both the true-position jump and the magnitude shown by B1/B1P/G.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..render import ARMS, ArmSpec, marker_text
from ..tokenization import Tokenizer
from .types import ItemV1


@dataclass
class RenderedV1:
    ids: list[int]
    pos: list[int]
    is_gap: list[bool]
    is_marker: list[bool]
    is_observed: list[bool]
    gap_g: list[int]
    next_pos: int
    g_model: list[int]          # deleted length of each gap in this model's tokens
    true_pos: list[int] = None  # true source position of every token (markers inside their span)
    true_next: int = 0          # true position of the first token after the context


def gap_lengths(item: ItemV1, tok: Tokenizer) -> list[int]:
    return [len(tok.encode(t)) for kind, t, _ in item.observed_and_deleted() if kind == "gap"]


def render_v1(item: ItemV1, arm: str | ArmSpec, tok: Tokenizer, gap_token_id: int, pos_offset: int = 0) -> RenderedV1:
    spec = ARMS[arm] if isinstance(arm, str) else arm
    ids, pos, is_gap, is_marker, is_obs, gap_g, g_model = [], [], [], [], [], [], []
    src_pos = pos_offset                                   # true position of the next source token
    for kind, text, _ in item.observed_and_deleted():
        t = tok.encode(text)
        if kind == "obs":
            for x in t:
                ids.append(x); pos.append(src_pos); src_pos += 1
                is_gap.append(False); is_marker.append(False); is_obs.append(True); gap_g.append(0)
            continue
        g = len(t)
        if g == 0:
            raise ValueError(f"{item.id}: a deleted span is empty in this tokenizer")
        g_model.append(g)
        if spec.marker == "literal":
            m = tok.encode(marker_text(g, spec.count))
            if len(m) > g:
                raise ValueError(f"{item.id}: marker ({len(m)} tokens) longer than gap ({g}) in {tok.name}")
            for j, x in enumerate(m):
                ids.append(x); pos.append(src_pos + j)
                is_gap.append(False); is_marker.append(True); is_obs.append(False); gap_g.append(g)
        elif spec.marker == "gap":
            ids.append(gap_token_id); pos.append(src_pos)
            is_gap.append(True); is_marker.append(True); is_obs.append(False); gap_g.append(g)
        elif spec.marker != "none":
            raise ValueError(spec.marker)
        src_pos += g
    tpos, true_next = list(pos), src_pos           # the loop above always numbers true positions
    if spec.positions == "renumber":
        pos = list(range(pos_offset, pos_offset + len(ids)))
        next_pos = pos_offset + len(ids)
    else:
        next_pos = src_pos
    return RenderedV1(ids, pos, is_gap, is_marker, is_obs, gap_g, next_pos, g_model, tpos, true_next)
