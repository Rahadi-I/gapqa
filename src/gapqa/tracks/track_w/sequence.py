"""Sequence assembly for GapQA v1 items with any chat template (protocol v1.1, Sections 6 and 7)."""
from __future__ import annotations

from dataclasses import dataclass

from ...item import ItemV1
from ...render import render_v1
from ...tokenization import Tokenizer
from .chat import ChatFormat


@dataclass
class Sequence:
    input_ids: list[int]
    position_ids: list[int]
    is_gap: list[bool]
    is_marker: list[bool]
    gap_g: list[int]
    labels: list[int]          # -100 everywhere except the target tokens
    n_prompt: int              # index of the first assistant (target) token
    n_target: int
    true_pos: list[int] | None = None   # true positions (Study 2 lag bias); None for v0 sequences


def build_sequence_v1(item: ItemV1, arm: str, tok: Tokenizer, chat: ChatFormat, gap_token_id: int, *,
                      with_target: bool = True, forced_answer_prefix: bool = False) -> Sequence:
    """Prefix gets positions 0..P-1; the context is rendered from P; suffix and target continue after it.

    forced_answer_prefix: append the tokens of "ANSWER:" after the assistant header (answer extraction at
    evaluation, Section 7.1); implies with_target=False.
    """
    prefix = tok.encode(chat.pre)
    p = len(prefix)
    ctx = render_v1(item, arm, tok, gap_token_id, pos_offset=p)
    suffix = tok.encode(chat.mid + item.question + chat.post)
    if forced_answer_prefix:
        suffix = suffix + tok.encode("ANSWER:")
        with_target = False
    target = tok.encode(chat.target(item.y_rec, item.answers[0])) if with_target else []
    ids = prefix + ctx.ids + suffix + target
    tail = len(suffix) + len(target)
    pos = list(range(p)) + ctx.pos + list(range(ctx.next_pos, ctx.next_pos + tail))
    tpos = list(range(p)) + ctx.true_pos + list(range(ctx.true_next, ctx.true_next + tail))
    is_gap = [False] * p + ctx.is_gap + [False] * tail
    is_marker = [False] * p + ctx.is_marker + [False] * tail
    gap_g = [0] * p + ctx.gap_g + [0] * tail
    n_prompt = len(ids) - len(target)
    labels = [-100] * n_prompt + target
    assert len(ids) == len(pos) == len(is_gap) == len(labels)
    assert all(b > a for a, b in zip(pos, pos[1:])), "positions must be strictly increasing"
    assert len(tpos) == len(ids) and all(b > a for a, b in zip(tpos, tpos[1:]))
    return Sequence(ids, pos, is_gap, is_marker, gap_g, labels, n_prompt, len(target), tpos)


def decision_token_ids_v1(tok: Tokenizer, chat: ChatFormat) -> tuple[int, int]:
    a = tok.encode(chat.target(1, "x"))[0]
    b = tok.encode(chat.target(0, ""))[0]
    if a == b:
        raise ValueError(f"{tok.name}: ANSWER and ABSTAIN share their first token")
    return a, b
