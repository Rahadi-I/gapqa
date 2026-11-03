"""Tokenizer abstraction.

The pipeline only needs: encode(text) -> ids, encode_with_offsets(text) -> (ids, char offsets),
and decode(ids). `HFTokenizer` wraps a Hugging Face tokenizer (Qwen2.5 in practice);
`WhitespaceTokenizer` is a deterministic stand-in used by the tests so that the data
logic can be checked without downloading anything.
"""
from __future__ import annotations

import re
from typing import Protocol


class Tokenizer(Protocol):
    name: str

    def encode(self, text: str) -> list[int]: ...

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]: ...

    def decode(self, ids: list[int]) -> str: ...


class HFTokenizer:
    """Thin wrapper around a Hugging Face fast tokenizer (no special tokens added)."""

    def __init__(self, name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct", revision: str | None = None):
        from transformers import AutoTokenizer  # lazy import

        self.tok = AutoTokenizer.from_pretrained(name_or_path, revision=revision)
        self.name = name_or_path

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    def token_id(self, token: str) -> int:
        return int(self.tok.convert_tokens_to_ids(token))


class WhitespaceTokenizer:
    """Deterministic test tokenizer: words, digits and punctuation, each with leading space.

    Digits are split one per token, mimicking Qwen2.5 so that the literal marker
    "[3443 tokens omitted]" has a g-dependent length, as it does in the real model.
    """

    _pat = re.compile(r"\s*(\d|\w+|[^\w\s])")

    def __init__(self) -> None:
        self.name = "whitespace-test"
        self.vocab: dict[str, int] = {}
        self.inv: list[str] = []

    def _id(self, piece: str) -> int:
        if piece not in self.vocab:
            self.vocab[piece] = len(self.inv)
            self.inv.append(piece)
        return self.vocab[piece]

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids, offs = [], []
        for m in self._pat.finditer(text):
            ids.append(self._id(m.group(0)))
            offs.append((m.start(), m.end()))
        return ids, offs

    def encode(self, text: str) -> list[int]:
        return self.encode_with_offsets(text)[0]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inv[i] for i in ids)


def char_span_to_token_span(offsets: list[tuple[int, int]], c0: int, c1: int) -> tuple[int, int]:
    """Smallest token span [t0, t1) whose characters cover [c0, c1)."""
    t0 = next(i for i, (s, e) in enumerate(offsets) if e > c0)
    t1 = max(i for i, (s, e) in enumerate(offsets) if s < c1) + 1
    return t0, t1
