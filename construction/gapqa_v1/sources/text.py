"""Sentence boundaries for English, Indonesian and Arabic, and helpers to locate evidence.

Lesson from the v0 pilot (protocol v1.1, Section 5.6): a naive splitter cuts evidence at abbreviations and initials, so
part of the evidence would survive a gap meant to delete it. Periods after known abbreviations and initials are
therefore not boundaries. Arabic has no letter case, so a boundary there is any sentence-final mark followed by space.
"""
from __future__ import annotations

import re

_ABBREV_EN = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "ft", "no", "vs", "etc", "e.g", "i.e", "cf",
    "ca", "fl", "approx", "est", "vol", "ch", "fig", "op", "rev", "gen", "col", "lt", "capt", "sgt", "gov", "sen",
    "rep", "pres", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "inc", "ltd",
    "co", "corp", "dept", "univ", "u.s", "u.k", "u.n", "a.d", "b.c", "a.m", "p.m", "ph.d", "al", "eq", "sec",
}
_ABBREV_ID = {
    "dll", "dsb", "dst", "tsb", "yg", "dgn", "hlm", "no", "kab", "kec", "prov", "jl", "jln", "drs", "dra", "ir", "h",
    "hj", "kh", "m", "sm", "skt", "bpk", "sdr", "tn", "ny", "nn", "st", "s.h", "s.e", "s.t", "m.a", "m.si", "dr",
    "prof", "ra", "saw", "swt", "as", "ra", "kg", "km", "cm", "mm", "rp",
}
_ABBREV = {"en": _ABBREV_EN, "id": _ABBREV_EN | _ABBREV_ID, "ar": set()}

_BOUNDARY_CASED = re.compile(r"[.!?][\"')\]”’]*\s+(?=[\"'(\[“‘]?[A-Z0-9])")
_BOUNDARY_AR = re.compile(r"[.!?؟۔][\"')\]»”]*\s+")


def _is_abbreviation(text: str, period_idx: int, lang: str) -> bool:
    j = period_idx
    while j > 0 and not text[j - 1].isspace() and text[j - 1] not in "(\"'[":
        j -= 1
    word = text[j:period_idx].lower()
    if word in _ABBREV.get(lang, set()):
        return True
    if lang == "ar":
        return False
    return (len(word) == 1 and word.isalpha()) or ("." in word and len(word) <= 6)


def sentence_bounds(text: str, lang: str = "en") -> list[tuple[int, int]]:
    """Character spans [start, end) of sentences; whitespace between sentences belongs to neither."""
    pat = _BOUNDARY_AR if lang == "ar" else _BOUNDARY_CASED
    bounds, start = [], 0
    for m in pat.finditer(text):
        if text[m.start()] == "." and _is_abbreviation(text, m.start(), lang):
            continue
        end = m.start() + len(m.group(0).rstrip())
        if end > start:
            bounds.append((start, end))
        start = m.end()
    if start < len(text):
        bounds.append((start, len(text.rstrip()) if text.rstrip() else len(text)))
    return [(a, b) for a, b in bounds if b > a]


def covering_sentences(text: str, a0: int, a1: int, lang: str = "en") -> tuple[int, int]:
    """Union of the sentences that overlap characters [a0, a1) (the evidence unit of SQuAD and TyDi GoldP)."""
    hits = [(s, e) for s, e in sentence_bounds(text, lang) if s < a1 and a0 < e]
    if not hits:
        raise ValueError("answer span outside every sentence")
    return hits[0][0], hits[-1][1]


def join_sentences(sents: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join pre-split sentences (HotpotQA, 2Wiki) with single spaces; return the text and each sentence's span."""
    parts, spans, cur = [], [], 0
    for s in sents:
        s = s.strip()
        if parts:
            cur += 1
        spans.append((cur, cur + len(s)))
        parts.append(s)
        cur += len(s)
    return " ".join(parts), spans


_WS = re.compile(r"\s+")


def normalise_ws(s: str) -> str:
    return _WS.sub(" ", s).strip()
