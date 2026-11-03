"""Data KATs for GapQA v1 (protocol v1.1, Phase P1 exit criterion) and descriptive statistics for the datasheet.

Every check returns a list of failures; the build is frozen only if all lists are empty.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .compose import norm_title
from .types import ItemV1

SINGLE_UNIT_EXTRACTIVE = {"squad", "tydi_id", "tydi_ar", "nq", "s1_kv"}


def kat_items(items: list[ItemV1], tok=None, t_obs: int = 3000, tol: int = 3) -> dict[str, list[str]]:
    fail = defaultdict(list)
    for it in items:
        if it.recompute_y_rec() != it.y_rec:
            fail["y_rec_recomputed"].append(it.id)
        if any(not (0 <= a < b <= len(it.text)) for a, b in it.evidence + it.gaps):
            fail["span_bounds"].append(it.id)
        gs = sorted(it.gaps)
        if any(gs[i][1] > gs[i + 1][0] for i in range(len(gs) - 1)):
            fail["gaps_overlap"].append(it.id)
        for a, b in it.evidence:                                   # no gap cuts an evidence unit partially
            for ga, ge in it.gaps:
                if a < ge and ga < b and not (ga <= a and b <= ge):
                    fail["partial_evidence_cut"].append(it.id)
        if it.source in SINGLE_UNIT_EXTRACTIVE and len(it.evidence) == 1:
            ev = it.text[slice(*it.evidence[0])]
            if not any(x in ev for x in it.answers):
                fail["answer_not_in_evidence"].append(it.id)
        if tok is not None:
            n = len(tok.encode(it.text))
            if abs(n - (t_obs + sum(it.g_ref))) > tol:
                fail["window_length"].append(f"{it.id}:{n}")
            for (a, b), g in zip(it.gaps, it.g_ref):
                if abs(len(tok.encode(it.text[a:b])) - g) > tol:
                    fail["gap_length_roundtrip"].append(it.id)
    return dict(fail)


def kat_pairs(items: list[ItemV1]) -> dict[str, list[str]]:
    fail = defaultdict(list)
    by = defaultdict(list)
    for it in items:
        by[it.pair_id].append(it)
    for pid, its in by.items():
        if len(its) != 2 or sorted(i.y_rec for i in its) != [0, 1]:
            fail["pair_composition"].append(pid)
            continue
        a, b = its
        if sorted(a.g_ref) != sorted(b.g_ref):
            fail["pair_gap_multiset"].append(pid)
        if a.meta["record"] == b.meta["record"] or a.source != b.source or a.split != b.split or a.slice != b.slice:
            fail["pair_members"].append(pid)
    recs = Counter(it.meta["record"] for it in items)
    fail["record_reused"] = [r for r, n in recs.items() if n > 1]
    return {k: v for k, v in fail.items() if v}


def kat_leakage(train_dev: list[ItemV1], test: list[ItemV1], test_titles: set[str] | None = None) -> dict[str, list[str]]:
    """No Wikipedia title of a test context in any train or dev context; train and dev disjoint by title too."""
    T = set(test_titles or ()) | {t for it in test for t in it.meta.get("titles", [])}
    fail = defaultdict(list)
    # train/dev are separated by the hash of the GOLD titles (a record's own distractor paragraphs may cross buckets)
    tr = {t for it in train_dev if it.split == "train" for t in it.meta.get("gold_titles", [])}
    dv = {t for it in train_dev if it.split == "dev" for t in it.meta.get("gold_titles", [])}
    for it in train_dev:
        if set(it.meta.get("titles", [])) & T:
            fail["test_title_in_train_dev"].append(it.id)
    if tr & dv:
        fail["train_dev_gold_title_overlap"] = sorted(tr & dv)[:50]
    q_test = {(it.source, it.question) for it in test}
    for it in train_dev:
        if (it.source, it.question) in q_test:
            fail["question_in_both"].append(it.id)
    return {k: v for k, v in fail.items() if v}


def smd(x: list[float], y: list[float]) -> float:
    """Standardised mean difference between strata (balance diagnostic; |SMD| < 0.1 is conventionally balanced)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2) if len(x) > 1 and len(y) > 1 else 0.0
    return float((x.mean() - y.mean()) / s) if s > 0 else 0.0


def ks_loguniform(g: list[int], lo: float, hi: float) -> float:
    """Kolmogorov-Smirnov distance between the gap lengths and the discretised LogUniform(lo, hi) the sampler draws
    from (floor(exp(u)), u ~ U(log lo, log(hi+1))): checks that pair formation, which redraws a multiset when no record
    can carry it, did not distort the pre-registered gap distribution."""
    x = np.sort(np.asarray(g, float))
    cdf = np.clip((np.log(x + 1) - np.log(lo)) / (np.log(hi + 1) - np.log(lo)), 0, 1)
    n = len(x)
    return float(max((np.arange(1, n + 1) / n - cdf).max(), (cdf - np.arange(n) / n).max())) if n else 0.0


def ks_null_95(n: int, lo: float, hi: float, reps: int = 300, seed: int = 0) -> float:
    """95th percentile of the KS statistic under the sampler itself (gaps come in multisets of K, so the n values are
    not independent draws; the asymptotic 1.36/sqrt(n) is not the right reference)."""
    from .compose import SliceSpec
    spec = SliceSpec("x", 5 if hi < 3000 else 3, int(lo), int(hi))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        g = []
        while len(g) < n:
            g += spec.sample(rng)
        out.append(ks_loguniform(g[:n], lo, hi))
    return float(np.percentile(out, 95))


def describe(items: list[ItemV1]) -> list[dict]:
    rows = []
    groups = defaultdict(list)
    for it in items:
        groups[(it.source, it.split, it.slice)].append(it)
    for (s, sp, sl), its in sorted(groups.items()):
        g = [x for it in its for x in it.g_ref]
        ev = {y: [sum(b - a for a, b in it.evidence) for it in its if it.y_rec == y] for y in (0, 1)}
        units = {y: [len(it.evidence) for it in its if it.y_rec == y] for y in (0, 1)}
        delf = [sum(b - a for a, b in it.gaps) / len(it.text) for it in its]
        rows.append({"source": s, "split": sp, "slice": sl, "n": len(its), "pairs": len({i.pair_id for i in its}),
                     "y1_share": round(float(np.mean([i.y_rec for i in its])), 3),
                     "K_mean": round(float(np.mean([len(i.g_ref) for i in its])), 2),
                     "g_median": int(np.median(g)), "g_min": int(min(g)), "g_max": int(max(g)),
                     "deleted_char_share_median": round(float(np.median(delf)), 3),
                     "evidence_chars_smd_y0_vs_y1": round(smd(ev[0], ev[1]), 3),
                     "evidence_units_smd_y0_vs_y1": round(smd(units[0], units[1]), 3),
                     "clusters": len({i.cluster for i in its}),
                     "ks_g_vs_loguniform": round(ks_loguniform(g, *((2049, 4096) if sl == "extrapolation" else (16, 2048))), 3),
                     "ks_null_95pct": round(ks_null_95(len(g), *((2049, 4096) if sl == "extrapolation" else (16, 2048))), 3)})
    return rows
