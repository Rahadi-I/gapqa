"""Selective-prediction metrics (plan section 3.3).

Conventions: `conf` is the confidence of *answering* (s = p(ANSWER) / (p(ANSWER) + p(ABSTAIN)));
`loss` is the per-item error if the item were answered, e_i in {0, 1}:
    e_i = 1  if y_rec = 0                       (any answer over a hole is a grounded claim on nothing)
    e_i = 1  if y_rec = 1 and EM(answer) = 0
    e_i = 0  otherwise.
Items are answered in order of decreasing confidence; ties are broken by averaging over the tie
block, so the result does not depend on the input order.
"""
from __future__ import annotations

import numpy as np


def _sorted_blocks(conf: np.ndarray):
    order = np.argsort(-conf, kind="stable")
    c = conf[order]
    starts = np.r_[0, np.flatnonzero(np.diff(c) != 0) + 1]
    ends = np.r_[starts[1:], len(c)]
    return order, starts, ends


def risk_coverage(conf, loss) -> tuple[np.ndarray, np.ndarray]:
    """Risk at each coverage k/n, k = 1..n, with tie blocks averaged (expected over random tie-breaks)."""
    conf = np.asarray(conf, float)
    loss = np.asarray(loss, float)
    n = len(conf)
    if n == 0:
        raise ValueError("empty input")
    order, starts, ends = _sorted_blocks(conf)
    l_sorted = loss[order].copy()
    for s, e in zip(starts, ends):             # within a tie block every order is equally likely
        l_sorted[s:e] = l_sorted[s:e].mean()
    cum = np.cumsum(l_sorted)
    k = np.arange(1, n + 1)
    return k / n, cum / k


def aurc(conf, loss) -> float:
    """Area under the risk-coverage curve, mean of risk over coverages 1/n..1 (Geifman et al. 2019)."""
    _, risk = risk_coverage(conf, loss)
    return float(risk.mean())


def aurc_fast(conf, loss) -> float:
    """Same value as `aurc` (tie blocks averaged), vectorised: for bootstrap loops over large test sets."""
    conf = np.asarray(conf, float)
    loss = np.asarray(loss, float)
    order = np.argsort(-conf, kind="stable")
    c = conf[order]
    l = loss[order]
    block = np.r_[0, np.cumsum(np.diff(c) != 0)]
    means = np.bincount(block, weights=l) / np.bincount(block)
    l = means[block]
    return float((np.cumsum(l) / np.arange(1, len(l) + 1)).mean())


def optimal_aurc(loss) -> float:
    """AURC of an oracle that answers all correct items first."""
    loss = np.sort(np.asarray(loss, float))
    k = np.arange(1, len(loss) + 1)
    return float((np.cumsum(loss) / k).mean())


def selective_risk_at(conf, loss, coverage: float) -> float:
    cov, risk = risk_coverage(conf, loss)
    i = int(np.ceil(coverage * len(cov))) - 1
    return float(risk[max(i, 0)])


def auroc(score, label) -> float:
    """P(score of a positive > score of a negative), ties count 1/2 (Mann-Whitney)."""
    score = np.asarray(score, float)
    label = np.asarray(label).astype(bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.r_[pos, neg]
    ranks = _average_ranks(allv)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def ece(conf, correct, n_bins: int = 15) -> float:
    """Expected calibration error of `conf` against the 0/1 outcome `correct` (= 1 - e)."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(tot)


def paired_bootstrap(metric, a: dict, b: dict, n_boot: int = 1000, seed: int = 0) -> dict:
    """CI for metric(a) - metric(b) with items resampled jointly (same indices for both arms).

    a, b: dicts of equal-length arrays, e.g. {"conf": ..., "loss": ...}; metric(**d) -> float.
    """
    n = len(next(iter(a.values())))
    assert all(len(v) == n for v in list(a.values()) + list(b.values())), "arms must be item-aligned"
    rng = np.random.default_rng(seed)
    point = metric(**a) - metric(**b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = metric(**{k: np.asarray(v)[idx] for k, v in a.items()}) - \
            metric(**{k: np.asarray(v)[idx] for k, v in b.items()})
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(point), "lo": float(lo), "hi": float(hi), "n": n, "n_boot": n_boot}


# ---------------------------------------------------------------- gate statistics (rationale register v0.2)
def bootstrap_diffs(metric, a: dict, b: dict, n_boot: int = 1000, seed: int = 0) -> tuple[float, np.ndarray]:
    """Point difference metric(a) - metric(b) and its paired-bootstrap replicates (items resampled jointly)."""
    n = len(next(iter(a.values())))
    assert all(len(v) == n for v in list(a.values()) + list(b.values())), "arms must be item-aligned"
    rng = np.random.default_rng(seed)
    point = metric(**a) - metric(**b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = metric(**{k: np.asarray(v)[idx] for k, v in a.items()}) - \
            metric(**{k: np.asarray(v)[idx] for k, v in b.items()})
    return float(point), diffs


def bootstrap_p(diffs: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: difference = 0 (percentile method)."""
    return float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))


def equivalence_tost(metric, a: dict, b: dict, delta: float, alpha: float = 0.05,
                     n_boot: int = 1000, seed: int = 0) -> dict:
    """Two one-sided tests (Schuirmann 1987; Lakens et al. 2018) via the paired bootstrap.

    Equivalence at level alpha is declared iff the (1 - 2 alpha) percentile CI of metric(a) - metric(b)
    lies inside (-delta, +delta). delta must be fixed before results are seen.
    """
    point, diffs = bootstrap_diffs(metric, a, b, n_boot, seed)
    lo, hi = np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])
    return {"diff": point, "ci_lo": float(lo), "ci_hi": float(hi), "delta": delta,
            "ci_level": 1 - 2 * alpha, "equivalent": bool(-delta < lo and hi < delta)}


def holm(pvals: dict, alpha: float = 0.05) -> dict:
    """Holm (1979) step-down correction. Returns adjusted p-values and rejections, keyed like the input."""
    keys = sorted(pvals, key=pvals.get)
    m = len(keys)
    adj, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return {k: {"p": pvals[k], "p_holm": adj[k], "reject": adj[k] < alpha} for k in pvals}
