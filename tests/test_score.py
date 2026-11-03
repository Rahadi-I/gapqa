"""Scorer. Unit checks on synthetic predictions, and KAT-R1: the released reference results reproduce the paper's
descriptive tables exactly (set GAPQA_TEST_ITEMS and GAPQA_REFERENCE_DIR; expected values in
tests/data/s1_benchmark_tables.json)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from gapqa.item import ItemV1
from gapqa.score.metrics import aurc, aurc_fast, auroc, optimal_aurc
from gapqa.score.predictions import read_predictions
from gapqa.score.report import join, summarise

HERE = Path(__file__).parent


def test_metric_identities():
    rng = np.random.default_rng(0)
    s = rng.random(200).round(2)                       # ties on purpose
    loss = (rng.random(200) < 0.3).astype(float)
    assert aurc(s, loss) == pytest.approx(aurc_fast(s, loss))
    assert optimal_aurc(loss) <= aurc_fast(s, loss)
    y = (rng.random(200) < 0.5).astype(int)
    assert auroc(y + 0.0, y) == 1.0 and auroc(-y + 0.0, y) == 0.0 and auroc(np.zeros(200), y) == 0.5


def test_prediction_file_checks(tmp_path):
    p = tmp_path / "p.jsonl"
    hdr = {"track": "T", "rendering": "B1", "s_source": "logprob", "pred_source": "reply", "model": "m", "seed": 0}
    p.write_text(json.dumps({"id": "a", "s": 0.3, "pred": None, **hdr}) + "\n"
                 + json.dumps({"id": "b", "s": 7.0, "pred": "x", **hdr}) + "\n")
    preds, h = read_predictions(p)
    assert preds["b"] == {"s": 7.0, "pred": "x"} and h["track"] == "T" and len(h["sha256"]) == 64
    p.write_text(json.dumps({"id": "a", "s": 0.3, **hdr}) + "\n" + json.dumps({"id": "a", "s": 0.4, **hdr}) + "\n")
    with pytest.raises(ValueError):
        read_predictions(p)
    p.write_text(json.dumps({"id": "a", "s": float("nan"), **hdr}) + "\n")
    with pytest.raises(ValueError):
        read_predictions(p)


def _items():
    out = []
    for k, src in enumerate(["squad", "qasper"]):
        for i in range(10):
            for y in (0, 1):
                out.append(ItemV1(id=f"{src}-{i}-y{y}", source=src, split="test", slice="in_range", lang="en",
                                  question="q", answers=["Paris"], y_rec=y, text="t", gaps=[(0, 1)],
                                  evidence=[(0, 1)] if y == 0 else [(2, 3)], g_ref=[16], pair_id=f"{src}-{i}",
                                  cluster=f"paper{i // 3}" if src == "qasper" else f"{src}-{i}"))
    return out


def test_join_recomputes_em_and_rejects_missing_ids():
    items = _items()
    preds = {it.id: {"s": float(it.y_rec), "pred": "the Paris" if it.y_rec else None} for it in items}
    recs = join(items, preds)
    assert all(r["em"] == 1 for r in recs if r["y_rec"] == 1)
    rep = summarise([recs], with_em=True, ci=True, n_boot=50)
    assert rep["in_domain_in_range"]["auroc"] == 1.0 and rep["held_out"]["auroc_ci95"] == [1.0, 1.0]
    assert rep["in_domain_in_range"]["aurc_em"] == pytest.approx(rep["in_domain_in_range"]["aurc_em_oracle"])
    del preds[items[0].id]
    with pytest.raises(ValueError):
        join(items, preds)


@pytest.mark.skipif(not (os.environ.get("GAPQA_TEST_ITEMS") and os.environ.get("GAPQA_REFERENCE_DIR")),
                    reason="KAT-R1 needs GAPQA_TEST_ITEMS and GAPQA_REFERENCE_DIR")
def test_kat_r1_reference_results_reproduce_the_paper():
    from gapqa import load_items
    from gapqa.score.report import score_files
    ref = json.loads((HERE / "data" / "s1_benchmark_tables.json").read_text(encoding="utf8"))
    runs = json.loads((HERE / "data" / "reference_runs.json").read_text(encoding="utf8"))
    items = load_items(os.environ["GAPQA_TEST_ITEMS"], split="test")
    rdir = Path(os.environ["GAPQA_REFERENCE_DIR"])
    n = 0
    for name, files in runs.items():
        rep = score_files(items, [rdir / f for f in files], with_em=name != "BM25",
                          ci=name == "BM25" or name.startswith("ZS_B1") or name.startswith("FT_B1:"))

        def cmp(a, b, path):
            nonlocal n
            if isinstance(b, dict):
                for k, v in b.items():
                    if k != "n_seeds":
                        cmp(a[k], v, f"{path}/{k}")
            elif isinstance(b, list):
                for x, y in zip(a, b):
                    cmp(x, y, path)
            elif b is None:
                assert a is None, path
            else:
                n += 1
                assert a == pytest.approx(b, abs=1e-12), (path, a, b)
        cmp(rep, ref["runs"][name], name)
    assert n > 2000
