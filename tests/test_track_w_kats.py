"""KAT-2 (embedding path) and KAT-3/3d (positions) of the Track W wrapper on GapQA v1 sequences.

Tiny randomly initialised Qwen2 model and the deterministic test tokenizer: runs on CPU without downloads.
KAT-3d guards against transformers' packed-sequence detection, which reads every jump in position_ids as the start
of a new sequence when no attention mask is given (paper, Section 4.5).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gapqa.tokenization import WhitespaceTokenizer  # noqa: E402
from gapqa.tracks.track_w import ChatFormat, GLAConfig, GLAModel, build_sequence_v1, decision_token_ids_v1, to_batch  # noqa: E402,E501

from .helpers import GAP_ID, item, qwen_like  # noqa: E402
from .test_track_w_runner import VOCAB, tiny_base  # noqa: E402


def seq(arm, y=1, **kw):
    return build_sequence_v1(item(y), arm, WhitespaceTokenizer(), ChatFormat.from_template(qwen_like), GAP_ID, **kw)


def fwd(m, b):
    return m(b["input_ids"], b["position_ids"], b["is_gap"], b["gap_g"]).logits


def test_kat2_gap_path_equals_plain_ids_when_gap_vector_is_the_placeholder_row():
    base = tiny_base()
    m = GLAModel(base, GAP_ID, GLAConfig(arm="G")).eval()
    with torch.no_grad():
        m.gap.e_gap.copy_(base.get_input_embeddings().weight[GAP_ID])      # W_g is zero at init
        for y in (0, 1):
            b = to_batch(seq("G", y))
            ref = base(input_ids=b["input_ids"], position_ids=b["position_ids"],
                       attention_mask=torch.ones_like(b["input_ids"])).logits
            assert torch.equal(fwd(m, b), ref)


@pytest.mark.parametrize("arm", ["B0", "B1", "B2", "B1P"])
def test_kat2_baseline_renderings_do_not_touch_embeddings(arm):
    base = tiny_base()
    m = GLAModel(base, GAP_ID, GLAConfig(arm=arm)).eval()
    assert m.gap is None and m.trainable_extra() == []
    b = to_batch(seq(arm))
    with torch.no_grad():
        ref = base(input_ids=b["input_ids"], position_ids=b["position_ids"],
                   attention_mask=torch.ones_like(b["input_ids"])).logits
        assert torch.equal(fwd(m, b), ref)


def test_kat2_G_equals_G0_at_init():
    base = tiny_base()
    g = GLAModel(base, GAP_ID, GLAConfig(arm="G")).eval()
    g0 = GLAModel(base, GAP_ID, GLAConfig(arm="G0")).eval()
    with torch.no_grad():
        g0.gap.e_gap.copy_(g.gap.e_gap)
        b = to_batch(seq("G"))
        assert torch.equal(fwd(g, b), fwd(g0, b))


def test_kat3_model_honours_position_ids():
    """B1 and B1P feed identical ids and differ only in positions; a constant shift of all positions changes nothing."""
    m = GLAModel(tiny_base(), GAP_ID, GLAConfig(arm="B1")).eval()
    s1, s1p = seq("B1"), seq("B1P")
    assert s1.input_ids == s1p.input_ids and s1.position_ids != s1p.position_ids
    b1, b1p = to_batch(s1), to_batch(s1p)
    with torch.no_grad():
        o1, o1p = fwd(m, b1), fwd(m, b1p)
        shifted = m(b1p["input_ids"], b1p["position_ids"] + 777, b1p["is_gap"], b1p["gap_g"]).logits
    assert (o1 - o1p).abs().max() > 1e-4
    assert (o1p - shifted).abs().max() < 1e-4


@pytest.mark.parametrize("arm", ["B2", "B1P", "G0", "G"])
def test_kat3d_tokens_after_a_gap_still_attend_to_tokens_before_it(arm):
    m = GLAModel(tiny_base(), GAP_ID, GLAConfig(arm=arm)).eval()
    s = seq(arm)
    b = to_batch(s)
    jumps = [i for i in range(1, len(s.position_ids)) if s.position_ids[i] - s.position_ids[i - 1] != 1]
    assert jumps, "a true-position rendering must contain at least one jump"
    before = jumps[0] - 2
    ids2 = b["input_ids"].clone()
    ids2[0, before] = (ids2[0, before] + 1) % (VOCAB - 1)
    with torch.no_grad():
        o = fwd(m, b)
        o2 = m(ids2, b["position_ids"], b["is_gap"], b["gap_g"]).logits
    assert (o2 - o)[0, -1].abs().max() > 1e-6


def test_decision_score_is_a_probability():
    m = GLAModel(tiny_base(), GAP_ID, GLAConfig(arm="G")).eval()
    a, ab = decision_token_ids_v1(WhitespaceTokenizer(), ChatFormat.from_template(qwen_like))
    p = m.decision_score(to_batch(seq("G", with_target=False), with_labels=False), a, ab)
    assert 0.0 < p < 1.0
