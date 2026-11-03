"""Known-answer tests of the Track W renderer and sequence assembly across chat templates (no downloads)."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from gapqa import marker_text, render_v1  # noqa: E402
from gapqa.prompt_text import SYSTEM as SHARED_SYSTEM  # noqa: E402
from gapqa.tokenization import WhitespaceTokenizer  # noqa: E402
from gapqa.tracks.track_w.chat import SYSTEM, ChatFormat  # noqa: E402
from gapqa.tracks.track_w.gap_module import U_MAX, GapEmbedding, fourier_features  # noqa: E402
from gapqa.tracks.track_w.sequence import build_sequence_v1, decision_token_ids_v1  # noqa: E402

from .helpers import GAP_ID, gemma_like, item, qwen_like  # noqa: E402,F401


@pytest.mark.parametrize("template", [qwen_like, gemma_like])
def test_chat_format_split_and_eot(template):
    cf = ChatFormat.from_template(template)
    assert SYSTEM in cf.pre and cf.pre.endswith("Context:\n")
    assert cf.mid == "\n\nQuestion: "
    assert cf.eot in ("<|im_end|>", "<end_of_turn>")
    assert cf.target(0, "") == "ABSTAIN" + cf.eot


def test_y_rec_recomputed_from_spans():
    assert item(1).recompute_y_rec() == 1 and item(0).recompute_y_rec() == 0


@pytest.mark.parametrize("arm", ["B0", "B1", "B2", "B1P", "G0", "G"])
def test_render_keeps_observed_text_and_true_positions(arm):
    tok = WhitespaceTokenizer()
    it = item(1)
    r = render_v1(it, arm, tok, GAP_ID)
    obs = [i for i, o in zip(r.ids, r.is_observed) if o]
    expected = sum((tok.encode(t) for k, t, _ in it.observed_and_deleted() if k == "obs"), [])
    assert obs == expected                                     # identical observed tokens in every arm
    assert all(b > a for a, b in zip(r.pos, r.pos[1:]))
    if arm in ("B2", "B1P", "G0", "G"):
        total = sum(len(tok.encode(t)) for _, t, _ in it.observed_and_deleted())
        assert r.next_pos == total                             # true positions span the whole source
    if arm in ("G0", "G"):
        assert sum(r.is_gap) == 2 and [g for g, x in zip(r.gap_g, r.is_gap) if x] == r.g_model
    if arm == "B1":
        assert tok.decode([i for i, m in zip(r.ids, r.is_marker) if m]).count("omitted") == 2


def test_marker_longer_than_gap_is_refused():
    tok = WhitespaceTokenizer()
    it = item(1)
    a = it.text.index("w10 ")
    it.gaps = [(a, a + 4)]                                      # one tiny gap: "w10 " -> 1 token < marker
    with pytest.raises(ValueError):
        render_v1(it, "B1P", tok, GAP_ID)
    assert len(tok.encode(marker_text(1))) > 1


@pytest.mark.parametrize("template", [qwen_like, gemma_like])
def test_sequence_v1_structure_and_decision_tokens(template):
    tok = WhitespaceTokenizer()
    cf = ChatFormat.from_template(template)
    for y in (0, 1):
        s = build_sequence_v1(item(y), "G", tok, cf, GAP_ID)
        assert all(b > a for a, b in zip(s.position_ids, s.position_ids[1:]))
        assert tok.decode(s.labels[s.n_prompt:]).replace(" ", "").startswith("ANSWER" if y else "ABSTAIN")
        f = build_sequence_v1(item(y), "G", tok, cf, GAP_ID, forced_answer_prefix=True)
        assert f.n_target == 0 and tok.decode(f.input_ids[-3:]).replace(" ", "").endswith("ANSWER:")
    a, b = decision_token_ids_v1(tok, cf)
    assert a != b


def test_fourier_features_have_linear_term_and_17_dims():
    u = torch.tensor([math.log1p(16.0), math.log1p(2048.0), math.log1p(4096.0)])
    f = fourier_features(u, 8)
    assert f.shape == (3, 17)
    assert torch.allclose(f[:, 0], u / U_MAX) and f[1, 0] == pytest.approx(1.0) and f[2, 0] > 1.0
    assert GapEmbedding(32, 8).w_g.in_features == 17


def test_gap_init_noise_depends_on_seed():
    w = torch.randn(100, 16)
    a = GapEmbedding.init_from_embeddings(w, seed=0)
    b = GapEmbedding.init_from_embeddings(w, seed=1)
    assert not torch.allclose(a, b) and torch.allclose(a, GapEmbedding.init_from_embeddings(w, seed=0))


def test_track_t_and_track_w_share_the_system_prompt():
    assert SYSTEM == SHARED_SYSTEM
