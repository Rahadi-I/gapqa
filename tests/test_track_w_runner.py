"""Track W runner: answer scoring, cached greedy decoding and resumable per-item scoring (tiny model, CPU)."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from gapqa.score.answers import exact_match, extract_answer, f1, normalize  # noqa: E402
from gapqa.tokenization import WhitespaceTokenizer  # noqa: E402
from gapqa.tracks.track_w import ChatFormat, GLAConfig, GLAModel, build_sequence_v1, score_items, to_batch  # noqa: E402

from .helpers import GAP_ID, item, qwen_like  # noqa: E402

VOCAB = 4000


def tiny_base(seed=0):
    """Tiny randomly initialised Qwen2 (the architecture class of Qwen2.5), eager attention, CPU."""
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32768,
                      rope_theta=1e6, tie_word_embeddings=True)
    cfg._attn_implementation = "eager"
    return Qwen2ForCausalLM(cfg).eval()


def test_squad_normalisation_and_aliases():
    assert normalize("The  Eiffel Tower!") == "eiffel tower"
    assert exact_match("eiffel tower", ["Tour Eiffel", "The Eiffel Tower"]) == 1
    assert f1("the big red dog", ["red dog"]) == pytest.approx(0.8)
    assert extract_answer(" Paris<|im_end|>junk", "<|im_end|>") == "Paris"
    assert extract_answer(" Paris\nmore", "<|im_end|>") == "Paris"


def test_cached_greedy_equals_uncached_greedy():
    tok = WhitespaceTokenizer()
    cf = ChatFormat.from_template(qwen_like)
    m = GLAModel(tiny_base(), GAP_ID, GLAConfig(arm="G")).eval()
    b = to_batch(build_sequence_v1(item(1), "G", tok, cf, GAP_ID, forced_answer_prefix=True), with_labels=False)
    got = m.greedy_decode(b, 6, stop_ids=set(), banned_ids={GAP_ID})
    ids, pos, gap, gg = b["input_ids"], b["position_ids"], b["is_gap"], b["gap_g"]
    ref = []
    with torch.no_grad():
        for _ in range(6):
            lg = m(ids, pos, gap, gg, logits_to_keep=1).logits[0, -1].float()
            lg[GAP_ID] = float("-inf")
            t = int(lg.argmax())
            ref.append(t)
            ids = torch.cat([ids, torch.tensor([[t]])], 1)
            pos = torch.cat([pos, pos[:, -1:] + 1], 1)
            gap = torch.cat([gap, torch.tensor([[False]])], 1)
            gg = torch.cat([gg, torch.tensor([[0]])], 1)
    assert got == ref and GAP_ID not in got


def test_scoring_is_resumable(tmp_path):
    tok = WhitespaceTokenizer()
    cf = ChatFormat.from_template(qwen_like)
    m = GLAModel(tiny_base(), GAP_ID, GLAConfig(arm="G")).eval()
    items = [item(1), item(0)]
    items[1].id = "t0b"
    out = tmp_path / "scores.jsonl"
    n1 = score_items(m, items[:1], "G", tok, cf, GAP_ID, out, device="cpu", max_new=4, log=lambda *_: None)
    with open(out, "a") as f:
        f.write('{"id": "trunc')                      # a line cut by a power loss
    n2 = score_items(m, items, "G", tok, cf, GAP_ID, out, device="cpu", max_new=4, log=lambda *_: None)
    assert (n1, n2) == (1, 1)
    from gapqa.tracks.track_w.runner import read_records
    recs = read_records(out)
    assert [r["id"] for r in recs] == ["t1", "t0b"]
    assert 0 < recs[0]["s"] < 1 and "em" in recs[0] and "em" not in recs[1]
