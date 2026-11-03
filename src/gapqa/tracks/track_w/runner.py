"""Per-item scoring of one checkpoint on one test set, resumable (protocol v1.1, Sections 7 and 10.1).

Each item yields one JSON line: decision score s, the forced-prefix greedy answer (when decoding is on),
EM/F1 over all gold answers, and the fields the analysis needs (source, slice, pair, cluster, K, g).
Items already present in the output file are skipped, so a pause never changes the results.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ...score.answers import exact_match, extract_answer, f1
from .model import to_batch
from .sequence import build_sequence_v1, decision_token_ids_v1


def read_records(path: str | Path) -> list[dict]:
    """All complete records; a line cut by a power loss is ignored (and its item is rescored)."""
    out = []
    with open(path, encoding="utf8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            try:
                out.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass                                    # a line cut by a power loss is simply redone
    return out


@torch.no_grad()
def score_items(model, items, arm: str, tok, chat, gap_id: int, out_path: str | Path, device: str = "cuda",
                decode: bool = True, max_new: int = 24, autocast_dtype=torch.bfloat16, log=print) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skip = done_ids(out_path)
    a_id, b_id = decision_token_ids_v1(tok, chat)
    eot_ids = tok.encode(chat.eot)
    stop = {eot_ids[0]} | set(tok.encode("\n")[-1:])
    model.eval()
    n = 0
    ctx = torch.autocast("cuda", dtype=autocast_dtype) if device.startswith("cuda") else torch.autocast("cpu", enabled=False)
    if out_path.exists() and out_path.stat().st_size:
        with open(out_path, "rb") as fb:
            fb.seek(-1, 2)
            if fb.read(1) != b"\n":                 # a cut last line: close it so new records start clean
                with open(out_path, "a", encoding="utf8") as fa:
                    fa.write("\n")
    with open(out_path, "a", encoding="utf8") as f, ctx:
        for it in items:
            if it.id in skip:
                continue
            s_seq = build_sequence_v1(it, arm, tok, chat, gap_id, with_target=False)
            s = model.decision_score(to_batch(s_seq, device=device, with_labels=False), a_id, b_id)
            rec = {"id": it.id, "source": it.source, "split": it.split, "slice": it.slice, "lang": it.lang,
                   "y_rec": it.y_rec, "s": s, "pair_id": it.pair_id, "cluster": it.cluster,
                   "K": len(it.gaps), "g_ref": it.g_ref, "arm": arm}
            if decode and it.y_rec == 1:
                d_seq = build_sequence_v1(it, arm, tok, chat, gap_id, forced_answer_prefix=True)
                gen = model.greedy_decode(to_batch(d_seq, device=device, with_labels=False), max_new, stop, {gap_id})
                pred = extract_answer(tok.decode(gen), chat.eot)
                rec.update(pred=pred, em=exact_match(pred, it.answers), f1=f1(pred, it.answers),
                           n_gen=len(gen), truncated=len(gen) >= max_new)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
    log(f"scored {n} new items -> {out_path} ({len(skip)} already done)")
    return n
