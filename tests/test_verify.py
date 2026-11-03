"""gapqa verify: frozen files byte for byte, copies record by record (release-only fields ignored)."""
from __future__ import annotations

import gzip
import hashlib
import json
import lzma

import pytest

from gapqa.verify import record_sha256, verify_frozen, verify_records

from .helpers import item


def _write(tmp_path):
    recs = [item(0).to_json(), item(1).to_json()]
    raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode()
    (tmp_path / "test.jsonl.xz").write_bytes(lzma.compress(raw))
    manifest = {"frozen_files": {"test.jsonl": {"sha256": hashlib.sha256(raw).hexdigest()}},
                "records": {r["id"]: {"split": "test", "sha256": record_sha256(r)} for r in recs}}
    return recs, manifest


def test_frozen_and_records(tmp_path):
    recs, manifest = _write(tmp_path)
    verify_frozen(tmp_path / "test.jsonl.xz", manifest)
    copy = [dict(r, meta=json.dumps(r["meta"]), source_license="CC-BY-4.0", gapqa_version="1.0.0", canary="X")
            for r in recs]                       # as in the Hugging Face copy: meta is a JSON string
    with gzip.open(tmp_path / "copy.jsonl.gz", "wt", encoding="utf8") as f:
        for r in copy:
            f.write(json.dumps(r) + "\n")
    assert verify_records(tmp_path / "copy.jsonl.gz", manifest, require_all="test") == 2


def test_altered_record_fails(tmp_path):
    recs, manifest = _write(tmp_path)
    bad = dict(recs[0], y_rec=1 - recs[0]["y_rec"])
    (tmp_path / "bad.jsonl").write_text(json.dumps(bad) + "\n" + json.dumps(recs[1]) + "\n")
    with pytest.raises(ValueError):
        verify_records(tmp_path / "bad.jsonl", manifest)
    (tmp_path / "part.jsonl").write_text(json.dumps(recs[1]) + "\n")
    with pytest.raises(ValueError):
        verify_records(tmp_path / "part.jsonl", manifest, require_all="test")


def test_item_reads_the_hugging_face_form():
    from gapqa.item import ItemV1
    r = dict(item(1).to_json(), meta=json.dumps({"K": 2}), source_license="x", gapqa_version="1.0.0", canary="")
    assert ItemV1.from_json(r).meta == {"K": 2}
