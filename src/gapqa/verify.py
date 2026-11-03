"""Integrity checks against MANIFEST.json (release specification, Section 7).

* Frozen files (Zenodo) are checked byte for byte: SHA-256 of the decompressed file.
* Any copy (Hugging Face, or a user's own export) is checked record by record: the release-only fields are removed,
  `meta` is decoded if it is a JSON string (as in the Hugging Face copy), and the record's canonical serialisation is
  hashed and compared with the manifest.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import lzma
from pathlib import Path

from .data import iter_records
from .item import RELEASE_FIELDS


def canonical(record: dict) -> bytes:
    d = {k: v for k, v in record.items() if k not in RELEASE_FIELDS}
    if isinstance(d.get("meta"), str):            # Hugging Face copy stores meta as a JSON string
        d["meta"] = json.loads(d["meta"])
    return json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf8")


def record_sha256(record: dict) -> str:
    return hashlib.sha256(canonical(record)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """SHA-256 of the file's content after decompression (.gz / .xz)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else lzma.open if path.suffix == ".xz" else open
    h = hashlib.sha256()
    with opener(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen(path: str | Path, manifest: dict) -> None:
    """`path` is train/dev/test.jsonl (optionally .xz/.gz); raises on mismatch."""
    name = Path(path).name.replace(".xz", "").replace(".gz", "")
    want = manifest["frozen_files"][name]["sha256"]
    got = file_sha256(path)
    if got != want:
        raise ValueError(f"{name}: SHA-256 {got} != manifest {want}")


def verify_records(paths, manifest: dict, require_all: str | None = None, **kw) -> int:
    """Every record of `paths` must appear in the manifest with the same hash. With `require_all` set to a split
    name, every manifest record of that split must also be present. Returns the number of records checked."""
    want = manifest["records"]
    seen = set()
    for p in ([paths] if isinstance(paths, (str, Path)) else paths):
        for r in iter_records(p, **kw):
            rid = r["id"]
            if rid not in want:
                raise ValueError(f"{rid}: not in the manifest")
            if record_sha256(r) != want[rid]["sha256"]:
                raise ValueError(f"{rid}: content differs from the frozen record")
            if rid in seen:
                raise ValueError(f"{rid}: duplicated")
            seen.add(rid)
    if require_all is not None:
        missing = [k for k, v in want.items() if v["split"] == require_all and k not in seen]
        if missing:
            raise ValueError(f"{len(missing)} {require_all} records missing, e.g. {missing[:3]}")
    return len(seen)
