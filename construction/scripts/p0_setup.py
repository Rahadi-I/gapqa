"""Phase P0 of the study protocol v1.1: download and pin models and datasets, check the environment.

    python scripts/p0_setup.py check                 # environment, GPU, disk
    python scripts/p0_setup.py models                # 7 models, revisions pinned  (needs `huggingface-cli login`
                                                     #  and approved access to Llama 3.2 and Gemma 3)
    python scripts/p0_setup.py datasets              # datasets with direct links; prints manual steps for the rest
    python scripts/p0_setup.py verify                # every expected file present; writes the P0 manifest

Re-running any step skips what is already present. Everything is recorded in manifests/p0_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MAN = ROOT / "manifests" / "p0_manifest.json"

MODELS = {  # protocol Section 4
    "Qwen/Qwen2.5-1.5B-Instruct": "stage1",
    "Qwen/Qwen2.5-0.5B-Instruct": "stage2",
    "Qwen/Qwen2.5-3B-Instruct": "stage2",
    "meta-llama/Llama-3.2-1B-Instruct": "stage2 (gated)",
    "meta-llama/Llama-3.2-3B-Instruct": "stage2 (gated)",
    "google/gemma-3-1b-it": "stage2 (gated)",
    "ibm-granite/granite-3.1-2b-instruct": "conditional; closed-book filter",
}

URLS = {  # direct, official links
    "locomo/locomo10.json": "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
    "2wiki/data_ids_april7.zip": "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1",
    "qasper/qasper-test-and-evaluator-v0.3.tgz":
        "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
    # held-out top-up pool (never trained on), deviation D2
    "qasper/qasper-train-dev-v0.3.tgz":
        "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
}

HF_DATASETS = {  # repo id -> (local dir, allow_patterns or None)
    "rajpurkar/squad": ("squad", None),
    "hotpotqa/hotpot_qa": ("hotpotqa", None),
    "allenai/qasper": ("qasper", None),
    # TyDi QA GoldP = HF config "secondary_task"; the original GCS bucket became non-public (deviation D1)
    "google-research-datasets/tydiqa": ("tydiqa", ["secondary_task/*"]),
}

MANUAL = """
Manual downloads (official links that need a browser or a Google Drive script):
  MuSiQue  : https://github.com/StonyBrookNLP/musique  -> run `bash download_data.sh` (or the Google Drive link in the
             README). Copy musique_ans_v1.0_train.jsonl and musique_ans_v1.0_dev.jsonl to data/raw/musique/
  QASPER   : if data/raw/qasper has no JSON files, take the test file from https://huggingface.co/datasets/allenai/qasper
  If a direct link above failed, fetch the same file from the source's README and put it at the listed path.
Then run: python scripts/p0_setup.py verify
"""

EXPECTED = [
    "squad", "hotpotqa", "qasper", "nq",
    "tydiqa/secondary_task",
    "locomo/locomo10.json", "2wiki/data_ids_april7.zip", "qasper/qasper-train-dev-v0.3.tgz",
    "musique/musique_ans_v1.0_train.jsonl", "musique/musique_ans_v1.0_dev.jsonl",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    return json.loads(MAN.read_text(encoding="utf8")) if MAN.exists() else {"models": {}, "datasets": {}, "env": {}}


def save_manifest(m: dict) -> None:
    MAN.parent.mkdir(parents=True, exist_ok=True)
    m["updated"] = datetime.now(timezone.utc).isoformat()
    MAN.write_text(json.dumps(m, indent=2), encoding="utf8")


def cmd_check(_):
    import torch
    import transformers
    m = load_manifest()
    free = shutil.disk_usage(ROOT).free / 2**30
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
           "transformers": transformers.__version__, "cuda": torch.cuda.is_available(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "free_disk_gib": round(free, 1)}
    try:
        import peft
        env["peft"] = peft.__version__
    except ImportError:
        env["peft"] = "MISSING: python -m pip install peft"
    m["env"] = env
    save_manifest(m)
    print(json.dumps(env, indent=2))
    if free < 80:
        print("WARNING: less than 80 GiB free; models ~26 GiB, datasets and NQ shards ~10 GiB, runs ~20 GiB.")


def cmd_models(_):
    from huggingface_hub import HfApi, snapshot_download
    api, m = HfApi(), load_manifest()
    for repo, role in MODELS.items():
        try:
            sha = api.model_info(repo).sha
            path = snapshot_download(repo, revision=sha, allow_patterns=["*.json", "*.safetensors", "*.model",
                                                                         "*.txt", "*.jinja", "*.tiktoken"])
            m["models"][repo] = {"revision": sha, "path": path, "role": role}
            print(f"[ok] {repo} @ {sha[:10]}")
        except Exception as e:  # gated repo without access, network, ...
            m["models"][repo] = {"error": str(e)[:300], "role": role}
            print(f"[FAIL] {repo}: {str(e)[:200]}")
        save_manifest(m)


def cmd_datasets(_):
    from huggingface_hub import HfApi, snapshot_download
    api, m = HfApi(), load_manifest()
    RAW.mkdir(parents=True, exist_ok=True)
    for repo, (local, patterns) in HF_DATASETS.items():
        try:
            sha = api.dataset_info(repo).sha
            snapshot_download(repo, repo_type="dataset", revision=sha, local_dir=RAW / local,
                              allow_patterns=patterns)
            m["datasets"][repo] = {"revision": sha, "path": str(RAW / local)}
            print(f"[ok] {repo} @ {sha[:10]}")
        except Exception as e:
            m["datasets"][repo] = {"error": str(e)[:300]}
            print(f"[FAIL] {repo}: {str(e)[:200]}")
        save_manifest(m)
    # Natural Questions: official release on the Hub; first 10 train shards + all validation shards
    repo = "google-research-datasets/natural_questions"
    try:
        info = api.dataset_info(repo)
        files = sorted(f.rfilename for f in info.siblings)
        train = [f for f in files if "train" in f and f.endswith(".parquet")][:10]
        val = [f for f in files if "validation" in f and f.endswith(".parquet")]
        snapshot_download(repo, repo_type="dataset", revision=info.sha, local_dir=RAW / "nq",
                          allow_patterns=train + val)
        m["datasets"][repo] = {"revision": info.sha, "train_shards": train, "validation_shards": val}
        print(f"[ok] {repo}: {len(train)} train + {len(val)} validation shards")
    except Exception as e:
        m["datasets"][repo] = {"error": str(e)[:300]}
        print(f"[FAIL] {repo}: {str(e)[:200]}")
    save_manifest(m)
    for rel, url in URLS.items():
        dst = RAW / rel
        if dst.exists():
            print(f"[skip] {rel}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, dst)
            m["datasets"][rel] = {"url": url, "sha256": sha256(dst)}
            print(f"[ok] {rel}")
        except Exception as e:
            m["datasets"][rel] = {"url": url, "error": str(e)[:300]}
            print(f"[FAIL] {rel}: {e}")
        save_manifest(m)
    print(MANUAL)


def cmd_verify(_):
    m, missing = load_manifest(), []
    for rel in EXPECTED:
        p = RAW / rel
        if not p.exists() or (p.is_dir() and not any(p.rglob("*"))):
            missing.append(rel)
        elif p.is_file() and rel not in m["datasets"]:
            m["datasets"][rel] = {"sha256": sha256(p), "source": "manual"}
    bad_models = [r for r, v in m["models"].items() if "error" in v] + [r for r in MODELS if r not in m["models"]]
    save_manifest(m)
    print("missing datasets:", missing or "none")
    print("models not ready:", bad_models or "none")
    print("P0 exit criterion met" if not missing and not bad_models else "P0 NOT complete")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("check", "models", "datasets", "verify"):
        sub.add_parser(n)
    a = ap.parse_args()
    {"check": cmd_check, "models": cmd_models, "datasets": cmd_datasets, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
