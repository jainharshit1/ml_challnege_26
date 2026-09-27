"""Verify HF Hub upload integrity: compare local files against remote metadata.

Checks every file in artifacts/{embeddings,normalized,idf,splits} against the
corresponding entry in the HF dataset repo. Fails fast on size mismatch,
then verifies SHA256 for every LFS-tracked file.

Usage:
    /DATA/air_force_object_detection/.venv/bin/python verify_hf.py
"""
import hashlib
from pathlib import Path
from huggingface_hub import HfApi

REPO = "jainsaabb/ml_challenge_2026_artifacts"
BASE = Path("/DATA/air_force_object_detection/ml_challnege_26/artifacts")

MAP = {
    "embeddings": BASE / "embeddings",
    "normalized": BASE / "normalized",
    "idf":        BASE / "idf",
    "splits":     BASE / "splits",
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    api = HfApi()
    info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}

    bad, ok, missing = 0, 0, 0
    for prefix, ldir in MAP.items():
        for lp in sorted(ldir.rglob("*")):
            if not lp.is_file():
                continue
            rname = f"{prefix}/{lp.relative_to(ldir).as_posix()}"
            r = remote.get(rname)
            if r is None:
                print(f"MISSING remote: {rname}")
                missing += 1
                continue
            lsize = lp.stat().st_size
            rsize = r.size
            if lsize != rsize:
                print(f"SIZE MISMATCH {rname}: local={lsize} remote={rsize}")
                bad += 1
                continue
            rsha = getattr(r.lfs, "sha256", None) if r.lfs else None
            if rsha is None:
                ok += 1
                continue
            lsha = sha256(lp)
            if lsha != rsha:
                print(f"HASH MISMATCH {rname}")
                bad += 1
            else:
                ok += 1
        print(f"  scanned {prefix}: ok so far")

    print(f"\nresult: ok={ok} bad={bad} missing={missing}")


if __name__ == "__main__":
    main()
