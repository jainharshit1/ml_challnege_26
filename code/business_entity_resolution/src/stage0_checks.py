"""Stage 0.5 checks (plan §1.5).

These are the *remaining* Stage 0 checks the plan calls for. The exploratory
questions Q1–Q7 are already answered in the top-level stage_0.py and its
output; this module produces stage0_report.txt with the fields Stages 1–4
depend on.

Outputs
-------
artifacts/reports/stage0_report.txt   Human-readable report.
artifacts/reports/stage0_report.json  Machine-readable copy.

Checks
------
1. Postal-code coverage per source × country (5- or 6-digit standalone token).
2. House-number coverage (leading number OR labelled: H.NO, PLOT, HN, S.NO).
3. Script distribution per source × country × field (name & address).
4. Name-only duplication per country (chain-name problem size).
5. Source asymmetry (S1↔S2 vs S1↔S3 name Jaro similarity on true pairs).
6. True matches split by source (S2 vs S3) per S1.
"""
from __future__ import annotations
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .io_utils import (
    explode_ground_truth,
    read_ground_truth,
    read_source_tsv,
)

POSTAL_RE = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")
HOUSE_LABEL_RE = re.compile(
    r"\b(?:h\.?\s*no|hno|plot\s*no|shp\s*no|shop\s*no|kh\.?\s*no|s\.?\s*no|"
    r"door|flat)\b\.?\s*[:.-]?\s*[\dA-Za-z][\dA-Za-z/\-]*",
    re.IGNORECASE,
)
LEADING_NUMBER_RE = re.compile(r"^\s*[\dA-Za-z][\dA-Za-z/\-]*\s+[A-Za-z]")


# Unicode-block ranges we care about (start, end, label). Any codepoint outside
# these falls into "other".
BLOCKS = [
    (0x0000, 0x007F, "latin"),
    (0x0080, 0x024F, "latin"),      # latin-1, extended
    (0x0900, 0x097F, "devanagari"),
    (0x0B80, 0x0BFF, "tamil"),
    (0x0980, 0x09FF, "bengali"),
    (0x0C00, 0x0C7F, "telugu"),
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
    (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0600, 0x06FF, "arabic"),
]


def _script_of(text: str) -> str:
    if not text:
        return "empty"
    counts: Counter[str] = Counter()
    for ch in text:
        if not ch.strip() or unicodedata.category(ch).startswith(("P", "N", "Z")):
            continue
        cp = ord(ch)
        found = False
        for lo, hi, lbl in BLOCKS:
            if lo <= cp <= hi:
                counts[lbl] += 1
                found = True
                break
        if not found:
            counts["other"] += 1
    if not counts:
        return "empty"
    return counts.most_common(1)[0][0]


def _has_postal(addr) -> bool:
    if not isinstance(addr, str) or not addr:
        return False
    return bool(POSTAL_RE.search(addr))


def _has_house(addr) -> bool:
    if not isinstance(addr, str) or not addr:
        return False
    return bool(HOUSE_LABEL_RE.search(addr) or LEADING_NUMBER_RE.search(addr))


def _cov(df: pd.DataFrame, fn) -> dict[str, float]:
    out: dict[str, float] = {}
    for c, g in df.groupby("country", sort=False):
        out[str(c)] = float(g["business_address"].apply(fn).mean())
    return out


def _script_dist(df: pd.DataFrame, col: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for c, g in df.groupby("country", sort=False):
        vals = g[col].fillna("").apply(_script_of).value_counts(normalize=True)
        out[str(c)] = {k: float(v) for k, v in vals.items()}
    return out


def _name_dup(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """% of records that share a normalized-lower name with ≥1 other, per country."""
    out: dict[str, dict[str, float]] = {}
    for c, g in df.groupby("country", sort=False):
        key = g["business_name"].fillna("").str.lower().str.strip()
        counts = key.value_counts()
        multi = counts[counts > 1]
        n_multi_records = int(multi.sum())
        out[str(c)] = {
            "records_sharing_name": n_multi_records,
            "unique_shared_names": int(len(multi)),
            "fraction_records_in_shared": n_multi_records / max(len(g), 1),
        }
    return out


def _source_asymmetry(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
                      gt_long: pd.DataFrame, sample: int = 20_000) -> dict:
    """Compare S1↔S2 vs S1↔S3 name Jaro similarity on true pairs (subsample)."""
    try:
        from rapidfuzz.distance import JaroWinkler
    except Exception:
        return {"error": "rapidfuzz not installed"}
    s1n = s1.set_index("entity_id")["business_name"].fillna("").to_dict()
    s2n = s2.set_index("entity_id")["business_name"].fillna("").to_dict()
    s3n = s3.set_index("entity_id")["business_name"].fillna("").to_dict()

    rng = np.random.default_rng(C.SEED)
    gt_long = gt_long.sample(n=min(len(gt_long), sample), random_state=C.SEED)
    result: dict[str, list[float]] = {"s2": [], "s3": []}
    for s1_id, mid in zip(gt_long["source1_entity_id"], gt_long["matched_id"]):
        a = s1n.get(s1_id, "")
        if mid.startswith("S2-"):
            b = s2n.get(mid)
            if b is not None:
                result["s2"].append(JaroWinkler.normalized_similarity(a.lower(), b.lower()))
        elif mid.startswith("S3-"):
            b = s3n.get(mid)
            if b is not None:
                result["s3"].append(JaroWinkler.normalized_similarity(a.lower(), b.lower()))
    return {
        k: {"n": len(v), "mean": float(np.mean(v)) if v else None,
            "p25": float(np.quantile(v, .25)) if v else None,
            "p50": float(np.quantile(v, .50)) if v else None,
            "p75": float(np.quantile(v, .75)) if v else None}
        for k, v in result.items()
    }


def _matches_split_by_source(gt: pd.DataFrame) -> dict[str, float]:
    n_s2, n_s3, total = 0, 0, 0
    for ids in gt["matched_entity_ids"]:
        for i in (ids or "").split(","):
            i = i.strip()
            if not i:
                continue
            total += 1
            if i.startswith("S2-"):
                n_s2 += 1
            elif i.startswith("S3-"):
                n_s3 += 1
    return {"total_true_pairs": total,
            "share_S2": n_s2 / max(total, 1),
            "share_S3": n_s3 / max(total, 1)}


def run() -> dict:
    C.ensure_dirs()
    print("[stage0_checks] loading TSVs…")
    train_s1 = read_source_tsv(C.TRAIN_SOURCE["s1"])
    train_s2 = read_source_tsv(C.TRAIN_SOURCE["s2"])
    train_s3 = read_source_tsv(C.TRAIN_SOURCE["s3"])
    test_s1 = read_source_tsv(C.TEST_SOURCE["s1"])
    test_s2 = read_source_tsv(C.TEST_SOURCE["s2"])
    test_s3 = read_source_tsv(C.TEST_SOURCE["s3"])
    gt = read_ground_truth(C.TRAIN_GT)
    gt_long = explode_ground_truth(gt)

    report: dict = {}
    print("[stage0_checks] postal-code coverage…")
    report["postal_coverage"] = {
        "train": {n: _cov(df, _has_postal) for n, df in
                  [("s1", train_s1), ("s2", train_s2), ("s3", train_s3)]},
        "test":  {n: _cov(df, _has_postal) for n, df in
                  [("s1", test_s1), ("s2", test_s2), ("s3", test_s3)]},
    }
    print("[stage0_checks] house-number coverage…")
    report["house_number_coverage"] = {
        "train": {n: _cov(df, _has_house) for n, df in
                  [("s1", train_s1), ("s2", train_s2), ("s3", train_s3)]},
        "test":  {n: _cov(df, _has_house) for n, df in
                  [("s1", test_s1), ("s2", test_s2), ("s3", test_s3)]},
    }
    print("[stage0_checks] script distribution…")
    script = {}
    for split_name, files in [("train", [("s1", train_s1), ("s2", train_s2), ("s3", train_s3)]),
                               ("test",  [("s1", test_s1), ("s2", test_s2), ("s3", test_s3)])]:
        script[split_name] = {}
        for name, df in files:
            script[split_name][name] = {
                "name": _script_dist(df, "business_name"),
                "address": _script_dist(df, "business_address"),
            }
    report["script_distribution"] = script

    print("[stage0_checks] name duplication (chain problem size)…")
    report["name_duplication"] = {
        "train": {n: _name_dup(df) for n, df in
                  [("s1", train_s1), ("s2", train_s2), ("s3", train_s3)]},
        "test":  {n: _name_dup(df) for n, df in
                  [("s1", test_s1), ("s2", test_s2), ("s3", test_s3)]},
    }

    print("[stage0_checks] source asymmetry (name similarity on true pairs)…")
    report["source_asymmetry_name_jaro"] = _source_asymmetry(
        train_s1, train_s2, train_s3, gt_long
    )
    print("[stage0_checks] matches split by source…")
    report["matches_by_source"] = _matches_split_by_source(gt)

    C.STAGE0_REPORT.parent.mkdir(parents=True, exist_ok=True)
    C.STAGE0_REPORT.write_text(_pretty(report), encoding="utf-8")
    (C.REPORTS_DIR / "stage0_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"[stage0_checks] wrote {C.STAGE0_REPORT}")
    return report


def _pretty(d: dict, indent: int = 0) -> str:
    lines: list[str] = []
    pad = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.append(_pretty(v, indent + 1))
        else:
            lines.append(f"{pad}{k}: {v}")
    return "\n".join(lines)


if __name__ == "__main__":
    run()
