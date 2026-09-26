"""Threshold tuning on V (plan §8.3–8.4).

Coarse then fine grid over (pair_thresh, singleton_thresh, margin_thresh),
scored against the FULL ground truth for V (not restricted to blocking-
survived pairs). Also implements the France procedure from §8.4:
  1. Read LOCO best thresholds (see train_gbdt.py loco mode).
  2. If unseen-country optimum shifts ≤ 0.03 vs in-country, use in-country.
  3. Else apply LOCO thresholds to France; keep in-country thresholds for
     US/India.
  4. Unsupervised diagnostics on France predictions guard against
     under-confidence.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import splits as splits_mod
from . import decide as decide_mod
from .metrics import macro_f_beta
from .io_utils import explode_ground_truth, read_ground_truth, read_source_tsv


def _load_v_scores(loco_tag: str = "") -> pd.DataFrame:
    split_df = splits_mod.load()
    v = set(split_df.loc[split_df["group"] == "V", "entity_id"])
    parts = []
    for p in C.FINAL_SCORES_DIR.glob(f"train__*__scores{loco_tag}.parquet"):
        country = p.stem.split("__")[1]
        df = pd.read_parquet(p)
        df = df[df["s1_id"].isin(v)].copy()
        df["_country"] = country
        parts.append(df)
    if not parts:
        raise RuntimeError("No V scored partitions found.")
    return pd.concat(parts, ignore_index=True)


def _v_ground_truth() -> dict[str, list[str]]:
    """FULL ground truth restricted to V entities."""
    split_df = splits_mod.load()
    v = set(split_df.loc[split_df["group"] == "V", "entity_id"])
    gt = read_ground_truth(C.TRAIN_GT)
    gt = gt[gt["source1_entity_id"].isin(v)]
    truth: dict[str, list[str]] = {}
    for sid, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        truth[sid] = [i for i in (ids or "").split(",") if i.strip()]
    # Include ALL V entities as keys (singletons count in eval)
    for sid in v:
        truth.setdefault(sid, [])
    return truth


def _grid_eval(scores_v: pd.DataFrame, truths: dict[str, list[str]],
               grid_pair: list[float], grid_singleton: list[float],
               grid_margin: list[float | None]) -> pd.DataFrame:
    rows = []
    for pt in grid_pair:
        for st in grid_singleton:
            if st < pt:
                continue
            for mg in grid_margin:
                kept = decide_mod.decide(
                    scores_v, pair_thresh=pt, singleton_thresh=st,
                    margin_thresh=mg, country_col="_country",
                )
                preds: dict[str, list[str]] = {sid: [] for sid in truths}
                for row in kept.itertuples(index=False):
                    preds[row.s1_id].append(row.cand_id)
                m = macro_f_beta(preds, truths)
                rows.append({
                    "pair": pt, "singleton": st,
                    "margin": mg if mg is not None else -1.0,
                    "macro_f05": m["macro_f_0_5"],
                    "singleton_f05": m["singleton_f_0_5"],
                    "nonsingleton_f05": m["nonsingleton_f_0_5"],
                    "pred_singleton_rate": m["predicted_singleton_rate"],
                    "mean_matches": m["mean_predicted_matches"],
                })
    return pd.DataFrame(rows)


def tune(loco_tag: str = "") -> dict:
    C.ensure_dirs()
    print(f"[tune] loading V scores (loco_tag={loco_tag!r})…")
    scores = _load_v_scores(loco_tag)
    truths = _v_ground_truth()

    print("[tune] coarse grid…")
    coarse = _grid_eval(scores, truths, C.COARSE_PAIR_GRID,
                        C.COARSE_PAIR_GRID, C.COARSE_MARGIN_GRID)
    best = coarse.sort_values("macro_f05", ascending=False).iloc[0]
    print(f"[tune] coarse best: pair={best['pair']}  singleton={best['singleton']}  "
          f"margin={best['margin']}  F0.5={best['macro_f05']:.4f}")

    p0, s0 = float(best["pair"]), float(best["singleton"])
    fine_pair = np.round(np.arange(max(0.01, p0 - C.FINE_HALFWIDTH),
                                    p0 + C.FINE_HALFWIDTH + 1e-9,
                                    C.FINE_STEP), 3).tolist()
    fine_sing = np.round(np.arange(max(p0, s0 - C.FINE_HALFWIDTH),
                                    s0 + C.FINE_HALFWIDTH + 1e-9,
                                    C.FINE_STEP), 3).tolist()
    fine = _grid_eval(scores, truths, fine_pair, fine_sing,
                      [None if best["margin"] < 0 else float(best["margin"])])
    best2 = fine.sort_values("macro_f05", ascending=False).iloc[0]
    print(f"[tune] fine best:   pair={best2['pair']}  singleton={best2['singleton']}  "
          f"margin={best2['margin']}  F0.5={best2['macro_f05']:.4f}")

    result = {
        "pair_thresh": float(best2["pair"]),
        "singleton_thresh": float(best2["singleton"]),
        "margin_thresh": None if best2["margin"] < 0 else float(best2["margin"]),
        "macro_f05_V": float(best2["macro_f05"]),
        "singleton_f05_V": float(best2["singleton_f05"]),
        "nonsingleton_f05_V": float(best2["nonsingleton_f05"]),
        "pred_singleton_rate": float(best2["pred_singleton_rate"]),
        "mean_matches": float(best2["mean_matches"]),
    }
    tag = loco_tag or "in_country"
    (C.REPORTS_DIR / f"thresholds_{tag}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    coarse.to_csv(C.REPORTS_DIR / f"grid_coarse_{tag}.tsv", sep="\t", index=False)
    fine.to_csv(C.REPORTS_DIR / f"grid_fine_{tag}.tsv", sep="\t", index=False)
    return result


def france_procedure() -> dict:
    """Implement §8.4. Requires that both in-country and LOCO training have
    already been run and their thresholds tuned.
    """
    inc = _read_or_none(C.REPORTS_DIR / "thresholds_in_country.json")
    us = _read_or_none(C.REPORTS_DIR / "thresholds___loco_US.json")
    ind = _read_or_none(C.REPORTS_DIR / "thresholds___loco_India.json")
    if inc is None:
        raise RuntimeError("Run tune() first (in-country).")

    france_pair = inc["pair_thresh"]
    france_singleton = inc["singleton_thresh"]
    france_margin = inc["margin_thresh"]
    shift_pair = 0.0
    if us and ind:
        avg_loco_pair = 0.5 * (us["pair_thresh"] + ind["pair_thresh"])
        avg_loco_sing = 0.5 * (us["singleton_thresh"] + ind["singleton_thresh"])
        shift_pair = abs(avg_loco_pair - inc["pair_thresh"])
        if shift_pair > 0.03:
            france_pair = avg_loco_pair
            france_singleton = avg_loco_sing
            print(f"[france] LOCO shift {shift_pair:.3f} > 0.03 — using LOCO "
                  f"thresholds for France: pair={france_pair:.2f} "
                  f"singleton={france_singleton:.2f}")

    return {
        "france_pair": france_pair,
        "france_singleton": france_singleton,
        "france_margin": france_margin,
        "loco_shift_pair": shift_pair,
        "in_country": inc,
    }


def france_diagnostic(france_pair: float, france_singleton: float,
                      france_margin: float | None) -> dict:
    """Predicted singleton rate & mean matches for France test predictions."""
    test_s1 = read_source_tsv(C.TEST_SOURCE["s1"])
    if C.NORMALIZE_ROW_CAP:
        test_s1 = test_s1.iloc[:C.NORMALIZE_ROW_CAP].copy()
    fr_ids = set(test_s1.loc[test_s1["country"] == "France", "entity_id"])
    if not fr_ids:
        return {"n_france": 0}
    p = C.FINAL_SCORES_DIR / "test__France__scores.parquet"
    if not p.exists():
        return {"missing": str(p)}
    scores = pd.read_parquet(p)
    kept = decide_mod.decide(scores, pair_thresh=france_pair,
                              singleton_thresh=france_singleton,
                              margin_thresh=france_margin)
    preds: dict[str, list[str]] = {sid: [] for sid in fr_ids}
    for row in kept.itertuples(index=False):
        preds[row.s1_id].append(row.cand_id)
    n_empty = sum(1 for v in preds.values() if not v)
    mean_matches = sum(len(v) for v in preds.values()) / max(len(preds), 1)
    result = {
        "n_france": len(preds),
        "pred_singleton_rate": n_empty / max(len(preds), 1),
        "mean_matches": mean_matches,
        "reference_train_singleton": 0.0558,
        "reference_train_mean_matches": 3.461,
    }
    (C.REPORTS_DIR / "france_diagnostics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"[france_diag] {result}")
    return result


def _read_or_none(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    tune()
