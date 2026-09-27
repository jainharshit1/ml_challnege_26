"""Stage-C context reranker (listwise re-scoring of Stage-B probabilities).

Stage-B scores each (S1, candidate) pair mostly in isolation. Stage-C re-scores
every pair with the Stage-B scores of its *neighbours*: the other candidates of
the same S1 (gap to best, number of strong candidates, best score in the other
source) and the other S1 entities competing for the same candidate.

Leakage control: Stage-B never trains on group V, so its V scores are honest.
V entities are split 50/50 (seeded): Stage-C trains on V_a and thresholds are
tuned on V_b only. Stage-C is adopted only if its tuned macro F0.5 on V_b beats
Stage-B tuned the same way on the same V_b.

Outputs (FINAL_SCORES_DIR): {split}__{country}__scores__s2.parquet (p_match)
Reports: reports/stage2.json (V_b comparison + thresholds used for decide).
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd

from . import config as C
from . import splits as splits_mod
from . import tune_thresholds as T

KEY = ["s1_id", "cand_id", "cand_source"]
BASE_FEATS = ["f_stage_a", "f_dense_cos", "f_name_jw_core", "f_name_phon_jw",
              "f_name_phon_set", "f_addr_jw", "f_addr_set", "f_house_eq",
              "f_postal_eq", "f_name_freq_log", "f_n_arms_hit", "f_cand_source_s2"]


def _context(df: pd.DataFrame) -> pd.DataFrame:
    """Neighbour features from Stage-B p_match (df has KEY + p_match + base)."""
    p = df["p_match"]
    g = df.groupby("s1_id", sort=False)["p_match"]
    out = pd.DataFrame(index=df.index)
    out["c_p"] = p
    out["c_s1_max"] = g.transform("max")
    out["c_s1_gap"] = out["c_s1_max"] - p
    out["c_s1_rank"] = g.rank(method="first", ascending=False)
    out["c_s1_sum"] = g.transform("sum")
    out["c_s1_n50"] = (p > 0.5).groupby(df["s1_id"], sort=False).transform("sum")
    out["c_s1_n80"] = (p > 0.8).groupby(df["s1_id"], sort=False).transform("sum")
    srt = df.assign(_p=p).sort_values(["s1_id", "_p"], ascending=[True, False])
    second = srt.groupby("s1_id", sort=False)["_p"].nth(1)
    second = pd.Series(second.to_numpy(), index=srt.loc[second.index, "s1_id"].to_numpy())
    out["c_s1_2nd"] = df["s1_id"].map(second).fillna(0.0).to_numpy()
    out["c_s1_gap12"] = out["c_s1_max"] - out["c_s1_2nd"]
    # best score for the same S1 in the other source (S2 vs S3)
    src_max = df.groupby(["s1_id", "cand_source"], sort=False)["p_match"].max()
    other = df["cand_source"].map({"S2": "S3", "S3": "S2"})
    idx = pd.MultiIndex.from_arrays([df["s1_id"], other])
    out["c_other_src_max"] = src_max.reindex(idx).fillna(-1.0).to_numpy()
    # competition for the same candidate across S1 entities
    gc = df.groupby("cand_id", sort=False)["p_match"]
    out["c_cand_max"] = gc.transform("max")
    out["c_cand_gap"] = out["c_cand_max"] - p
    out["c_cand_n"] = gc.transform("size")
    out["c_cand_rank"] = gc.rank(method="first", ascending=False)
    for f in BASE_FEATS:
        if f in df.columns:
            out[f] = df[f].to_numpy()
    return out.astype("float32")


def _load(split: str, country: str) -> pd.DataFrame:
    f = pd.read_parquet(C.FINAL_SCORES_DIR / f"{split}__{country}__features.parquet",
                        columns=None)
    s = pd.read_parquet(C.FINAL_SCORES_DIR / f"{split}__{country}__scores.parquet")
    keep = KEY + [c for c in BASE_FEATS if c in f.columns] + (["y"] if "y" in f.columns else [])
    return s.merge(f[keep], on=KEY, how="left")


def _tune(scores: pd.DataFrame, truths: dict) -> dict:
    """Coarse + fine grid exactly as tune_thresholds.tune (on the given data)."""
    coarse = T._grid_eval(scores, truths, C.COARSE_PAIR_GRID, C.COARSE_PAIR_GRID,
                          C.COARSE_MARGIN_GRID)
    b = coarse.sort_values("macro_f05", ascending=False).iloc[0]
    p0, s0 = float(b["pair"]), float(b["singleton"])
    fp = np.round(np.arange(max(0.01, p0 - C.FINE_HALFWIDTH), p0 + C.FINE_HALFWIDTH + 1e-9, C.FINE_STEP), 3).tolist()
    fs = np.round(np.arange(max(p0, s0 - C.FINE_HALFWIDTH), s0 + C.FINE_HALFWIDTH + 1e-9, C.FINE_STEP), 3).tolist()
    fine = T._grid_eval(scores, truths, fp, fs, [None if b["margin"] < 0 else float(b["margin"])])
    b2 = fine.sort_values("macro_f05", ascending=False).iloc[0]
    return {"pair": float(b2["pair"]), "singleton": float(b2["singleton"]),
            "margin": None if b2["margin"] < 0 else float(b2["margin"]),
            "f05": float(b2["macro_f05"])}


def run() -> dict:
    import lightgbm as lgb
    C.ensure_dirs()
    sp = splits_mod.load()
    v = sp[sp["group"] == "V"]
    rng = np.random.default_rng(C.SEED)
    v_ids = v["entity_id"].to_numpy().copy()
    rng.shuffle(v_ids)
    va = set(v_ids[: len(v_ids) // 2]); vb = set(v_ids[len(v_ids) // 2:])

    train_countries = sorted(p.stem.split("__")[1] for p in
                             C.FINAL_SCORES_DIR.glob("train__*__scores.parquet"))
    # Context is computed on ALL train candidates of a country (as at test
    # time, where every S1 competes), then restricted to V rows.
    parts, xparts = [], []
    for c in train_countries:
        d = _load("train", c)
        ctx = _context(d)
        m = d["s1_id"].isin(va | vb).to_numpy()
        d = d[m].reset_index(drop=True); d["_country"] = c
        parts.append(d); xparts.append(ctx[m].reset_index(drop=True))
    V = pd.concat(parts, ignore_index=True)
    X = pd.concat(xparts, ignore_index=True)
    y = V["y"].to_numpy()
    in_a = V["s1_id"].isin(va).to_numpy()

    # Stage-C model on V_a (10 % of V_a entities for early stopping)
    a_ids = np.array(sorted(va)); rng.shuffle(a_ids)
    es = set(a_ids[: len(a_ids) // 10])
    tr = in_a & ~V["s1_id"].isin(es).to_numpy()
    ev = in_a & V["s1_id"].isin(es).to_numpy()
    params = dict(objective="binary", metric=["binary_logloss"], num_leaves=31,
                  learning_rate=0.05, feature_fraction=0.9, bagging_fraction=0.8,
                  bagging_freq=1, min_data_in_leaf=200, verbose=-1, seed=C.SEED,
                  num_threads=C.N_JOBS)
    booster = lgb.train(params, lgb.Dataset(X[tr].values, label=y[tr]),
                        num_boost_round=1000,
                        valid_sets=[lgb.Dataset(X[ev].values, label=y[ev])],
                        callbacks=[lgb.early_stopping(30, first_metric_only=True),
                                   lgb.log_evaluation(100)])
    feat_names = list(X.columns)
    booster.save_model(str(C.MODELS_DIR / "stage_c_lgbm.txt"))
    (C.MODELS_DIR / "stage_c_features.txt").write_text("\n".join(feat_names), encoding="utf-8")

    # Compare on V_b: Stage-B vs Stage-C, each tuned per country on V_b
    truths_all = T._v_ground_truth()
    Vb = V[~in_a].copy()
    Vb_c = Vb.copy(); Vb_c["p_match"] = booster.predict(X[~in_a].values)
    report = {"countries": {}}
    for c in train_countries:
        tr_c = {s: t for s, t in truths_all.items()
                if s in vb and s in set(v.loc[v["country"] == c, "entity_id"])}
        sb = _tune(Vb[Vb["_country"] == c][KEY + ["p_match", "_country"]], tr_c)
        sc = _tune(Vb_c[Vb_c["_country"] == c][KEY + ["p_match", "_country"]], tr_c)
        report["countries"][c] = {"stage_b": sb, "stage_c": sc}
        print(f"[stage2] {c} V_b  Stage-B F0.5 {sb['f05']:.4f}  ->  Stage-C F0.5 {sc['f05']:.4f}", flush=True)
    tr_b = {s: t for s, t in truths_all.items() if s in vb}
    gb = _tune(Vb[KEY + ["p_match", "_country"]], tr_b)
    gc = _tune(Vb_c[KEY + ["p_match", "_country"]], tr_b)
    report["global"] = {"stage_b": gb, "stage_c": gc}
    report["adopt"] = bool(gc["f05"] > gb["f05"])
    print(f"[stage2] ALL V_b  Stage-B F0.5 {gb['f05']:.4f}  ->  Stage-C F0.5 {gc['f05']:.4f}  "
          f"adopt={report['adopt']}", flush=True)

    # Score every split/country with Stage-C (written regardless; decide uses
    # them only when adopted)
    for p in sorted(C.FINAL_SCORES_DIR.glob("*__scores.parquet")):
        split, country, _ = p.stem.split("__")
        d = _load(split, country)
        Xc = _context(d).reindex(columns=feat_names)
        out = d[KEY].copy(); out["p_match"] = booster.predict(Xc.values).astype(np.float32)
        out.to_parquet(C.FINAL_SCORES_DIR / f"{split}__{country}__scores__s2.parquet", index=False)
        print(f"[stage2] scored {split}/{country} ({len(out):,} pairs)", flush=True)
    (C.REPORTS_DIR / "stage2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def decide_test() -> None:
    """Write output/matching_results.tsv from Stage-C scores with V_b-tuned
    thresholds (per seen country; global for unseen countries such as France)."""
    from . import decide as D
    rep = json.loads((C.REPORTS_DIR / "stage2.json").read_text("utf-8"))
    if not rep.get("adopt"):
        print("[stage2] Stage-C not better on V_b -> leaving Stage-B output untouched")
        return
    g = rep["global"]["stage_c"]
    overrides = {c: (r["stage_c"]["pair"], r["stage_c"]["singleton"], r["stage_c"]["margin"])
                 for c, r in rep["countries"].items()}
    print(f"[stage2] decide with Stage-C: global {g}, per-country {overrides}")
    D.decide_all_test(pair_thresh=g["pair"], singleton_thresh=g["singleton"],
                      margin_thresh=g["margin"], loco_tag="__s2",
                      country_overrides=overrides)


if __name__ == "__main__":
    run()
    decide_test()
