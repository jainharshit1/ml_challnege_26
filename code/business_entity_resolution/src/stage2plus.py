"""Stage-C+ anchor reranker (transitive evidence from each S1's best match).

Stage-B compares a candidate c only with its S1 record s. For most S1s one
candidate is already an almost-certain match (the *anchor*). The anchor is a
second, often richer view of the same entity (full address, house number,
another script), so comparing every other candidate with the anchor
  * rescues true variants that disagree with s but agree with the anchor,
  * pushes down chain branches that share only the name.

Anchor of pair (s, c) = highest Stage-B p_match candidate of s other than c.
Model input = all Stage-B features + stage2 context features + anchor features.
Trained on V_a, compared on held-out V_b (same seeded split as stage2/efm).
Writes output_stagecplus/matching_results.tsv (never touches output/).
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd

from . import config as C
from . import decide as D
from . import embed as embed_mod
from . import splits as splits_mod
from . import stage2 as S2
from . import tune_thresholds as T
from .features import _phon, _positions
from .io_utils import read_source_tsv, write_id_list_tsv

KEY = S2.KEY
STR_COLS = ["core_name", "address_expanded", "house_number", "all_numbers"]
MISSING = -1.0
CHUNK = 200_000


def _pool(split: str, country: str) -> dict:
    ids, src, cols = [], [], {c: [] for c in STR_COLS}
    for s in ("s2", "s3"):
        p = C.NORMALIZED_DIR / f"{split}__{s}__{country}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["entity_id"] + STR_COLS)
        ids.append(d["entity_id"].to_numpy()); src.append(np.full(len(d), s.upper(), dtype=object))
        for c in STR_COLS:
            cols[c].append(d[c].to_numpy(dtype=object, na_value=""))
    out = {"ids": np.concatenate(ids), "src": np.concatenate(src)}
    for c in STR_COLS:
        out[c] = np.concatenate(cols[c])
    return out


def _anchors(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Row index of each pair's anchor (-1 if the S1 has a single candidate)."""
    idx = np.arange(len(d))
    srt = pd.DataFrame({"s": d["s1_id"].to_numpy(), "p": d["p_match"].to_numpy(), "i": idx})
    srt = srt.sort_values(["s", "p"], ascending=[True, False], kind="stable")
    rk = srt.groupby("s", sort=False).cumcount().to_numpy()
    rank = np.empty(len(d), dtype=np.int64); rank[srt["i"].to_numpy()] = rk
    first = pd.Series(srt["i"].to_numpy()[rk == 0], index=srt["s"].to_numpy()[rk == 0])
    second = pd.Series(srt["i"].to_numpy()[rk == 1], index=srt["s"].to_numpy()[rk == 1])
    a1 = d["s1_id"].map(first).to_numpy(dtype=np.float64)
    a2 = d["s1_id"].map(second).to_numpy(dtype=np.float64)
    a = np.where(rank == 0, a2, a1)
    return np.where(np.isnan(a), -1, a).astype(np.int64), rank


def _anchor_feats(d: pd.DataFrame, split: str, country: str, pool: dict) -> pd.DataFrame:
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import JaroWinkler
    n = len(d)
    a, rank = _anchors(d)
    has = a >= 0
    cid = d["cand_id"].to_numpy(); aid = np.where(has, cid[np.maximum(a, 0)], "")
    cpos = _positions(pool["ids"], cid); apos = _positions(pool["ids"], aid)
    ok = has & (cpos >= 0) & (apos >= 0)
    rows = np.flatnonzero(ok)

    def g(col, pos):
        return pool[col][pos[rows]].tolist()

    out = {f: np.full(n, MISSING, dtype=np.float32) for f in (
        "an_name_jw", "an_name_set", "an_phon_jw", "an_addr_jw", "an_addr_set",
        "an_nums_set", "an_house_eq", "an_dense_cos")}
    workers = max(1, C.N_JOBS)
    if len(rows):
        cn, an = g("core_name", cpos), g("core_name", apos)
        ca, aa = g("address_expanded", cpos), g("address_expanded", apos)
        ch, ah = g("house_number", cpos), g("house_number", apos)
        cu, au = g("all_numbers", cpos), g("all_numbers", apos)
        pd_ = lambda x, y, sc, s=1.0: (process.cpdist(x, y, scorer=sc, workers=workers, dtype=np.float64) / s).astype(np.float32)
        out["an_name_jw"][rows] = pd_(cn, an, JaroWinkler.normalized_similarity)
        out["an_name_set"][rows] = pd_(cn, an, fuzz.token_set_ratio, 100.0)
        memo = {}
        ph = lambda x: memo[x] if x in memo else memo.setdefault(x, _phon(x))
        out["an_phon_jw"][rows] = pd_([ph(x) for x in cn], [ph(x) for x in an], JaroWinkler.normalized_similarity)
        out["an_addr_jw"][rows] = pd_(ca, aa, JaroWinkler.normalized_similarity)
        out["an_addr_set"][rows] = pd_(ca, aa, fuzz.token_set_ratio, 100.0)
        out["an_nums_set"][rows] = pd_(cu, au, fuzz.token_set_ratio, 100.0)
        chh, ahh = np.asarray(ch, dtype=object), np.asarray(ah, dtype=object)
        out["an_house_eq"][rows] = ((chh == ahh) & (chh != "")).astype(np.float32)
        # dense cosine candidate <-> anchor
        try:
            vec, pos = {}, {}
            for s in ("S2", "S3"):
                v, i = embed_mod.load_embeddings(split, s.lower(), country)
                vec[s] = v; pos[s] = i["entity_id"].to_numpy()
            csrc, asrc = pool["src"][cpos[rows]], pool["src"][apos[rows]]
            cvp = np.full(len(rows), -1, dtype=np.int64); avp = np.full(len(rows), -1, dtype=np.int64)
            for s in ("S2", "S3"):
                m = csrc == s
                cvp[m] = _positions(pos[s], cid[rows][m])
                m = asrc == s
                avp[m] = _positions(pos[s], aid[rows][m])
            res = np.full(len(rows), MISSING, dtype=np.float32)
            for st in range(0, len(rows), CHUNK):
                sl = slice(st, st + CHUNK)
                A = np.zeros((len(cvp[sl]), vec["S2"].shape[1]), dtype=np.float32); B = np.zeros_like(A)
                good = np.ones(len(A), dtype=bool)
                for s in ("S2", "S3"):
                    m = (csrc[sl] == s) & (cvp[sl] >= 0); A[m] = vec[s][cvp[sl][m]]
                    m2 = (asrc[sl] == s) & (avp[sl] >= 0); B[m2] = vec[s][avp[sl][m2]]
                good &= (cvp[sl] >= 0) & (avp[sl] >= 0)
                r = np.einsum("ij,ij->i", A, B); r[~good] = MISSING
                res[sl] = r
            out["an_dense_cos"][rows] = res
        except Exception as e:
            print(f"[stage2plus] {split}/{country}: dense anchor cos unavailable ({type(e).__name__}: {e})", flush=True)
        empty_a = (np.asarray(aa, dtype=object) == "").astype(np.float32)
        empty_c = (np.asarray(ca, dtype=object) == "").astype(np.float32)
    else:
        empty_a = empty_c = np.zeros(0, dtype=np.float32)
    ap = np.full(n, MISSING, dtype=np.float32); ap[has] = d["p_match"].to_numpy()[a[has]]
    same = np.zeros(n, dtype=np.float32)
    same[has] = (d["cand_source"].to_numpy()[has] == d["cand_source"].to_numpy()[a[has]]).astype(np.float32)
    out["an_p"] = ap; out["an_same_src"] = same; out["an_is_top"] = (rank == 0).astype(np.float32)
    out["an_addr_empty_c"] = np.full(n, MISSING, dtype=np.float32); out["an_addr_empty_c"][rows] = empty_c
    out["an_addr_empty_a"] = np.full(n, MISSING, dtype=np.float32); out["an_addr_empty_a"][rows] = empty_a
    return pd.DataFrame(out, index=d.index)


def _load_all(split: str, country: str) -> pd.DataFrame:
    s = pd.read_parquet(C.FINAL_SCORES_DIR / f"{split}__{country}__scores.parquet")
    f = pd.read_parquet(C.FINAL_SCORES_DIR / f"{split}__{country}__features.parquet")
    return s.merge(f, on=KEY, how="left")


def _design(d: pd.DataFrame, ctx: pd.DataFrame, anc: pd.DataFrame, f_cols: list[str]) -> pd.DataFrame:
    X = pd.concat([d[f_cols].reset_index(drop=True),
                   ctx[[c for c in ctx.columns if c.startswith("c_")]].reset_index(drop=True),
                   anc.reset_index(drop=True)], axis=1)
    return X.astype("float32")


def run() -> dict:
    import lightgbm as lgb
    sp = splits_mod.load()
    v = sp[sp["group"] == "V"]
    rng = np.random.default_rng(C.SEED)                       # identical to stage2/efm split
    v_ids = v["entity_id"].to_numpy().copy(); rng.shuffle(v_ids)
    va = set(v_ids[: len(v_ids) // 2]); vb = set(v_ids[len(v_ids) // 2:])
    countries = sorted(p.stem.split("__")[1] for p in C.FINAL_SCORES_DIR.glob("train__*__scores.parquet"))

    Xs, Ds, f_cols = [], [], None
    for c in countries:
        d = _load_all("train", c)
        if f_cols is None:
            f_cols = [x for x in d.columns if x.startswith("f_")]
        ctx = S2._context(d)                                   # on ALL train pairs (as at test)
        m = d["s1_id"].isin(va | vb).to_numpy()
        dv = d[m].reset_index(drop=True); dv["_country"] = c
        anc = _anchor_feats(dv, "train", c, _pool("train", c))
        Xs.append(_design(dv, ctx[m], anc, f_cols)); Ds.append(dv[KEY + ["p_match", "y", "_country"]])
        print(f"[stage2plus] train/{c}: {len(dv):,} V pairs featurised", flush=True)
    X = pd.concat(Xs, ignore_index=True); V = pd.concat(Ds, ignore_index=True)
    y = V["y"].to_numpy(); in_a = V["s1_id"].isin(va).to_numpy()
    a_ids = np.array(sorted(va)); rng.shuffle(a_ids); es = set(a_ids[: len(a_ids) // 10])
    tr = in_a & ~V["s1_id"].isin(es).to_numpy(); ev = in_a & V["s1_id"].isin(es).to_numpy()
    params = dict(objective="binary", metric=["binary_logloss"], num_leaves=63, learning_rate=0.05,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=200,
                  verbose=-1, seed=C.SEED, num_threads=C.N_JOBS)
    booster = lgb.train(params, lgb.Dataset(X[tr].values, label=y[tr]), num_boost_round=2000,
                        valid_sets=[lgb.Dataset(X[ev].values, label=y[ev])],
                        callbacks=[lgb.early_stopping(50, first_metric_only=True), lgb.log_evaluation(200)])
    feat_names = list(X.columns)
    booster.save_model(str(C.MODELS_DIR / "stage_cplus_lgbm.txt"))
    (C.MODELS_DIR / "stage_cplus_features.txt").write_text("\n".join(feat_names), encoding="utf-8")
    imp = pd.Series(booster.feature_importance("gain"), index=feat_names).sort_values(ascending=False)
    print("[stage2plus] top features:", ", ".join(imp.index[:12]), flush=True)

    truths_all = T._v_ground_truth()
    Vb = V[~in_a][KEY + ["p_match", "_country"]].copy()
    Vc = Vb.copy(); Vc["p_match"] = booster.predict(X[~in_a].values)
    rep = {"countries": {}}
    for c in countries:
        ids_c = set(v.loc[v["country"] == c, "entity_id"])
        tr_c = {s: t for s, t in truths_all.items() if s in vb and s in ids_c}
        b = S2._tune(Vb[Vb["_country"] == c], tr_c); p = S2._tune(Vc[Vc["_country"] == c], tr_c)
        rep["countries"][c] = {"stage_b": b, "stage_cplus": p}
        print(f"[stage2plus] {c} V_b  Stage-B {b['f05']:.5f}  ->  Stage-C+ {p['f05']:.5f}", flush=True)
    tr_b = {s: t for s, t in truths_all.items() if s in vb}
    gb, gp = S2._tune(Vb, tr_b), S2._tune(Vc, tr_b)
    rep["global"] = {"stage_b": gb, "stage_cplus": gp}
    rep["adopt"] = bool(gp["f05"] > gb["f05"])
    print(f"[stage2plus] ALL V_b  Stage-B {gb['f05']:.5f}  ->  Stage-C+ {gp['f05']:.5f}  adopt={rep['adopt']}", flush=True)
    (C.REPORTS_DIR / "stage2plus.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    # ---- test: score + decide (per-country thresholds from V_b; global for unseen) ----
    test_s1 = read_source_tsv(C.TEST_SOURCE["s1"]); order = test_s1["entity_id"].tolist()
    by_s1: dict[str, list[tuple[str, float]]] = {s: [] for s in order}
    g = rep["global"]["stage_cplus"]
    for p in sorted(C.FINAL_SCORES_DIR.glob("test__*__scores.parquet")):
        country = p.stem.split("__")[1]
        d = _load_all("test", country)
        ctx = S2._context(d)
        anc = _anchor_feats(d, "test", country, _pool("test", country))
        Xt = _design(d, ctx, anc, f_cols).reindex(columns=feat_names)
        sc = d[KEY].copy(); sc["p_match"] = booster.predict(Xt.values).astype(np.float32)
        sc.to_parquet(C.FINAL_SCORES_DIR / f"test__{country}__scores__cplus.parquet", index=False)
        t = rep["countries"].get(country, {}).get("stage_cplus", g)
        kept = D.decide(sc, pair_thresh=t["pair"], singleton_thresh=t["singleton"], margin_thresh=t["margin"])
        for s, cnd, q in zip(kept["s1_id"], kept["cand_id"], kept["p_match"]):
            by_s1[s].append((cnd, float(q)))
        print(f"[stage2plus] test/{country}: {len(kept):,} matches (thr {t['pair']}/{t['singleton']}/{t['margin']})", flush=True)
    out_dir = C.OUTPUT_DIR.parent / "output_stagecplus"; out_dir.mkdir(parents=True, exist_ok=True)
    lists = [[c for c, _ in sorted(by_s1[s], key=lambda x: -x[1])] for s in order]
    write_id_list_tsv(out_dir / "matching_results.tsv", order, lists, id_column="matched_entity_ids")
    print(f"[stage2plus] wrote {out_dir / 'matching_results.tsv'}  non-empty={sum(1 for x in lists if x):,}/{len(order):,}  "
          f"mean-matches/S1={sum(map(len, lists)) / max(len(order), 1):.3f}", flush=True)
    return rep


if __name__ == "__main__":
    run()
