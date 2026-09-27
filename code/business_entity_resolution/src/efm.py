"""Expected-F0.5 maximising decoder (per-entity Bayes-optimal match sets).

The leaderboard metric is F0.5 computed PER Source-1 entity and then averaged,
so a single global probability threshold is not optimal: the best set to
predict for an entity depends on its whole candidate distribution.

For every S1 we take its candidates' calibrated probabilities q_1 >= ... >= q_n
(after one-to-one resolution: a candidate is kept only for the S1 that gives
it the highest probability), treat them as independent Bernoullis, add
Poisson(lambda) true matches that blocking never surfaced, and compute the
exact expected F0.5 of predicting the top-k set for every k = 0..n:

    k = 0 :  E[F] = P(no true match at all)
    k >= 1:  E[F] = E[ 1.25 t / (k + 0.25 (t + u)) ]
             t = #true among top-k, u = #true among the rest + missed

and predict the arg-max set. Probabilities are calibrated with isotonic
regression on V_a; lambda (per country) and its scale are fitted on V_a; the
decoder is adopted only if it beats threshold tuning on the held-out V_b.
"""
from __future__ import annotations
import json
import math

import numpy as np
import pandas as pd

from . import config as C
from . import splits as splits_mod
from . import tune_thresholds as T
from .io_utils import read_source_tsv, write_id_list_tsv
from .metrics import macro_f_beta

KEY = ["s1_id", "cand_id", "cand_source"]
NMAX = 12
POIS_TERMS = 12


def _one_to_one(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("q", ascending=False, kind="stable")
    return df[~df.duplicated("cand_id", keep="first")]


def efm_select(df: pd.DataFrame, lam: float) -> pd.DataFrame:
    """df: s1_id, cand_id, q. Returns the rows of the expected-F-optimal sets."""
    df = _one_to_one(df[["s1_id", "cand_id", "q"]])
    df = df.sort_values(["s1_id", "q"], ascending=[True, False], kind="stable")
    df = df.assign(_r=df.groupby("s1_id", sort=False).cumcount().to_numpy())
    df = df[df["_r"] < NMAX]
    codes, uniq = pd.factorize(df["s1_id"], sort=False)
    n_ent = len(uniq)
    P = np.zeros((n_ent, NMAX))
    P[codes, df["_r"].to_numpy()] = df["q"].to_numpy()

    pref = [np.ones((n_ent, 1))]                       # dist. of #true in top-k
    for k in range(NMAX):
        prev, p = pref[-1], P[:, k:k + 1]
        new = np.zeros((n_ent, k + 2))
        new[:, :k + 1] += prev * (1 - p)
        new[:, 1:] += prev * p
        pref.append(new)
    suf = [None] * (NMAX + 1)                          # dist. of #true in items k..
    suf[NMAX] = np.ones((n_ent, 1))
    for k in range(NMAX - 1, -1, -1):
        nxt, p = suf[k + 1], P[:, k:k + 1]
        L = nxt.shape[1]
        new = np.zeros((n_ent, L + 1))
        new[:, :L] += nxt * (1 - p)
        new[:, 1:] += nxt * p
        suf[k] = new
    pois = np.array([math.exp(-lam) * lam ** j / math.factorial(j) for j in range(POIS_TERMS)])

    E = np.zeros((n_ent, NMAX + 1))
    E[:, 0] = suf[0][:, 0] * pois[0]
    for k in range(1, NMAX + 1):
        U = suf[k]
        L = U.shape[1]
        Uc = np.zeros((n_ent, L + POIS_TERMS - 1))
        for j, w in enumerate(pois):
            Uc[:, j:j + L] += U * w
        b = np.arange(Uc.shape[1], dtype=np.float64)
        A = pref[k]
        for a in range(1, k + 1):
            E[:, k] += A[:, a] * (Uc @ (1.25 * a / (k + 0.25 * (a + b))))
    kstar = E.argmax(axis=1)                           # first max -> smaller set on ties
    return df[df["_r"].to_numpy() < kstar[codes]]


def _preds(kept: pd.DataFrame, ids) -> dict:
    out = {s: [] for s in ids}
    for s, c in zip(kept["s1_id"], kept["cand_id"]):
        if s in out:
            out[s].append(c)
    return out


def _tune_thresholds(scores: pd.DataFrame, truths: dict) -> float:
    coarse = T._grid_eval(scores, truths, C.COARSE_PAIR_GRID, C.COARSE_PAIR_GRID,
                          C.COARSE_MARGIN_GRID)
    b = coarse.sort_values("macro_f05", ascending=False).iloc[0]
    p0, s0 = float(b["pair"]), float(b["singleton"])
    fp = np.round(np.arange(max(0.01, p0 - C.FINE_HALFWIDTH), p0 + C.FINE_HALFWIDTH + 1e-9, C.FINE_STEP), 3).tolist()
    fs = np.round(np.arange(max(p0, s0 - C.FINE_HALFWIDTH), s0 + C.FINE_HALFWIDTH + 1e-9, C.FINE_STEP), 3).tolist()
    fine = T._grid_eval(scores, truths, fp, fs, [None if b["margin"] < 0 else float(b["margin"])])
    return float(fine["macro_f05"].max())


def run(out_dir_name: str = "output_efm") -> dict:
    from sklearn.isotonic import IsotonicRegression
    sp = splits_mod.load()
    v = sp[sp["group"] == "V"]
    rng = np.random.default_rng(C.SEED)                # same V_a / V_b split as stage2
    v_ids = v["entity_id"].to_numpy().copy()
    rng.shuffle(v_ids)
    va = set(v_ids[: len(v_ids) // 2]); vb = set(v_ids[len(v_ids) // 2:])

    countries = sorted(p.stem.split("__")[1] for p in C.FINAL_SCORES_DIR.glob("train__*__scores.parquet"))
    parts = []
    for c in countries:
        s = pd.read_parquet(C.FINAL_SCORES_DIR / f"train__{c}__scores.parquet")
        y = pd.read_parquet(C.FINAL_SCORES_DIR / f"train__{c}__features.parquet", columns=KEY + ["y"])
        d = s.merge(y, on=KEY, how="left")
        d = d[d["s1_id"].isin(va | vb)].copy(); d["_country"] = c
        parts.append(d)
    V = pd.concat(parts, ignore_index=True)
    in_a = V["s1_id"].isin(va).to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5)
    iso.fit(V.loc[in_a, "p_match"].to_numpy(), V.loc[in_a, "y"].to_numpy())
    V["q"] = iso.predict(V["p_match"].to_numpy())

    truths_all = T._v_ground_truth()
    rep = {"countries": {}}
    lam_c = {}
    for c in countries:
        ids_c = set(v.loc[v["country"] == c, "entity_id"])
        tr_a = {s: t for s, t in truths_all.items() if s in va and s in ids_c}
        cand_a = V[in_a & (V["_country"] == c).to_numpy()]
        found = int(cand_a["y"].sum())
        lam_c[c] = max(0.0, (sum(len(t) for t in tr_a.values()) - found) / max(len(tr_a), 1))
    # scale of lambda chosen on V_a (all countries together)
    best = None
    for m in (0.0, 0.5, 1.0):
        preds, truths = {}, {}
        for c in countries:
            ids_c = set(v.loc[v["country"] == c, "entity_id"])
            tr_a = {s: t for s, t in truths_all.items() if s in va and s in ids_c}
            kept = efm_select(V[in_a & (V["_country"] == c).to_numpy()], lam_c[c] * m)
            preds.update(_preds(kept, tr_a)); truths.update(tr_a)
        f = macro_f_beta(preds, truths)["macro_f_0_5"]
        print(f"[efm] V_a lambda-scale {m}: F0.5 {f:.5f}", flush=True)
        if best is None or f > best[1]:
            best = (m, f)
    m = best[0]

    # held-out comparison on V_b: EFM vs tuned thresholds (Stage-B p_match)
    preds, truths = {}, {}
    for c in countries:
        ids_c = set(v.loc[v["country"] == c, "entity_id"])
        tr_b = {s: t for s, t in truths_all.items() if s in vb and s in ids_c}
        kept = efm_select(V[~in_a & (V["_country"] == c).to_numpy()], lam_c[c] * m)
        pc = _preds(kept, tr_b)
        rep["countries"][c] = {"lambda": lam_c[c], "efm_f05_Vb": macro_f_beta(pc, tr_b)["macro_f_0_5"]}
        preds.update(pc); truths.update(tr_b)
    f_efm = macro_f_beta(preds, truths)["macro_f_0_5"]
    f_thr = _tune_thresholds(V[~in_a][KEY + ["p_match", "_country"]], truths)
    rep.update({"lambda_scale": m, "efm_f05_Vb": f_efm, "thresholds_f05_Vb": f_thr,
                "adopt": bool(f_efm > f_thr)})
    print(f"[efm] V_b  thresholds (tuned on V_b) F0.5 {f_thr:.5f}  ->  EFM F0.5 {f_efm:.5f}  "
          f"adopt={rep['adopt']}  per-country {rep['countries']}", flush=True)

    # test output (written regardless; use only if adopt)
    test_s1 = read_source_tsv(C.TEST_SOURCE["s1"])
    order = test_s1["entity_id"].tolist()
    lam_default = float(np.mean(list(lam_c.values()))) if lam_c else 0.0
    by_s1: dict[str, list[tuple[str, float]]] = {s: [] for s in order}
    for p in sorted(C.FINAL_SCORES_DIR.glob("test__*__scores.parquet")):
        country = p.stem.split("__")[1]
        d = pd.read_parquet(p)
        d["q"] = iso.predict(d["p_match"].to_numpy())
        kept = efm_select(d, lam_c.get(country, lam_default) * m)
        for s, c, q in zip(kept["s1_id"], kept["cand_id"], kept["q"]):
            by_s1[s].append((c, float(q)))
        print(f"[efm] test/{country}: {len(kept):,} matches for {d['s1_id'].nunique():,} S1", flush=True)
    out_dir = C.OUTPUT_DIR.parent / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    lists = [[c for c, _ in sorted(by_s1[s], key=lambda x: -x[1])] for s in order]
    write_id_list_tsv(out_dir / "matching_results.tsv", order, lists, id_column="matched_entity_ids")
    n_ne = sum(1 for x in lists if x)
    print(f"[efm] wrote {out_dir / 'matching_results.tsv'}  non-empty={n_ne:,}/{len(order):,}  "
          f"mean-matches/S1={sum(map(len, lists)) / max(len(order), 1):.3f}", flush=True)
    (C.REPORTS_DIR / "efm.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


if __name__ == "__main__":
    run()
