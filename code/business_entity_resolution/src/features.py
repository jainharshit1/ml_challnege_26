"""Stage-B feature builder (plan §7.1).

Produces the full per-pair feature matrix for the final GBDT. Combines
normalized fields, blocking-arm outputs, Stage-A score, cross-encoder score,
IDF-weighted overlaps, chain frequencies, S1-side competition features and
candidate-side competition features.

Country is NEVER emitted as a feature (plan §7.1 last line).

Usage (per partition):
    X, keys, y? = build(split, country)
"""
from __future__ import annotations
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import normalize as norm_mod
from .io_utils import explode_ground_truth, read_ground_truth


def _load_pool(split: str, country: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for src in ("s2", "s3"):
        p = C.NORMALIZED_DIR / f"{split}__{src}__{country}.parquet"
        if p.exists():
            out[src.upper()] = pd.read_parquet(p)
    return out


def _prep_lookup(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("entity_id")


_PHON_REP = (("sh", "s"), ("ch", "k"), ("kh", "k"), ("gh", "g"), ("th", "t"),
             ("dh", "d"), ("bh", "b"), ("ph", "p"), ("jh", "j"), ("ck", "k"),
             ("f", "p"), ("v", "b"), ("w", "b"), ("c", "k"), ("q", "k"),
             ("z", "j"), ("x", "ks"))
_PHON_VOWELS = re.compile(r"[aeiouy]")
_PHON_DOUBLE = re.compile(r"(.)\1+")
_PHON_NASAL = re.compile(r"m(?=[^aeiouym])")      # anusvara: 'imdastri' ~ 'indastri'
# consonant skeletons of legal suffixes (incl. transliterated 'praiveta limiteda')
_PHON_LEGAL = {"prbt", "prbr", "pbt", "lmtd", "lmt", "ltd", "lmrd", "llp", "ink",
               "inkrprtd", "kmpn", "krprtn", "krp", "ko"}


def _phon(name: str) -> str:
    """Transliteration-robust consonant skeleton of a romanised name.

    Indic names romanised via IAST carry an inherent 'a' after consonants and
    different consonant spellings than native Latin writing ('sivama inphoteka'
    vs 'shivam infotech'); both collapse to 'sbm inptk'. Legal suffixes are
    dropped because they survive transliteration unrecognised.
    """
    out = []
    for t in name.split():
        t = _PHON_NASAL.sub("n", t)
        for a, b in _PHON_REP:
            t = t.replace(a, b)
        t = _PHON_DOUBLE.sub(r"\1", t[:1] + _PHON_VOWELS.sub("", t[1:]))
        if t and t not in _PHON_LEGAL:
            out.append(t)
    return " ".join(out)


DENSE_COS_MISSING = -2.0   # outside cosine range; same value at train and test
_DENSE_COS_CHUNK = 100_000


def _eq_nonempty_list(a: list, b: list) -> np.ndarray:
    a = np.asarray(a, dtype=object); b = np.asarray(b, dtype=object)
    return ((a == b) & (a != "")).astype(np.int8)


def _positions(ids: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Row position of each key in `ids` (first occurrence), -1 if absent."""
    dup = pd.Index(ids).duplicated()
    uniq_pos = np.flatnonzero(~dup)
    pos = pd.Index(ids[~dup]).get_indexer(keys)
    return np.where(pos >= 0, uniq_pos[np.maximum(pos, 0)], -1)


def _dense_cos(split: str, country: str, s1_ids: np.ndarray,
               cand_ids: np.ndarray, src_arr: np.ndarray) -> np.ndarray:
    """Exact bge-m3 cosine for EVERY candidate pair, not only A1 hits.

    Reads the fp16 embedding memmaps written by the embed stage (vectors are
    unit-norm, so cosine = dot). Pairs whose embeddings are unavailable get
    DENSE_COS_MISSING; any failure degrades to that value instead of raising,
    so the feature column always exists with identical semantics.
    """
    from . import embed as embed_mod
    n = len(s1_ids)
    out = np.full(n, DENSE_COS_MISSING, dtype=np.float32)
    try:
        q_vec, q_ids = embed_mod.load_embeddings(split, "s1", country)
        q_pos = _positions(q_ids["entity_id"].to_numpy(), s1_ids)
    except Exception as e:
        print(f"[features] {split}/{country}: dense cos unavailable for S1 "
              f"({type(e).__name__}: {e}); using {DENSE_COS_MISSING}", flush=True)
        return out
    for src in pd.unique(src_arr):
        rows = np.flatnonzero(src_arr == src)
        try:
            p_vec, p_ids = embed_mod.load_embeddings(split, str(src).lower(), country)
            c_pos = _positions(p_ids["entity_id"].to_numpy(), cand_ids[rows])
            ok = (q_pos[rows] >= 0) & (c_pos >= 0)
            rows, qp, cp = rows[ok], q_pos[rows][ok], c_pos[ok]
            for s in range(0, len(rows), _DENSE_COS_CHUNK):
                e = s + _DENSE_COS_CHUNK
                a = np.asarray(q_vec[qp[s:e]], dtype=np.float32)
                b = np.asarray(p_vec[cp[s:e]], dtype=np.float32)
                out[rows[s:e]] = np.einsum("ij,ij->i", a, b)
        except Exception as e:
            print(f"[features] {split}/{country}: dense cos unavailable for {src} "
                  f"({type(e).__name__}: {e}); using {DENSE_COS_MISSING}", flush=True)
    return out


def _idf_sum(tokens: list[str], idf: dict[str, float]) -> float:
    return float(sum(idf.get(t, 0.0) for t in tokens))


def _idf_jaccard(a: str, b: str, idf: dict[str, float]) -> float:
    A = set((a or "").split())
    B = set((b or "").split())
    inter = A & B
    union = A | B
    if not union:
        return 1.0 if not A and not B else 0.0
    return _idf_sum(list(inter), idf) / max(_idf_sum(list(union), idf), 1e-9)


def _char3(s: str) -> set[str]:
    return set(s[i : i + 3] for i in range(len(s) - 2)) if len(s) >= 3 else set()


def _char3_jaccard(a: str, b: str) -> float:
    A, B = _char3(a), _char3(b)
    if not A and not B:
        return 1.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0


def _name_freq_map(split: str, country: str) -> dict[str, int]:
    """`name_freq` per row: |{records in country with same core_name}|."""
    cnt: Counter = Counter()
    for src in ("s1", "s2", "s3"):
        p = C.NORMALIZED_DIR / f"{split}__{src}__{country}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["core_name"])
        for x in df["core_name"].fillna(""):
            cnt[x] += 1
    return dict(cnt)


def build(split: str, country: str, include_labels: bool = True
          ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray | None]:
    """Return (X features, keys[s1_id, cand_id, cand_source], y_or_None)."""
    from rapidfuzz import fuzz as rf_fuzz  # type: ignore
    from rapidfuzz.distance import JaroWinkler, Levenshtein  # type: ignore

    pref = C.PREFILTER_DIR / f"{split}__{country}.parquet"
    rerank = C.RERANKER_DIR / f"{split}__{country}.parquet"
    if not pref.exists():
        return pd.DataFrame(), pd.DataFrame(), None
    cand = pd.read_parquet(pref)
    if rerank.exists():
        r = pd.read_parquet(rerank)
        cand = cand.merge(r, on=["s1_id", "cand_id"], how="left")
    else:
        # No reranker run (skipped on CPU): use the Stage-A score as the
        # rerank signal, as the cascade does, so the competition features
        # below stay informative instead of a constant -1.
        cand["rerank_score"] = cand["stage_a_score"]

    s1_lookup = _prep_lookup(pd.read_parquet(
        C.NORMALIZED_DIR / f"{split}__s1__{country}.parquet"
    ))
    str_cols = ["all_numbers", "street_tokens", "locality_tokens", "landmark_text", "acronym", "name_domain"]
    flag_cols = ["address_missing", "postal_missing", "house_number_missing",
                 "expansion_changed_name", "romanization_ok"]
    for col in str_cols:
        if col in s1_lookup: s1_lookup[col] = s1_lookup[col].fillna("")
    for col in flag_cols:
        if col in s1_lookup: s1_lookup[col] = s1_lookup[col].fillna(0)
    
    pools = _load_pool(split, country)
    pool_lookups = {}
    for k, v in pools.items():
        v = _prep_lookup(v)
        for col in str_cols:
            if col in v: v[col] = v[col].fillna("")
        for col in flag_cols:
            if col in v: v[col] = v[col].fillna(0)
        pool_lookups[k] = v

    idf = norm_mod.load_idf(split, country)
    name_freq = _name_freq_map(split, country)

    # ---- Bulk column lookup (C-level indexer) instead of per-row .loc ----
    s1_lookup = s1_lookup[~s1_lookup.index.duplicated()]
    pool_lookups = {k: v[~v.index.duplicated()] for k, v in pool_lookups.items()}
    s_pos = s1_lookup.index.get_indexer(cand["s1_id"].to_numpy())
    src_arr = cand["cand_source"].to_numpy()
    cid_arr = cand["cand_id"].to_numpy()
    c_pos = np.full(len(cand), -1, dtype=np.int64)
    for k, v in pool_lookups.items():
        m = src_arr == k
        if m.any():
            c_pos[m] = v.index.get_indexer(cid_arr[m])
    keep = (s_pos >= 0) & (c_pos >= 0)
    cand = cand[keep].reset_index(drop=True)
    s_pos, c_pos, src_arr = s_pos[keep], c_pos[keep], src_arr[keep]
    n = len(cand)
    if n == 0:
        return pd.DataFrame(), pd.DataFrame(), None

    def _s(col):
        return s1_lookup[col].to_numpy()[s_pos]

    def _c(col):
        out = np.empty(n, dtype=object)
        for k, v in pool_lookups.items():
            m = src_arr == k
            if m.any():
                out[m] = v[col].to_numpy()[c_pos[m]]
        return out

    def _str(a):
        return [x if isinstance(x, str) else "" for x in a]

    def _num(a):
        return pd.to_numeric(pd.Series(a), errors="coerce").fillna(0).to_numpy()

    s_core, c_core = _str(_s("core_name")), _str(_c("core_name"))
    s_exp, c_exp = _str(_s("name_expanded")), _str(_c("name_expanded"))
    s_addr, c_addr = _str(_s("address_expanded")), _str(_c("address_expanded"))
    s_leg, c_leg = _str(_s("legal_suffix")), _str(_c("legal_suffix"))
    s_acr, c_acr = _str(_s("acronym")), _str(_c("acronym"))
    s_dom, c_dom = _str(_s("name_domain")), _str(_c("name_domain"))
    s_nums, c_nums = _str(_s("all_numbers")), _str(_c("all_numbers"))
    s_street, c_street = _str(_s("street_tokens")), _str(_c("street_tokens"))
    s_loc, c_loc = _str(_s("locality_tokens")), _str(_c("locality_tokens"))
    s_lm, c_lm = _str(_s("landmark_text")), _str(_c("landmark_text"))

    # ---- rapidfuzz scores, element-wise and multi-threaded in C++ ----
    import os as _os
    from rapidfuzz import process as rf_process  # type: ignore
    workers = max(1, C.N_JOBS // max(1, int(_os.environ.get("FEATURES_JOBS", "1"))))

    def _pd(a, b, scorer, scale=1.0):
        return rf_process.cpdist(a, b, scorer=scorer, workers=workers,
                                 dtype=np.float64) / scale

    f_name_jw_core = _pd(s_core, c_core, JaroWinkler.normalized_similarity)
    f_name_jw_exp = _pd(s_exp, c_exp, JaroWinkler.normalized_similarity)
    f_name_lev = _pd(s_core, c_core, Levenshtein.normalized_similarity)
    f_name_sort = _pd(s_core, c_core, rf_fuzz.token_sort_ratio, 100.0)
    f_name_set = _pd(s_core, c_core, rf_fuzz.token_set_ratio, 100.0)
    f_name_partial = _pd(s_core, c_core, rf_fuzz.partial_ratio, 100.0)
    f_addr_jw = _pd(s_addr, c_addr, JaroWinkler.normalized_similarity)
    f_addr_set = _pd(s_addr, c_addr, rf_fuzz.token_set_ratio, 100.0)

    # Transliteration-robust name similarity on phonetic skeletons
    _pk: dict = {}

    def _pk_of(x):
        r = _pk.get(x)
        if r is None:
            r = _pk[x] = _phon(x)
        return r

    s_ph = [_pk_of(x) for x in s_core]
    c_ph = [_pk_of(x) for x in c_core]
    del _pk
    f_name_phon_jw = _pd(s_ph, c_ph, JaroWinkler.normalized_similarity)
    f_name_phon_set = _pd(s_ph, c_ph, rf_fuzz.token_set_ratio, 100.0)
    f_name_phon_eq = _eq_nonempty_list(s_ph, c_ph)

    # ---- Set-based features: tight loop with memoised tokenisation ----
    _c3: dict = {}
    _tk: dict = {}

    def c3(x):
        r = _c3.get(x)
        if r is None:
            r = _c3[x] = _char3(x)
        return r

    def tk(x):
        r = _tk.get(x)
        if r is None:
            r = _tk[x] = set(x.split())
        return r

    def ini(x):
        return "".join(t[0] for t in x.split())

    f_name_char3 = np.empty(n); f_name_word_jac = np.empty(n)
    f_name_idf_jac = np.empty(n); f_acronym_match = np.empty(n, dtype=np.int8)
    f_legal_state = np.empty(n, dtype=np.int8); f_nums_jac = np.empty(n)
    f_nums_shared = np.empty(n); f_street_idf_jac = np.empty(n)
    f_loc_overlap = np.empty(n); f_landmark_overlap = np.empty(n)
    f_s1_idf_sum = np.empty(n); f_name_freq = np.empty(n)
    for i in range(n):
        sc, cc = s_core[i], c_core[i]
        A3, B3 = c3(sc), c3(cc)
        f_name_char3[i] = (1.0 if not A3 and not B3 else
                           len(A3 & B3) / len(A3 | B3))
        st, ct = tk(sc), tk(cc)
        f_name_word_jac[i] = len(st & ct) / max(len(st | ct), 1)
        f_name_idf_jac[i] = _idf_jaccard(sc, cc, idf)
        sa, ca = s_acr[i], c_acr[i]
        f_acronym_match[i] = bool((sa and sa == ca) or (sa and sa == ini(cc))
                                  or (ca and ca == ini(sc)))
        sl, cl = s_leg[i], c_leg[i]
        f_legal_state[i] = (0 if not sl and not cl else 1 if sl == cl
                            else 2 if not sl or not cl else 3)
        sn, cn = tk(s_nums[i]), tk(c_nums[i])
        inter = len(sn & cn)
        f_nums_jac[i] = inter / max(len(sn | cn), 1)
        f_nums_shared[i] = inter
        f_street_idf_jac[i] = _idf_jaccard(s_street[i], c_street[i], idf)
        sL, cL = tk(s_loc[i]), tk(c_loc[i])
        f_loc_overlap[i] = len(sL & cL) / max(len(sL | cL), 1)
        f_landmark_overlap[i] = len(tk(s_lm[i]) & tk(c_lm[i]))
        f_s1_idf_sum[i] = _idf_sum(sc.split(), idf)
        f_name_freq[i] = name_freq.get(cc, 0)
    del _c3, _tk

    def _eq_nonempty(a, b):
        a = np.asarray(a, dtype=object); b = np.asarray(b, dtype=object)
        return ((a == b) & (a != "")).astype(np.int8)

    def _raw_eq_nonempty(col):
        # original: bool(s[col]) and s[col] == c[col]   (NaN is never equal)
        a = pd.Series(_s(col)); b = pd.Series(_c(col))
        return (a.notna() & (a != "") & (a == b)).to_numpy().astype(np.int8)

    s_rom, c_rom = _num(_s("romanization_ok")), _num(_c("romanization_ok"))
    s_scr, c_scr = pd.Series(_s("name_script")), pd.Series(_c("name_script"))
    df = pd.DataFrame({
        "s1_id": cand["s1_id"].to_numpy(),
        "cand_id": cand["cand_id"].to_numpy(),
        "cand_source": cand["cand_source"].to_numpy(),

        # ---- Name features ----
        "f_name_jw_core": f_name_jw_core,
        "f_name_jw_exp": f_name_jw_exp,
        "f_name_lev": f_name_lev,
        "f_name_sort": f_name_sort,
        "f_name_set": f_name_set,
        "f_name_partial": f_name_partial,
        "f_name_char3": f_name_char3,
        "f_name_word_jac": f_name_word_jac,
        "f_name_idf_jac": f_name_idf_jac,
        "f_name_core_eq": _eq_nonempty(s_core, c_core),
        "f_name_phon_jw": f_name_phon_jw,
        "f_name_phon_set": f_name_phon_set,
        "f_name_phon_eq": f_name_phon_eq,
        "f_name_sorted_eq": _raw_eq_nonempty("name_sorted"),
        "f_acronym_match": f_acronym_match,
        "f_legal_state": f_legal_state,
        "f_domain_match": _eq_nonempty(s_dom, c_dom),
        "f_expansion_changed_s": (_num(_s("expansion_changed_name")) != 0).astype(np.int8),
        "f_expansion_changed_c": (_num(_c("expansion_changed_name")) != 0).astype(np.int8),
        "f_romanized_used": (~((s_rom != 0) & (c_rom != 0))
                             | (s_scr != "latin").to_numpy()
                             | (c_scr != "latin").to_numpy()).astype(np.int8),
        "f_script_same": (s_scr.notna() & (s_scr == c_scr)).to_numpy().astype(np.int8),

        # ---- Address features ----
        "f_postal_eq": _raw_eq_nonempty("postal_code"),
        "f_postal_prefix_eq": _raw_eq_nonempty("postal_prefix3"),
        "f_house_eq": _raw_eq_nonempty("house_number"),
        "f_nums_jac": f_nums_jac,
        "f_nums_shared": f_nums_shared,
        "f_street_idf_jac": f_street_idf_jac,
        "f_loc_overlap": f_loc_overlap,
        "f_addr_jw": f_addr_jw,
        "f_addr_set": f_addr_set,
        "f_landmark_overlap": f_landmark_overlap,
        "f_addr_missing_s": _num(_s("address_missing")),
        "f_addr_missing_c": _num(_c("address_missing")),
        "f_postal_missing_s": _num(_s("postal_missing")),
        "f_postal_missing_c": _num(_c("postal_missing")),
        "f_house_missing_s": _num(_s("house_number_missing")),
        "f_house_missing_c": _num(_c("house_number_missing")),

        # ---- Model scores ----
        "f_dense_score": cand["dense_score"].fillna(-1.0).to_numpy(),
        "f_dense_cos": _dense_cos(split, country, cand["s1_id"].to_numpy(),
                                  cand["cand_id"].to_numpy(), src_arr),
        "f_stage_a": cand["stage_a_score"].astype(float).to_numpy(),
        "f_rerank": cand["rerank_score"].fillna(-1.0).to_numpy(),

        # ---- Blocking provenance ----
        "f_dense_rank": cand["dense_rank"].fillna(99.0).to_numpy(),
        "f_name_tfidf_rank": cand["name_tfidf_rank"].fillna(99.0).to_numpy(),
        "f_addr_tfidf_rank": cand["addr_tfidf_rank"].fillna(99.0).to_numpy(),
        "f_n_arms_hit": cand["n_arms_hit"].astype(int).to_numpy(),
        "f_cand_source_s2": (src_arr == "S2").astype(np.int8),

        # ---- Chain / frequency ----
        "f_name_freq": f_name_freq,
        "f_name_freq_log": np.log1p(f_name_freq),
        "f_s1_idf_sum": f_s1_idf_sum,
    })

    # ---- Competition features (S1-side + candidate-side; §7.1) ----
    df = df.sort_values(["s1_id", "f_rerank"], ascending=[True, False])
    grp = df.groupby("s1_id", sort=False)
    df["f_rank_by_rerank"] = grp["f_rerank"].rank(method="first", ascending=False)
    df["f_rerank_max"] = grp["f_rerank"].transform("max")
    df["f_rerank_mean"] = grp["f_rerank"].transform("mean")
    df["f_rerank_2nd"] = grp["f_rerank"].transform(
        lambda x: sorted(x, reverse=True)[1] if len(x) > 1 else x.iloc[0]
    )
    df["f_rerank_gap_best"] = df["f_rerank_max"] - df["f_rerank"]
    df["f_rerank_best_2nd_gap"] = df["f_rerank_max"] - df["f_rerank_2nd"]
    df["f_rerank_above_05"] = grp["f_rerank"].transform(lambda x: (x > 0.5).sum())

    # Same for stage-a
    df["f_rank_by_stagea"] = grp["f_stage_a"].rank(method="first", ascending=False)
    df["f_stagea_max"] = grp["f_stage_a"].transform("max")

    # Per-source (S2/S3) same statistics
    for src in ("S2", "S3"):
        mask = (df["cand_source"] == src)
        sub_grp = df[mask].groupby("s1_id", sort=False)
        rr_max = sub_grp["f_rerank"].transform("max").reindex(df.index)
        df[f"f_rerank_max_{src}"] = rr_max.fillna(-1.0)
        df[f"f_rerank_mean_{src}"] = sub_grp["f_rerank"].transform("mean").reindex(df.index).fillna(-1.0)

    # ---- Candidate-side competition features (crucial for 1-to-1) ----
    cand_grp = df.groupby("cand_id", sort=False)
    df["f_cand_competitors"] = cand_grp["s1_id"].transform("nunique")
    df["f_cand_best_rerank_for_others"] = cand_grp["f_rerank"].transform("max")
    df["f_cand_gap_to_best"] = df["f_cand_best_rerank_for_others"] - df["f_rerank"]
    df["f_cand_rank_among_competitors"] = cand_grp["f_rerank"].rank(
        method="first", ascending=False
    )

    if C.ENABLE_SUPPORT_FEATURES:
        # For each candidate, best rerank among the *other-source* candidates of
        # the same S1 that agree in name/address; approximated by max rerank
        # among opposite-source candidates for the same S1.
        df["f_cross_src_best"] = np.where(df["cand_source"] == "S2",
                                          df["f_rerank_max_S3"],
                                          df["f_rerank_max_S2"])

    keys = df[["s1_id", "cand_id", "cand_source"]].copy()
    X = df.drop(columns=["s1_id", "cand_id", "cand_source"]).astype("float32")

    y = None
    if include_labels and split == "train":
        gt = explode_ground_truth(read_ground_truth(C.TRAIN_GT))
        true_pairs = set(zip(gt["source1_entity_id"], gt["matched_id"]))
        y = np.array([(s, c) in true_pairs for s, c in
                      zip(keys["s1_id"], keys["cand_id"])], dtype=np.int8)

    return X, keys, y


def build_and_save(split: str, country: str) -> Path | None:
    X, keys, y = build(split, country, include_labels=(split == "train"))
    if X.empty:
        return None
    out = C.FINAL_SCORES_DIR / f"{split}__{country}__features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([keys.reset_index(drop=True),
                          X.reset_index(drop=True)], axis=1)
    if y is not None:
        combined["y"] = y
    combined.to_parquet(out, index=False)
    print(f"[features] {split}/{country}: wrote {out} ({len(combined):,} rows, "
          f"{X.shape[1]} feats)")
    return out


if __name__ == "__main__":
    countries = sorted({p.stem.split("__")[-1] for p in
                        C.PREFILTER_DIR.glob("*__*.parquet")})
    for split in ("train", "test"):
        for country in countries:
            if (C.PREFILTER_DIR / f"{split}__{country}.parquet").exists():
                build_and_save(split, country)
