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
        cand["rerank_score"] = np.nan

    s1_lookup = _prep_lookup(pd.read_parquet(
        C.NORMALIZED_DIR / f"{split}__s1__{country}.parquet"
    ))
    pools = _load_pool(split, country)
    pool_lookups = {k: _prep_lookup(v) for k, v in pools.items()}

    idf = norm_mod.load_idf(split, country)
    name_freq = _name_freq_map(split, country)

    rows: list[dict] = []
    for row in cand.itertuples(index=False):
        s = s1_lookup.loc[row.s1_id] if row.s1_id in s1_lookup.index else None
        p_src = pool_lookups.get(row.cand_source)
        if s is None or p_src is None or row.cand_id not in p_src.index:
            continue
        c = p_src.loc[row.cand_id]

        # Names
        s_core = s["core_name"] or ""
        c_core = c["core_name"] or ""
        s_exp = s["name_expanded"] or ""
        c_exp = c["name_expanded"] or ""
        s_roman = s["name_roman"] or s_core
        c_roman = c["name_roman"] or c_core
        s_toks = s_core.split()
        c_toks = c_core.split()

        # Addresses
        s_addr = s["address_expanded"] or ""
        c_addr = c["address_expanded"] or ""

        # Legal suffix agreement
        s_leg = s["legal_suffix"] or ""
        c_leg = c["legal_suffix"] or ""
        if not s_leg and not c_leg:
            leg = 0        # both missing
        elif s_leg == c_leg:
            leg = 1        # same
        elif not s_leg or not c_leg:
            leg = 2        # one missing
        else:
            leg = 3        # different

        rows.append({
            "s1_id": row.s1_id,
            "cand_id": row.cand_id,
            "cand_source": row.cand_source,

            # ---- Name features ----
            "f_name_jw_core": JaroWinkler.normalized_similarity(s_core, c_core),
            "f_name_jw_exp": JaroWinkler.normalized_similarity(s_exp, c_exp),
            "f_name_lev": 1.0 - Levenshtein.normalized_distance(s_core, c_core),
            "f_name_sort": rf_fuzz.token_sort_ratio(s_core, c_core) / 100.0,
            "f_name_set": rf_fuzz.token_set_ratio(s_core, c_core) / 100.0,
            "f_name_partial": rf_fuzz.partial_ratio(s_core, c_core) / 100.0,
            "f_name_char3": _char3_jaccard(s_core, c_core),
            "f_name_word_jac": len(set(s_toks) & set(c_toks)) /
                              max(len(set(s_toks) | set(c_toks)), 1),
            "f_name_idf_jac": _idf_jaccard(s_core, c_core, idf),
            "f_name_core_eq": int(s_core == c_core and s_core != ""),
            "f_name_sorted_eq": int(s["name_sorted"] == c["name_sorted"]
                                    and s["name_sorted"] != ""),
            "f_acronym_match": int(
                (s["acronym"] and s["acronym"] == c["acronym"])
                or (s["acronym"] and s["acronym"] == "".join(t[0] for t in c_toks))
                or (c["acronym"] and c["acronym"] == "".join(t[0] for t in s_toks))
            ),
            "f_legal_state": leg,
            "f_domain_match": int(
                bool(s["name_domain"]) and s["name_domain"] == c["name_domain"]
            ),
            "f_expansion_changed_s": int(s["expansion_changed_name"]),
            "f_expansion_changed_c": int(c["expansion_changed_name"]),
            "f_romanized_used": int(not (s["romanization_ok"] and c["romanization_ok"])
                                     or s["name_script"] != "latin"
                                     or c["name_script"] != "latin"),
            "f_script_same": int(s["name_script"] == c["name_script"]),

            # ---- Address features ----
            "f_postal_eq": int(bool(s["postal_code"])
                                and s["postal_code"] == c["postal_code"]),
            "f_postal_prefix_eq": int(bool(s["postal_prefix3"])
                                       and s["postal_prefix3"] == c["postal_prefix3"]),
            "f_house_eq": int(bool(s["house_number"])
                               and s["house_number"] == c["house_number"]),
            "f_nums_jac": len(set((s["all_numbers"] or "").split())
                               & set((c["all_numbers"] or "").split())) /
                           max(len(set((s["all_numbers"] or "").split())
                                    | set((c["all_numbers"] or "").split())), 1),
            "f_nums_shared": len(set((s["all_numbers"] or "").split())
                                 & set((c["all_numbers"] or "").split())),
            "f_street_idf_jac": _idf_jaccard(s["street_tokens"] or "",
                                              c["street_tokens"] or "", idf),
            "f_loc_overlap": len(set((s["locality_tokens"] or "").split())
                                  & set((c["locality_tokens"] or "").split())) /
                              max(len(set((s["locality_tokens"] or "").split())
                                       | set((c["locality_tokens"] or "").split())), 1),
            "f_addr_jw": JaroWinkler.normalized_similarity(s_addr, c_addr),
            "f_addr_set": rf_fuzz.token_set_ratio(s_addr, c_addr) / 100.0,
            "f_landmark_overlap": len(set((s["landmark_text"] or "").split())
                                       & set((c["landmark_text"] or "").split())),
            "f_addr_missing_s": int(s["address_missing"]),
            "f_addr_missing_c": int(c["address_missing"]),
            "f_postal_missing_s": int(s["postal_missing"]),
            "f_postal_missing_c": int(c["postal_missing"]),
            "f_house_missing_s": int(s["house_number_missing"]),
            "f_house_missing_c": int(c["house_number_missing"]),

            # ---- Model scores ----
            "f_dense_score": row.dense_score if not pd.isna(row.dense_score) else -1.0,
            "f_stage_a": float(row.stage_a_score),
            "f_rerank": row.rerank_score if not pd.isna(row.rerank_score) else -1.0,

            # ---- Blocking provenance ----
            "f_dense_rank": row.dense_rank if not pd.isna(row.dense_rank) else 99.0,
            "f_name_tfidf_rank": row.name_tfidf_rank if not pd.isna(row.name_tfidf_rank) else 99.0,
            "f_addr_tfidf_rank": row.addr_tfidf_rank if not pd.isna(row.addr_tfidf_rank) else 99.0,
            "f_n_arms_hit": int(row.n_arms_hit),
            "f_cand_source_s2": int(row.cand_source == "S2"),

            # ---- Chain / frequency ----
            "f_name_freq": name_freq.get(c_core, 0),
            "f_name_freq_log": float(np.log1p(name_freq.get(c_core, 0))),
            "f_s1_idf_sum": _idf_sum(s_toks, idf),
        })

    if not rows:
        return pd.DataFrame(), pd.DataFrame(), None

    df = pd.DataFrame(rows)

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
        other = df["cand_source"].map({"S2": "S3", "S3": "S2"})
        df["_pair"] = list(zip(df["s1_id"], other))
        best_other = df.groupby(["s1_id", "cand_source"])["f_rerank"].transform("max")
        df["f_cross_src_best"] = best_other
        df.drop(columns="_pair", inplace=True)

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
