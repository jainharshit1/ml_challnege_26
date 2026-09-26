"""Stage-A pre-filter (plan §5).

Cheap features + small LightGBM narrow blocking candidates to top-N per S1
BEFORE the cross-encoder. The kept pairs on the *test* partition become the
authoritative `output/candidate_pairs.tsv`.

Training uses group G with labels from train GT; scoring runs on G, V and
test.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import splits as splits_mod
from .io_utils import (
    explode_ground_truth,
    read_ground_truth,
    read_source_tsv,
    write_id_list_tsv,
    write_parquet,
)


# ---------------------------------------------------------------------------
# Feature engineering (§5 step 1) — vectorised via rapidfuzz.
# ---------------------------------------------------------------------------
def _rf():
    try:
        import rapidfuzz  # noqa: F401
        return rapidfuzz
    except Exception as e:  # pragma: no cover
        raise RuntimeError("rapidfuzz is required for Stage-A features") from e


def _jaccard(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    A = set(a.split())
    B = set(b.split())
    if not A and not B:
        return 1.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0


def _char3(s: str) -> set[str]:
    return set(s[i : i + 3] for i in range(len(s) - 2)) if len(s) >= 3 else set()


def _char3_jaccard(a: str, b: str) -> float:
    A, B = _char3(a), _char3(b)
    if not A and not B:
        return 1.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0


def _pairwise_feats(cands: pd.DataFrame,
                    s1_map: pd.DataFrame,
                    pool_maps: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute Stage-A features on a candidate table (already partitioned)."""
    rf = _rf()
    from rapidfuzz import fuzz as rf_fuzz  # type: ignore
    from rapidfuzz.distance import JaroWinkler  # type: ignore

    cols_needed = ["entity_id", "core_name", "name_roman", "address_expanded",
                   "postal_code", "house_number", "all_numbers",
                   "locality_tokens", "core_name"]

    s1_lookup = s1_map.set_index("entity_id")
    pool_lookup = {k: v.set_index("entity_id") for k, v in pool_maps.items()}

    def _row_feats(s1_id: str, cand_id: str, cand_source: str) -> dict:
        s = s1_lookup.loc[s1_id]
        p_src = pool_lookup[cand_source]
        if cand_id not in p_src.index:
            return {}
        c = p_src.loc[cand_id]

        s_name = s["core_name"] or s["name_roman"] or ""
        c_name = c["core_name"] or c["name_roman"] or ""
        s_addr = s["address_expanded"] or ""
        c_addr = c["address_expanded"] or ""

        return {
            "f_name_jw": JaroWinkler.normalized_similarity(s_name, c_name),
            "f_name_sort": rf_fuzz.token_sort_ratio(s_name, c_name) / 100.0,
            "f_name_set": rf_fuzz.token_set_ratio(s_name, c_name) / 100.0,
            "f_name_char3": _char3_jaccard(s_name, c_name),
            "f_addr_set": rf_fuzz.token_set_ratio(s_addr, c_addr) / 100.0,
            "f_nums_jac": _jaccard(s["all_numbers"] or "", c["all_numbers"] or ""),
            "f_house_eq": int(bool(s["house_number"])
                              and s["house_number"] == c["house_number"]),
            "f_postal_eq": int(bool(s["postal_code"])
                                and s["postal_code"] == c["postal_code"]),
            "f_loc_overlap": _jaccard(s["locality_tokens"] or "",
                                        c["locality_tokens"] or ""),
        }

    feats = [_row_feats(a, b, s) for a, b, s in
             zip(cands["s1_id"], cands["cand_id"], cands["cand_source"])]
    fdf = pd.DataFrame(feats).reindex(cands.index)
    return fdf


def _cheap_features(cands: pd.DataFrame, split: str, country: str
                    ) -> pd.DataFrame:
    """Load the country's normalized parquet for S1/S2/S3 and add features."""
    s1 = pd.read_parquet(C.NORMALIZED_DIR / f"{split}__s1__{country}.parquet")
    pools = {}
    for src in ("S2", "S3"):
        p = C.NORMALIZED_DIR / f"{split}__{src.lower()}__{country}.parquet"
        if p.exists():
            pools[src] = pd.read_parquet(p)
    feats = _pairwise_feats(cands.reset_index(drop=True), s1, pools)
    # Combine with the blocking-arm scores/ranks already on cands.
    for col in ("dense_score", "name_tfidf_score", "addr_tfidf_score",
                "dense_rank", "name_tfidf_rank", "addr_tfidf_rank",
                "n_arms_hit", "key_locrare_hit", "key_house_hit", "key_acr_hit"):
        if col in cands.columns:
            feats[col] = cands[col].values
    return feats.astype("float32")


# ---------------------------------------------------------------------------
# Model training / scoring
# ---------------------------------------------------------------------------
def _label_from_gt(cands: pd.DataFrame) -> np.ndarray:
    gt = explode_ground_truth(read_ground_truth(C.TRAIN_GT))
    true_pairs = set(zip(gt["source1_entity_id"], gt["matched_id"]))
    y = np.array([(s, c) in true_pairs for s, c in
                  zip(cands["s1_id"], cands["cand_id"])], dtype=np.int8)
    return y


def train(model_path: Path | None = None):
    import lightgbm as lgb
    C.ensure_dirs()

    split_df = splits_mod.load()
    g_ids = set(split_df.loc[split_df["group"] == "G", "entity_id"])
    parts: list[pd.DataFrame] = []
    for p in C.BLOCKING_DIR.glob("train__*.parquet"):
        country = p.stem.split("__")[-1]
        cand = pd.read_parquet(p)
        cand = cand[cand["s1_id"].isin(g_ids)].copy()
        if cand.empty:
            continue
        cand["_country"] = country
        parts.append(cand)
    if not parts:
        raise RuntimeError("No G-partition blocking candidates found.")
    cands = pd.concat(parts, ignore_index=True)
    print(f"[prefilter] G training pairs: {len(cands):,}")

    X_parts = []
    for country, sub in cands.groupby("_country"):
        X_parts.append(_cheap_features(sub, "train", country))
    X = pd.concat(X_parts, ignore_index=True)
    y = _label_from_gt(cands)
    feat_names = list(X.columns)

    device = C.lgb_device()
    print(f"[prefilter] LightGBM device: {device}")
    model = lgb.LGBMClassifier(
        n_estimators=C.STAGE_A_N_ESTIMATORS,
        num_leaves=C.STAGE_A_NUM_LEAVES,
        learning_rate=C.STAGE_A_LR,
        n_jobs=-1,
        random_state=C.SEED,
        objective="binary",
        device_type=device,
    )
    model.fit(X, y)
    if model_path is None:
        model_path = C.MODELS_DIR / "stage_a_lgbm.txt"
    model.booster_.save_model(str(model_path))
    (C.MODELS_DIR / "stage_a_features.txt").write_text(
        "\n".join(feat_names), encoding="utf-8"
    )
    print(f"[prefilter] wrote {model_path}")
    return model


def _load_model():
    import lightgbm as lgb
    model_path = C.MODELS_DIR / "stage_a_lgbm.txt"
    return lgb.Booster(model_file=str(model_path))


def score_partition(split: str, country: str, top_n: int = C.PREFILTER_TOP_N
                    ) -> pd.DataFrame:
    blk = C.BLOCKING_DIR / f"{split}__{country}.parquet"
    if not blk.exists():
        print(f"[prefilter] no blocking file for {split}/{country}")
        return pd.DataFrame()
    cands = pd.read_parquet(blk)
    if cands.empty:
        return cands

    X = _cheap_features(cands, split, country)
    booster = _load_model()
    feat_names = (C.MODELS_DIR / "stage_a_features.txt").read_text(encoding="utf-8").split()
    X = X.reindex(columns=feat_names).fillna(-1.0)
    cands["stage_a_score"] = booster.predict(X)
    # top-N per S1
    cands = cands.sort_values(["s1_id", "stage_a_score"], ascending=[True, False])
    kept = cands.groupby("s1_id", as_index=False).head(top_n)
    out = C.PREFILTER_DIR / f"{split}__{country}.parquet"
    write_parquet(kept, out)
    print(f"[prefilter] {split}/{country}: kept {len(kept):,}/{len(cands):,} "
          f"({kept['s1_id'].nunique():,} S1)")
    return kept


def score_all() -> None:
    countries = sorted({p.stem.split("__")[-1] for p in
                        C.BLOCKING_DIR.glob("*__*.parquet")})
    for split in ("train", "test"):
        for country in countries:
            if (C.BLOCKING_DIR / f"{split}__{country}.parquet").exists():
                score_partition(split, country)


# ---------------------------------------------------------------------------
# candidate_pairs.tsv writer — MUST cover every test S1 (empty if none).
# ---------------------------------------------------------------------------
def write_candidate_pairs_tsv() -> Path:
    """Aggregate every test partition into a single output/candidate_pairs.tsv.

    Every test S1 gets a row (empty string if the pipeline produced none).
    """
    test_s1 = read_source_tsv(C.TEST_SOURCE["s1"])
    if C.NORMALIZE_ROW_CAP:
        test_s1 = test_s1.iloc[:C.NORMALIZE_ROW_CAP].copy()
    all_s1 = test_s1["entity_id"].tolist()
    by_s1: dict[str, list[str]] = {sid: [] for sid in all_s1}

    for p in C.PREFILTER_DIR.glob("test__*.parquet"):
        cand = pd.read_parquet(p, columns=["s1_id", "cand_id", "stage_a_score"])
        cand = cand.sort_values(["s1_id", "stage_a_score"],
                                ascending=[True, False])
        for sid, cid in zip(cand["s1_id"], cand["cand_id"]):
            if sid in by_s1:
                by_s1[sid].append(cid)

    out = C.OUTPUT_DIR / "candidate_pairs.tsv"
    write_id_list_tsv(out, all_s1, [by_s1[s] for s in all_s1],
                      id_column="candidate_entity_ids")
    print(f"[prefilter] wrote {out} ({sum(1 for v in by_s1.values() if v):,} "
          f"non-empty / {len(by_s1):,} total)")
    return out


# ---------------------------------------------------------------------------
# Prefilter recall check on V (plan §5 step 4)
# ---------------------------------------------------------------------------
def prefilter_recall_report() -> None:
    split_df = splits_mod.load()
    v_ids = set(split_df.loc[split_df["group"] == "V", "entity_id"])
    gt = explode_ground_truth(read_ground_truth(C.TRAIN_GT))
    gt_v = gt[gt["source1_entity_id"].isin(v_ids)]
    gt_pairs = set(zip(gt_v["source1_entity_id"], gt_v["matched_id"]))

    kept_pairs = set()
    all_block_pairs = set()
    for country_path in C.PREFILTER_DIR.glob("train__*.parquet"):
        cand = pd.read_parquet(country_path, columns=["s1_id", "cand_id"])
        cand = cand[cand["s1_id"].isin(v_ids)]
        for s, c in zip(cand["s1_id"], cand["cand_id"]):
            kept_pairs.add((s, c))
    for country_path in C.BLOCKING_DIR.glob("train__*.parquet"):
        blk = pd.read_parquet(country_path, columns=["s1_id", "cand_id"])
        blk = blk[blk["s1_id"].isin(v_ids)]
        for s, c in zip(blk["s1_id"], blk["cand_id"]):
            all_block_pairs.add((s, c))

    r_block = len(gt_pairs & all_block_pairs) / max(len(gt_pairs), 1)
    r_kept = len(gt_pairs & kept_pairs) / max(len(gt_pairs), 1)
    loss = r_block - r_kept
    msg = (f"blocking_recall_V={r_block:.4f}  prefilter_recall_V={r_kept:.4f}  "
           f"loss={loss:.4f} (target < 0.005)")
    print(f"[prefilter] {msg}")
    (C.REPORTS_DIR / "prefilter_recall.txt").write_text(msg, encoding="utf-8")


if __name__ == "__main__":
    train()
    score_all()
    prefilter_recall_report()
    write_candidate_pairs_tsv()
