"""Stage-A pre-filter (plan §5).

Cheap features + small LightGBM narrow blocking candidates to top-N per S1
BEFORE the cross-encoder. The kept pairs on the *test* partition become the
authoritative `output/candidate_pairs.tsv`.

Training uses group G with labels from train GT; scoring runs on G, V and
test.
"""
from __future__ import annotations
import os
import sys
import threading
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


HF_REPO = "jainsaabb/ml_challenge_2026_artifacts"


def _upload_partition_async(local_path: Path, remote_prefix: str) -> None:
    """Fire-and-forget upload — same pattern as blocking.py."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        return

    def _do():
        try:
            os.environ["HF_HUB_OFFLINE"] = "0"
            from huggingface_hub import upload_file
            upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=f"{remote_prefix}/{local_path.name}",
                repo_id=HF_REPO,
                repo_type="dataset",
                commit_message=f"partial: {remote_prefix}/{local_path.name}",
                token=token,
            )
            print(f"[prefilter]   HF upload OK: {remote_prefix}/{local_path.name}",
                  flush=True)
        except Exception as e:
            print(f"[prefilter]   HF upload FAILED "
                  f"({type(e).__name__}: {e})", flush=True)

    threading.Thread(target=_do, daemon=True,
                     name=f"hfupload-{local_path.name}").start()


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
    """Vectorized Stage-A features.

    Uses pd.Series.reindex (C-level hash lookup) to bulk-extract each column
    once, then tight rapidfuzz loops for string sims. ~200x faster than the
    per-row `.loc[]` version.
    """
    _rf()
    from rapidfuzz import fuzz as rf_fuzz  # type: ignore
    from rapidfuzz.distance import JaroWinkler  # type: ignore

    n = len(cands)
    if n == 0:
        return pd.DataFrame()

    s1_lookup = s1_map.drop_duplicates("entity_id").set_index("entity_id")
    pool_lookup = {k: v.drop_duplicates("entity_id").set_index("entity_id")
                   for k, v in pool_maps.items()}

    s1_ids_arr = cands["s1_id"].to_numpy()
    cand_ids_arr = cands["cand_id"].to_numpy()
    cand_src_arr = cands["cand_source"].to_numpy()

    def _s(col: str) -> np.ndarray:
        return s1_lookup[col].reindex(s1_ids_arr).values

    def _c(col: str) -> np.ndarray:
        result = np.empty(n, dtype=object)
        result[:] = ""
        for src, pool_df in pool_lookup.items():
            mask = (cand_src_arr == src)
            if not mask.any():
                continue
            result[mask] = pool_df[col].reindex(cand_ids_arr[mask]).values
        return result

    def _str(a: np.ndarray) -> np.ndarray:
        return np.where(pd.isna(a), "", a).astype(object)

    print(f"[prefilter]   extracting columns for {n:,} pairs…", flush=True)
    s_name_core = _str(_s("core_name"))
    c_name_core = _str(_c("core_name"))
    s_name_roman = _str(_s("name_roman"))
    c_name_roman = _str(_c("name_roman"))
    s_addr = _str(_s("address_expanded"))
    c_addr = _str(_c("address_expanded"))
    s_postal = _str(_s("postal_code"))
    c_postal = _str(_c("postal_code"))
    s_house = _str(_s("house_number"))
    c_house = _str(_c("house_number"))
    s_nums = _str(_s("all_numbers"))
    c_nums = _str(_c("all_numbers"))
    s_loc = _str(_s("locality_tokens"))
    c_loc = _str(_c("locality_tokens"))

    # Choose name: core preferred, fall back to roman
    s_name = np.where(s_name_core != "", s_name_core, s_name_roman)
    c_name = np.where(c_name_core != "", c_name_core, c_name_roman)
    del s_name_core, c_name_core, s_name_roman, c_name_roman

    print(f"[prefilter]   computing name sims…", flush=True)
    # rapidfuzz scores element-wise in multi-threaded C++ (cpdist).
    from rapidfuzz import process as rf_process  # type: ignore
    workers = max(1, C.N_JOBS // max(1, int(os.environ.get("PREFILTER_JOBS", "1"))))
    s_name_l, c_name_l = s_name.tolist(), c_name.tolist()

    def _pd(a, b, scorer, scale=1.0):
        return (rf_process.cpdist(a, b, scorer=scorer, workers=workers,
                                  dtype=np.float64) / scale).astype(np.float32)

    f_name_jw = _pd(s_name_l, c_name_l, JaroWinkler.normalized_similarity)
    f_name_sort = _pd(s_name_l, c_name_l, rf_fuzz.token_sort_ratio, 100.0)
    f_name_set = _pd(s_name_l, c_name_l, rf_fuzz.token_set_ratio, 100.0)
    f_addr_set = _pd(s_addr.tolist(), c_addr.tolist(), rf_fuzz.token_set_ratio, 100.0)

    # Set-based sims: tight loop, tokenisation memoised (S1 repeats ~40x).
    f_name_char3 = np.empty(n, dtype=np.float32)
    f_nums_jac = np.empty(n, dtype=np.float32)
    f_loc_overlap = np.empty(n, dtype=np.float32)
    _c3: dict = {}
    _tk: dict = {}

    def c3(x):
        r = _c3.get(x)
        if r is None:
            r = _c3[x] = _char3(x)
        return r

    def jac(a, b):
        if not a and not b:
            return 1.0
        A = _tk.get(a)
        if A is None:
            A = _tk[a] = set(a.split())
        B = _tk.get(b)
        if B is None:
            B = _tk[b] = set(b.split())
        if not A and not B:
            return 1.0
        union = len(A | B)
        return len(A & B) / union if union else 0.0

    for i in range(n):
        A3, B3 = c3(s_name_l[i]), c3(c_name_l[i])
        f_name_char3[i] = (1.0 if not A3 and not B3 else
                           len(A3 & B3) / len(A3 | B3))
        f_nums_jac[i] = jac(s_nums[i], c_nums[i])
        f_loc_overlap[i] = jac(s_loc[i], c_loc[i])
        if i and i % 2_000_000 == 0:
            print(f"[prefilter]     {i:,}/{n:,}", flush=True)
    del _c3, _tk

    # Vectorized equality features
    f_house_eq = ((s_house == c_house) & (s_house != "")).astype(np.int8)
    f_postal_eq = ((s_postal == c_postal) & (s_postal != "")).astype(np.int8)

    return pd.DataFrame({
        "f_name_jw": f_name_jw,
        "f_name_sort": f_name_sort,
        "f_name_set": f_name_set,
        "f_name_char3": f_name_char3,
        "f_addr_set": f_addr_set,
        "f_nums_jac": f_nums_jac,
        "f_house_eq": f_house_eq,
        "f_postal_eq": f_postal_eq,
        "f_loc_overlap": f_loc_overlap,
    })


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
    cands = cands.sort_values("_country", kind="stable").reset_index(drop=True)
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
          f"({kept['s1_id'].nunique():,} S1)", flush=True)
    _upload_partition_async(out, "prefilter")
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
