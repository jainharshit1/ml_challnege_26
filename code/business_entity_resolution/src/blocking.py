"""Country-partitioned hybrid blocking (plan §4).

Runs one country partition at a time; each partition unions six arms:
  A1 dense kNN on bge-m3 (FAISS IVF or Flat depending on size)
  A2 char TF-IDF top-K on `core_name`
  A3 char TF-IDF top-K on `address_expanded`
  A4 exact key: (rare_token, locality_token | postal_code)   with block cap
  A5 exact key: (house_number, street_token)                 with block cap
  A6 exact key: (acronym, postal_prefix3 | locality_token)   with block cap
  A7 phonetic (optional, off by default; §4.2)

Output: artifacts/blocking/{split}__{country}.parquet
  columns [s1_id, cand_id, cand_source,
           dense_score, name_tfidf_score, addr_tfidf_score,
           key_locrare_hit, key_house_hit, key_acr_hit,
           dense_rank, name_tfidf_rank, addr_tfidf_rank,
           n_arms_hit]

Optimised for a 32 GB RAM / 8 vCPU CPU-only box:
  * FAISS add + search in batches so peak memory ≈ index_size + one batch,
    never index_size + full-fp32-pool + full-fp32-query.
  * Fully vectorized top-k extraction (no Python row loop).
  * Sequential pool loading (S2 processed and freed before S3 is loaded).
  * Flush-forced logging every ~1-2 minutes of wall time so progress is
    visible in a redirected/tee'd log file.
"""
from __future__ import annotations
import gc
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config as C
from . import embed as embed_mod
from .io_utils import write_parquet


HF_REPO = "jainsaabb/ml_challenge_2026_artifacts"


def _upload_partition_async(local_path: Path, remote_prefix: str) -> None:
    """Fire-and-forget upload of a single parquet to HF Hub.

    Runs in a daemon thread so the main pipeline doesn't wait for network I/O.
    Silently swallows failures; per-stage bulk upload (via watcher_full.sh)
    is the safety net.
    """
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
            _log(
                f"[blocking]   HF upload OK: {remote_prefix}/{local_path.name}")
        except Exception as e:
            _log(f"[blocking]   HF upload FAILED ({type(e).__name__}: {e}) "
                 f"— watcher will retry at stage end")

    t = threading.Thread(target=_do, daemon=True,
                         name=f"hfupload-{local_path.name}")
    t.start()


ARM_COLS = [
    "s1_id", "cand_id", "cand_source",
    "dense_score", "name_tfidf_score", "addr_tfidf_score",
    "key_locrare_hit", "key_house_hit", "key_acr_hit",
    "dense_rank", "name_tfidf_rank", "addr_tfidf_rank",
]


_TAG = ""   # set per worker when partitions run in parallel


def _log(msg: str) -> None:
    """Force-flushed print so buffered stdout doesn't hide progress."""
    print(f"{_TAG}{msg}", flush=True)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Dense arm (A1) — memory-bounded FAISS
# ---------------------------------------------------------------------------
def _dense_knn(query_vecs: np.ndarray, pool_vecs: np.ndarray,
               k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores [N,k], indices [N,k]).

    Uses FAISS with batched fp32 conversion so peak memory ≈ index storage
    plus one batch (fp32) rather than the whole pool converted at once.
    """
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "faiss is required for the dense blocking arm") from e

    # Let FAISS use all available cores
    try:
        faiss.omp_set_num_threads(C.N_JOBS)
    except Exception:
        pass

    n_pool = int(pool_vecs.shape[0])
    n_q = int(query_vecs.shape[0])
    d = int(pool_vecs.shape[1])

    # ---- Build index ----
    if n_pool <= C.FAISS_FLAT_MAX:
        index = faiss.IndexFlatIP(d)
        idx_kind = "Flat"
    else:
        quantizer = faiss.IndexFlatIP(d)
        nlist = min(C.FAISS_NLIST, max(64, int(np.sqrt(n_pool) * 4)))
        if C.FAISS_SQ_FP16:
            # Vectors are stored as fp16 on disk, so fp16 codes are lossless
            # and halve index memory / scan bandwidth vs IVFFlat (fp32).
            index = faiss.IndexIVFScalarQuantizer(
                quantizer, d, nlist, faiss.ScalarQuantizer.QT_fp16,
                faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexIVFFlat(quantizer, d, nlist,
                                       faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(C.SEED)
        train_n = min(C.FAISS_TRAIN_SAMPLE, n_pool)
        train_idx = rng.choice(n_pool, size=train_n, replace=False)
        # Copy just the training sample to fp32 (much smaller than full pool)
        train_sample = np.ascontiguousarray(
            pool_vecs[train_idx], dtype=np.float32)
        _log(
            f"[blocking]   FAISS IVF train: nlist={nlist}  sample={train_n:,}")
        index.train(train_sample)
        del train_sample
        gc.collect()
        index.nprobe = C.FAISS_NPROBE_DEFAULT
        idx_kind = (f"IVF{'SQfp16' if C.FAISS_SQ_FP16 else 'Flat'}"
                    f"(nlist={nlist}, nprobe={index.nprobe})")

    _log(
        f"[blocking]   FAISS index kind: {idx_kind}  d={d}  n_pool={n_pool:,}")

    # ---- Batched add ----
    ADD_BATCH = 200_000
    added = 0
    for start in range(0, n_pool, ADD_BATCH):
        end = min(start + ADD_BATCH, n_pool)
        chunk = np.ascontiguousarray(pool_vecs[start:end], dtype=np.float32)
        index.add(chunk)
        added += end - start
        del chunk
        if (start // ADD_BATCH) % 5 == 0 or end == n_pool:
            _log(f"[blocking]   FAISS add: {added:,}/{n_pool:,}")
    gc.collect()

    # ---- Batched search ----
    D_all = np.zeros((n_q, k), dtype=np.float32)
    I_all = np.full((n_q, k), -1, dtype=np.int64)
    Q_BATCH = 100_000
    searched = 0
    for start in range(0, n_q, Q_BATCH):
        end = min(start + Q_BATCH, n_q)
        chunk = np.ascontiguousarray(query_vecs[start:end], dtype=np.float32)
        D_all[start:end], I_all[start:end] = index.search(chunk, k)
        searched += end - start
        del chunk
        if (start // Q_BATCH) % 3 == 0 or end == n_q:
            _log(f"[blocking]   FAISS search: {searched:,}/{n_q:,}")

    # Explicit cleanup — reset the index to release its internal buffers
    try:
        index.reset()
    except Exception:
        pass
    del index
    gc.collect()
    return D_all, I_all


# ---------------------------------------------------------------------------
# Sparse TF-IDF arms (A2 / A3)
# ---------------------------------------------------------------------------
def _tfidf_mats(texts_p: list[str], texts_q: list[str], analyzer: str,
                ngram_range: tuple[int, int], max_df: float
                ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Fit on pool+query and return (mat_p, mat_q) from ONE analyzer pass
    (fit_transform + row slice instead of fit, transform, transform)."""
    mat = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=C.TFIDF_MIN_DF,
        max_df=max_df,
        max_features=C.TFIDF_MAX_FEATURES,
        norm="l2",
        sublinear_tf=True,
        dtype=np.float32,
    ).fit_transform(texts_p + texts_q).tocsr()
    n_p = len(texts_p)
    return mat[:n_p], mat[n_p:]


def _fill_topk(out: sp.csr_matrix, D: np.ndarray, I: np.ndarray,
               offset: int) -> None:
    """Scatter a sorted top-k CSR block into D/I rows [offset, offset+n)."""
    indptr = out.indptr
    row_lens = np.diff(indptr)                       # entries per row
    total = int(row_lens.sum())
    if total == 0:
        return
    row_ids = offset + np.repeat(np.arange(len(row_lens), dtype=np.int64),
                                 row_lens)
    # Position within row: (position in flat data) - (start of that row)
    starts = indptr[:-1].astype(np.int64)
    positions = np.arange(total, dtype=np.int64) - starts.repeat(row_lens)
    D[row_ids, positions] = out.data
    I[row_ids, positions] = out.indices


def _sparse_topk(mat_q: sp.csr_matrix, mat_p: sp.csr_matrix,
                 k: int, row_batch: int = C.TFIDF_ROW_BATCH
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Cosine top-k of q rows against p rows (both L2-normalised).

    Uses sparse_dot_topn when available, in query-row batches so progress
    and ETA are logged. Only a missing library triggers the dense fallback
    (the fallback densifies row_batch x n_pool and would OOM on big pools).
    """
    n_q = mat_q.shape[0]
    D = np.zeros((n_q, k), dtype=np.float32)
    I = np.full((n_q, k), -1, dtype=np.int64)
    try:
        from sparse_dot_topn import sp_matmul_topn  # type: ignore
    except Exception as e:
        sp_matmul_topn = None
        _log(f"[blocking]   sparse_dot_topn unavailable ({type(e).__name__}); "
             "falling back to chunked matmul")

    if sp_matmul_topn is not None:
        pT = mat_p.T.tocsr()
        _log(f"[blocking]     nnz q={mat_q.nnz:,} p={mat_p.nnz:,}")
        t0 = time.time()
        for start in range(0, n_q, C.TFIDF_TOPN_BATCH):
            end = min(start + C.TFIDF_TOPN_BATCH, n_q)
            out = sp_matmul_topn(mat_q[start:end], pT, top_n=k, threshold=0.20,
                                 n_threads=C.N_JOBS, sort=True).tocsr()
            _fill_topk(out, D, I, start)
            el = time.time() - t0
            _log(f"[blocking]     top-k {end:,}/{n_q:,}  "
                 f"{el / 60:.1f} min, ETA {el / end * (n_q - end) / 60:.1f} min")
        return D, I
    else:
        pT = mat_p.T
        for start in range(0, n_q, row_batch):
            end = min(start + row_batch, n_q)
            sim = (mat_q[start:end] @ pT).toarray()
            top_idx = np.argpartition(-sim, kth=min(k, sim.shape[1] - 1),
                                      axis=1)[:, :k]
            for r_local in range(sim.shape[0]):
                cols = top_idx[r_local]
                vals = sim[r_local, cols]
                order = np.argsort(-vals)
                D[start + r_local] = vals[order]
                I[start + r_local] = cols[order]
        return D, I


# ---------------------------------------------------------------------------
# Key arms (A4 / A5 / A6)
# ---------------------------------------------------------------------------
def _split_toks(x: str | None) -> list[str]:
    if not isinstance(x, str) or not x:
        return []
    return [t for t in x.split() if t]


def _key_arm(s1_df: pd.DataFrame, pool_df: pd.DataFrame,
             s1_keys_fn, pool_keys_fn,
             cap: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (query_idx_arr, pool_idx_arr, dropped_key_count) as numpy arrays."""
    pool_index: dict[str, list[int]] = defaultdict(list)
    for j, row in enumerate(pool_df.itertuples(index=False)):
        for k in pool_keys_fn(row):
            pool_index[k].append(j)
    kept = {k: v for k, v in pool_index.items() if 0 < len(v) <= cap}
    dropped = len(pool_index) - len(kept)
    del pool_index
    gc.collect()

    q_arr: list[int] = []
    p_arr: list[int] = []
    for i, row in enumerate(s1_df.itertuples(index=False)):
        seen = set()
        for k in s1_keys_fn(row):
            for j in kept.get(k, ()):
                if j not in seen:
                    seen.add(j)
                    q_arr.append(i)
                    p_arr.append(j)
    del kept
    gc.collect()
    return (np.array(q_arr, dtype=np.int64),
            np.array(p_arr, dtype=np.int64),
            dropped)


def _s1_keys_locrare(row):
    core = getattr(row, "core_name", "")
    if not isinstance(core, str) or not core:
        return []
    toks = core.split()
    if not toks:
        return []
    rare = max(toks, key=len)
    locs = set(_split_toks(getattr(row, "locality_tokens", "")))
    postal = getattr(row, "postal_code", "")
    if isinstance(postal, str) and postal:
        locs.add(postal)
    return [f"{rare}||{loc}" for loc in locs]


def _pool_keys_locrare(row):
    return _s1_keys_locrare(row)


def _s1_keys_house(row):
    hn = getattr(row, "house_number", "")
    if not isinstance(hn, str) or not hn:
        return []
    return [f"{hn}||{s}" for s in _split_toks(getattr(row, "street_tokens", ""))]


def _pool_keys_house(row):
    return _s1_keys_house(row)


def _s1_keys_acr(row):
    a = getattr(row, "acronym", "")
    if not isinstance(a, str) or len(a) < 2:
        return []
    keys = []
    pp = getattr(row, "postal_prefix3", "")
    if isinstance(pp, str) and pp:
        keys.append(f"{a}||{pp}")
    for loc in _split_toks(getattr(row, "locality_tokens", "")):
        keys.append(f"{a}||{loc}")
    return keys


def _pool_keys_acr(row):
    return _s1_keys_acr(row)


# ---------------------------------------------------------------------------
# Vectorized per-arm record builders
# ---------------------------------------------------------------------------
def _empty_arm_df(n: int) -> dict:
    """Dict of NaN/zero arrays for the non-key numeric columns."""
    return {
        "dense_score": np.full(n, np.nan, dtype=np.float32),
        "name_tfidf_score": np.full(n, np.nan, dtype=np.float32),
        "addr_tfidf_score": np.full(n, np.nan, dtype=np.float32),
        "key_locrare_hit": np.zeros(n, dtype=np.int8),
        "key_house_hit": np.zeros(n, dtype=np.int8),
        "key_acr_hit": np.zeros(n, dtype=np.int8),
        "dense_rank": np.full(n, np.nan, dtype=np.float32),
        "name_tfidf_rank": np.full(n, np.nan, dtype=np.float32),
        "addr_tfidf_rank": np.full(n, np.nan, dtype=np.float32),
    }


def _rows_from_topk(D: np.ndarray, I: np.ndarray,
                    s1_ids_arr: np.ndarray, pool_ids_arr: np.ndarray,
                    src_tag: str,
                    score_col: str, rank_col: str,
                    drop_zero_score: bool) -> pd.DataFrame:
    n_s1, K = I.shape
    ranks = np.broadcast_to(np.arange(K, dtype=np.int32), (n_s1, K))
    s1_bc = np.broadcast_to(s1_ids_arr.reshape(-1, 1), (n_s1, K))

    valid = I >= 0
    if drop_zero_score:
        valid = valid & (D > 0)
    v = valid.ravel()
    n = int(v.sum())
    if n == 0:
        return pd.DataFrame(columns=ARM_COLS)

    cols = _empty_arm_df(n)
    cols[score_col] = D.ravel()[v].astype(np.float32)
    cols[rank_col] = ranks.ravel()[v].astype(np.float32)
    df = pd.DataFrame({
        "s1_id": s1_bc.ravel()[v],
        "cand_id": pool_ids_arr[I.ravel()[v]],
        "cand_source": src_tag,
        **cols,
    })
    return df


def _rows_from_key_arm(q_arr: np.ndarray, p_arr: np.ndarray,
                       s1_ids_arr: np.ndarray, pool_ids_arr: np.ndarray,
                       src_tag: str,
                       key_col: str) -> pd.DataFrame:
    n = int(q_arr.shape[0])
    if n == 0:
        return pd.DataFrame(columns=ARM_COLS)
    cols = _empty_arm_df(n)
    cols[key_col] = np.ones(n, dtype=np.int8)
    df = pd.DataFrame({
        "s1_id": s1_ids_arr[q_arr],
        "cand_id": pool_ids_arr[p_arr],
        "cand_source": src_tag,
        **cols,
    })
    return df


# ---------------------------------------------------------------------------
# Union + top-N cap
# ---------------------------------------------------------------------------
def _union_and_cap(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=ARM_COLS + ["n_arms_hit"])

    _log(f"[blocking]   union: aggregating {len(df):,} arm records…")
    agg = {
        "dense_score": "max", "name_tfidf_score": "max", "addr_tfidf_score": "max",
        "key_locrare_hit": "max", "key_house_hit": "max", "key_acr_hit": "max",
        "dense_rank": "min", "name_tfidf_rank": "min", "addr_tfidf_rank": "min",
    }
    df = df.groupby(["s1_id", "cand_id", "cand_source"], as_index=False,
                    sort=False).agg(agg)
    _log(f"[blocking]   union: {len(df):,} unique pairs after collapse")

    score_hit = df[["dense_score", "name_tfidf_score",
                    "addr_tfidf_score"]].notna().sum(axis=1)
    key_hit = (df[["key_locrare_hit", "key_house_hit", "key_acr_hit"]]
               .fillna(0).astype(np.int8).sum(axis=1))
    df["n_arms_hit"] = (score_hit + key_hit).astype(np.int8)

    best = df[["dense_score", "name_tfidf_score", "addr_tfidf_score"]
              ].max(axis=1).fillna(0.0)
    df = df.assign(_best=best).sort_values(
        ["s1_id", "n_arms_hit", "_best"],
        ascending=[True, False, False])
    df = df.groupby("s1_id", as_index=False, sort=False).head(top_n)
    df = df.drop(columns="_best").reset_index(drop=True)
    _log(f"[blocking]   union: {len(df):,} pairs after top-{top_n}-per-S1 cap")
    return df


# ---------------------------------------------------------------------------
# Per-partition driver
# ---------------------------------------------------------------------------
KEEP_COLS = [
    "entity_id", "core_name", "name_roman", "address_expanded",
    "locality_tokens", "postal_code", "postal_prefix3",
    "house_number", "street_tokens", "acronym",
]


def _process_pool(split: str, country: str, src_tag: str,
                  s1: pd.DataFrame, s1_ids_arr: np.ndarray,
                  pool_path: Path, train_mask=None) -> list[pd.DataFrame]:
    """Run all six arms for a single pool (S2 or S3). Returns arm frames."""
    pool = pd.read_parquet(pool_path, columns=KEEP_COLS)
    pool_ids_arr = pool["entity_id"].to_numpy()
    _log(f"[blocking] {split}/{country} vs {src_tag}: "
         f"{len(s1_ids_arr):,} × {len(pool_ids_arr):,}")

    frames: list[pd.DataFrame] = []

    # ---- A1 dense ----
    try:
        _log(f"[blocking]   A1 dense: loading embeddings…")
        q_vec, q_ids = embed_mod.load_embeddings(split, "s1", country)
        q_vec = np.asarray(q_vec)
        q_ids_arr = q_ids["entity_id"].to_numpy()
        if train_mask is not None:
            q_vec = q_vec[train_mask]
            q_ids_arr = q_ids_arr[train_mask]
        if not np.array_equal(q_ids_arr, s1_ids_arr):
            raise RuntimeError("S1 embedding ids misaligned with normalized parquet")
        p_vec, p_ids = embed_mod.load_embeddings(split, src_tag.lower(), country)
        if not np.array_equal(p_ids["entity_id"].to_numpy(), pool_ids_arr):
            raise RuntimeError(f"{src_tag} embedding ids misaligned with normalized parquet")
        D, I = _dense_knn(q_vec, np.asarray(p_vec), C.K_DENSE)
        df_a1 = _rows_from_topk(D, I, s1_ids_arr, pool_ids_arr, src_tag,
                                score_col="dense_score",
                                rank_col="dense_rank",
                                drop_zero_score=False)
        _log(f"[blocking]   A1 dense: {len(df_a1):,} rows")
        frames.append(df_a1)
        del D, I, q_vec, p_vec, df_a1
        gc.collect()
    except Exception as e:
        _log(
            f"[blocking] A1 dense skipped ({type(e).__name__}: {e}); continuing")

    # ---- A2 name TF-IDF ----
    _log(f"[blocking]   A2 name TF-IDF: fitting…")
    mat_p, mat_q = _tfidf_mats(pool["core_name"].fillna("").tolist(),
                               s1["core_name"].fillna("").tolist(),
                               "char_wb", C.TFIDF_NGRAM, C.TFIDF_MAX_DF_NAME)
    gc.collect()
    _log(
        f"[blocking]   A2 name TF-IDF: top-k on q{mat_q.shape} p{mat_p.shape}…")
    D, I = _sparse_topk(mat_q, mat_p, C.K_NAME_TFIDF)
    del mat_p, mat_q
    gc.collect()
    df_a2 = _rows_from_topk(D, I, s1_ids_arr, pool_ids_arr, src_tag,
                            score_col="name_tfidf_score",
                            rank_col="name_tfidf_rank",
                            drop_zero_score=True)
    _log(f"[blocking]   A2 name TF-IDF: {len(df_a2):,} rows")
    frames.append(df_a2)
    del D, I, df_a2
    gc.collect()

    # ---- A3 address TF-IDF ----
    _log(f"[blocking]   A3 addr TF-IDF: fitting…")
    mat_p, mat_q = _tfidf_mats(pool["address_expanded"].fillna("").tolist(),
                               s1["address_expanded"].fillna("").tolist(),
                               C.TFIDF_ADDR_ANALYZER, C.TFIDF_ADDR_NGRAM,
                               C.TFIDF_MAX_DF)
    gc.collect()
    _log(
        f"[blocking]   A3 addr TF-IDF: top-k on q{mat_q.shape} p{mat_p.shape}…")
    D, I = _sparse_topk(mat_q, mat_p, C.K_ADDR_TFIDF)
    del mat_p, mat_q
    gc.collect()
    df_a3 = _rows_from_topk(D, I, s1_ids_arr, pool_ids_arr, src_tag,
                            score_col="addr_tfidf_score",
                            rank_col="addr_tfidf_rank",
                            drop_zero_score=True)
    _log(f"[blocking]   A3 addr TF-IDF: {len(df_a3):,} rows")
    frames.append(df_a3)
    del D, I, df_a3
    gc.collect()

    # ---- A4 (rare-token, locality|postal) ----
    _log(f"[blocking]   A4 key locrare: indexing…")
    q_arr, p_arr, dropped = _key_arm(s1, pool, _s1_keys_locrare,
                                     _pool_keys_locrare, C.KEY_BLOCK_CAP)
    _log(f"[blocking]   A4 pairs={len(q_arr):,} dropped-keys={dropped}")
    frames.append(_rows_from_key_arm(q_arr, p_arr, s1_ids_arr, pool_ids_arr,
                                     src_tag, "key_locrare_hit"))
    del q_arr, p_arr
    gc.collect()

    # ---- A5 (house_number, street_token) ----
    _log(f"[blocking]   A5 key house: indexing…")
    q_arr, p_arr, dropped = _key_arm(s1, pool, _s1_keys_house,
                                     _pool_keys_house, C.KEY_BLOCK_CAP)
    _log(f"[blocking]   A5 pairs={len(q_arr):,} dropped-keys={dropped}")
    frames.append(_rows_from_key_arm(q_arr, p_arr, s1_ids_arr, pool_ids_arr,
                                     src_tag, "key_house_hit"))
    del q_arr, p_arr
    gc.collect()

    # ---- A6 (acronym, postal_prefix3|locality) ----
    _log(f"[blocking]   A6 key acr: indexing…")
    q_arr, p_arr, dropped = _key_arm(s1, pool, _s1_keys_acr,
                                     _pool_keys_acr, C.KEY_ACRONYM_CAP)
    _log(f"[blocking]   A6 pairs={len(q_arr):,} dropped-keys={dropped}")
    frames.append(_rows_from_key_arm(q_arr, p_arr, s1_ids_arr, pool_ids_arr,
                                     src_tag, "key_acr_hit"))
    del q_arr, p_arr, pool
    gc.collect()

    return frames


def block_partition(split: str, country: str) -> pd.DataFrame:
    """Run the six arms for one (split, country) partition and write results."""
    C.ensure_dirs()
    out_path = C.BLOCKING_DIR / f"{split}__{country}.parquet"
    if out_path.exists():
        _log(f"[blocking] cached: {out_path}")
        return pd.read_parquet(out_path)

    s1_path = C.NORMALIZED_DIR / f"{split}__s1__{country}.parquet"
    s2_path = C.NORMALIZED_DIR / f"{split}__s2__{country}.parquet"
    s3_path = C.NORMALIZED_DIR / f"{split}__s3__{country}.parquet"
    if not s1_path.exists():
        _log(f"[blocking] no S1 for {split}/{country} — skipping")
        return pd.DataFrame()

    s1 = pd.read_parquet(s1_path, columns=KEEP_COLS)

    train_mask = None
    # Train split: keep only S1 groups that downstream stages use
    # (BLOCK_TRAIN_GROUPS; R is only needed when the reranker is trained).
    if split == "train":
        from . import splits as splits_mod
        split_df = splits_mod.load()
        valid_ids = set(split_df.loc[
            split_df["group"].isin(list(C.BLOCK_TRAIN_GROUPS)), "entity_id"])
        train_mask = s1["entity_id"].isin(valid_ids).to_numpy()
        s1 = s1[train_mask].reset_index(drop=True)

    s1_ids_arr = s1["entity_id"].to_numpy()
    _log(f"[blocking] === {split}/{country}: {len(s1_ids_arr):,} S1 rows ===")

    all_frames: list[pd.DataFrame] = []

    for src_tag, pool_path in (("S2", s2_path), ("S3", s3_path)):
        if not pool_path.exists():
            continue
        frames = _process_pool(split, country, src_tag,
                               s1, s1_ids_arr, pool_path, train_mask)
        all_frames.extend(frames)
        del frames
        gc.collect()

    del s1
    gc.collect()

    _log(f"[blocking] {split}/{country}: concatenating "
         f"{len(all_frames)} arm frames…")
    big = pd.concat(all_frames, ignore_index=True, copy=False)
    all_frames.clear()
    del all_frames
    gc.collect()
    _log(f"[blocking] {split}/{country}: total pre-union {len(big):,} rows "
         f"({big.memory_usage(deep=True).sum() / 1e9:.2f} GB)")

    out = _union_and_cap(big, C.UNION_TOP_N)
    del big
    gc.collect()
    write_parquet(out, out_path)
    _log(f"[blocking] wrote {out_path}  {len(out):,} pairs "
         f"({out['s1_id'].nunique():,} unique S1)")

    # Fire-and-forget upload — crash-safe checkpoint before next partition.
    _upload_partition_async(out_path, "blocking")
    return out


def _block_worker(split: str, country: str, n_threads: int) -> str:
    """Loky worker entry: one partition with a share of the cores."""
    global _TAG
    C.N_JOBS = n_threads
    _TAG = f"[{split}/{country}] "
    block_partition(split, country)
    return f"{split}/{country}"


def block_all_partitions() -> None:
    countries = sorted({p.stem.split("__")[-1] for p in
                        C.NORMALIZED_DIR.glob("*__s1__*.parquet")})
    _log(f"[blocking] countries detected: {countries}")
    jobs = [(split, country) for split in ("train", "test") for country in countries
            if (C.NORMALIZED_DIR / f"{split}__s1__{country}.parquet").exists()]
    # Biggest pools first so the slowest partition starts immediately.
    def _size(sc):
        return sum(p.stat().st_size for p in
                   C.NORMALIZED_DIR.glob(f"{sc[0]}__s[23]__{sc[1]}.parquet"))
    jobs.sort(key=_size, reverse=True)
    n_jobs = max(1, min(C.BLOCK_JOBS, len(jobs)))
    if n_jobs == 1:
        for split, country in jobs:
            block_partition(split, country)
        return
    from joblib import Parallel, delayed
    threads = max(1, round(C.N_JOBS / n_jobs))
    _log(f"[blocking] {len(jobs)} partitions on {n_jobs} workers × "
         f"{threads} threads: {jobs}")
    for done in Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered")(
            delayed(_block_worker)(s, c, threads) for s, c in jobs):
        _log(f"[blocking] partition done: {done}")


# ---------------------------------------------------------------------------
# Blocking recall diagnostic (against ground truth) — plan §4.5
# ---------------------------------------------------------------------------
def blocking_recall_report() -> None:
    from .io_utils import explode_ground_truth, read_ground_truth
    from . import splits as splits_mod
    gt = explode_ground_truth(read_ground_truth(C.TRAIN_GT))
    split = splits_mod.load()
    gv = split.loc[split["group"].isin(["G", "V"])]
    gt_gv = gt[gt["source1_entity_id"].isin(set(gv["entity_id"]))]

    per_country = {}
    for country_path in C.BLOCKING_DIR.glob("train__*.parquet"):
        country = country_path.stem.split("__")[-1]
        cand = pd.read_parquet(country_path,
                               columns=["s1_id", "cand_id", "n_arms_hit"])
        cand_set = set(zip(cand["s1_id"], cand["cand_id"]))
        # All G∪V S1 of this country (not only those that got candidates),
        # otherwise S1 with zero candidates silently inflate recall.
        gt_country = gt_gv[gt_gv["source1_entity_id"].isin(
            set(gv.loc[gv["country"] == country, "entity_id"]))]
        gt_pairs = set(
            zip(gt_country["source1_entity_id"], gt_country["matched_id"]))
        hits = len(gt_pairs & cand_set)
        per_country[country] = {
            "true_pairs": len(gt_pairs),
            "hits": hits,
            "recall": hits / max(len(gt_pairs), 1),
        }
    report = "\n".join(
        f"{k}: {v['recall']:.4f} (hits {v['hits']:,}/{v['true_pairs']:,})"
        for k, v in per_country.items()
    )
    _log("[blocking] recall on train G∪V:\n" + report)
    (C.REPORTS_DIR / "blocking_recall.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    block_all_partitions()
    blocking_recall_report()
