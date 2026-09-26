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

Every arm may return NaN for pairs it did not produce; scores + ranks are
per-S1. Union is capped at UNION_TOP_N per S1 by (n_arms_hit desc, best
normalized arm score).
"""
from __future__ import annotations
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


# ---------------------------------------------------------------------------
# Dense arm (A1)
# ---------------------------------------------------------------------------
def _dense_knn(query_vecs: np.ndarray, pool_vecs: np.ndarray,
               k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores [N,k], indices [N,k]) using FAISS if available."""
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("faiss is required for the dense blocking arm") from e

    n_pool = pool_vecs.shape[0]
    d = pool_vecs.shape[1]
    pool = np.ascontiguousarray(pool_vecs.astype(np.float32))
    query = np.ascontiguousarray(query_vecs.astype(np.float32))

    if n_pool <= C.FAISS_FLAT_MAX:
        index = faiss.IndexFlatIP(d)
        index.add(pool)
    else:
        quantizer = faiss.IndexFlatIP(d)
        nlist = min(C.FAISS_NLIST, max(64, int(np.sqrt(n_pool) * 4)))
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(C.SEED)
        train_n = min(C.FAISS_TRAIN_SAMPLE, n_pool)
        train_idx = rng.choice(n_pool, size=train_n, replace=False)
        index.train(pool[train_idx])
        index.add(pool)
        index.nprobe = C.FAISS_NPROBE_DEFAULT

    # move to GPU if available
    used_gpu = False
    try:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        used_gpu = True
    except Exception:
        pass
    print(f"[blocking]   FAISS device: {'gpu' if used_gpu else 'cpu'}  "
          f"pool={n_pool:,}  k={k}")

    D, I = index.search(query, k)
    return D, I


# ---------------------------------------------------------------------------
# Sparse TF-IDF arms (A2 / A3)
# ---------------------------------------------------------------------------
def _fit_tfidf(texts: Iterable[str]) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=C.TFIDF_NGRAM,
        min_df=C.TFIDF_MIN_DF,
        max_features=C.TFIDF_MAX_FEATURES,
        norm="l2",
        sublinear_tf=True,
    ).fit(texts)


def _sparse_topk(mat_q: sp.csr_matrix, mat_p: sp.csr_matrix,
                 k: int, row_batch: int = C.TFIDF_ROW_BATCH
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Cosine top-k of q rows against p rows (both L2-normalised).

    Uses sparse_dot_topn when available; otherwise chunked matmul (slower
    but zero-dep).
    """
    try:
        from sparse_dot_topn import sp_matmul_topn  # type: ignore
        pT = mat_p.T.tocsr()
        out = sp_matmul_topn(mat_q, pT, top_n=k, threshold=0.0, sort=True)
        # convert to dense k arrays
        n = mat_q.shape[0]
        D = np.zeros((n, k), dtype=np.float32)
        I = -np.ones((n, k), dtype=np.int64)
        for i in range(n):
            row = out.getrow(i)
            cols = row.indices
            vals = row.data
            order = np.argsort(-vals)[:k]
            D[i, : len(order)] = vals[order]
            I[i, : len(order)] = cols[order]
        return D, I
    except Exception:
        pT = mat_p.T
        n_q = mat_q.shape[0]
        D = np.zeros((n_q, k), dtype=np.float32)
        I = -np.ones((n_q, k), dtype=np.int64)
        for start in range(0, n_q, row_batch):
            end = min(start + row_batch, n_q)
            sim = mat_q[start:end] @ pT
            sim = sim.toarray()
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
    if not x:
        return []
    return [t for t in x.split() if t]


def _key_arm(s1_df: pd.DataFrame, pool_df: pd.DataFrame,
             s1_keys_fn, pool_keys_fn,
             cap: int) -> list[tuple[int, int]]:
    """Return list of (query_idx, pool_idx) pairs whose keys intersect.

    query_idx/pool_idx are positional; the caller maps back to entity ids.
    Keys are produced by *_keys_fn(row) → Iterable[str]. Any key producing
    more than `cap` pool matches is dropped.
    """
    pool_index: dict[str, list[int]] = defaultdict(list)
    for j, row in enumerate(pool_df.itertuples(index=False)):
        for k in pool_keys_fn(row):
            pool_index[k].append(j)
    # drop over-cap keys
    kept = {k: v for k, v in pool_index.items() if 0 < len(v) <= cap}
    dropped = len(pool_index) - len(kept)

    pairs: list[tuple[int, int]] = []
    for i, row in enumerate(s1_df.itertuples(index=False)):
        seen = set()
        for k in s1_keys_fn(row):
            for j in kept.get(k, ()):
                if j not in seen:
                    seen.add(j)
                    pairs.append((i, j))
    return pairs, dropped


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
    hn = getattr(row, "house_number", "") or ""
    if not hn:
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
# Union + top-N cap
# ---------------------------------------------------------------------------
def _union_and_cap(records: list[dict], top_n: int) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=[
            "s1_id", "cand_id", "cand_source",
            "dense_score", "name_tfidf_score", "addr_tfidf_score",
            "key_locrare_hit", "key_house_hit", "key_acr_hit",
            "dense_rank", "name_tfidf_rank", "addr_tfidf_rank",
            "n_arms_hit",
        ])
    df = pd.DataFrame(records)
    # collapse duplicate (s1_id, cand_id, cand_source) triples across arms
    agg = {
        "dense_score": "max", "name_tfidf_score": "max", "addr_tfidf_score": "max",
        "key_locrare_hit": "max", "key_house_hit": "max", "key_acr_hit": "max",
        "dense_rank": "min", "name_tfidf_rank": "min", "addr_tfidf_rank": "min",
    }
    df = df.groupby(["s1_id", "cand_id", "cand_source"], as_index=False).agg(agg)
    # arm indicator: score non-null OR key hit
    arm_cols = [
        "dense_score", "name_tfidf_score", "addr_tfidf_score",
        "key_locrare_hit", "key_house_hit", "key_acr_hit",
    ]
    hits = df[arm_cols].notna().astype(int)
    for k in ("key_locrare_hit", "key_house_hit", "key_acr_hit"):
        hits[k] = df[k].fillna(0).astype(int)
    df["n_arms_hit"] = hits.sum(axis=1)
    # rank by (n_arms_hit desc, best normalized score desc)
    df["_best"] = df[["dense_score", "name_tfidf_score", "addr_tfidf_score"]
                     ].max(axis=1).fillna(0)
    df = df.sort_values(["s1_id", "n_arms_hit", "_best"],
                         ascending=[True, False, False])
    df = df.groupby("s1_id", as_index=False).head(top_n).drop(columns="_best")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-partition driver
# ---------------------------------------------------------------------------
def block_partition(split: str, country: str) -> pd.DataFrame:
    """Run the six arms for one (split, country) partition and write results."""
    C.ensure_dirs()
    out_path = C.BLOCKING_DIR / f"{split}__{country}.parquet"
    if out_path.exists():
        print(f"[blocking] cached: {out_path}")
        return pd.read_parquet(out_path)

    s1_path = C.NORMALIZED_DIR / f"{split}__s1__{country}.parquet"
    s2_path = C.NORMALIZED_DIR / f"{split}__s2__{country}.parquet"
    s3_path = C.NORMALIZED_DIR / f"{split}__s3__{country}.parquet"
    if not s1_path.exists():
        print(f"[blocking] no S1 for {split}/{country} — skipping")
        return pd.DataFrame()

    keep_cols = [
        "entity_id", "core_name", "name_roman", "address_expanded",
        "locality_tokens", "postal_code", "postal_prefix3",
        "house_number", "street_tokens", "acronym",
    ]
    s1 = pd.read_parquet(s1_path, columns=keep_cols)
    s1_ids = s1["entity_id"].tolist()
    pools: dict[str, pd.DataFrame] = {}
    if s2_path.exists():
        pools["S2"] = pd.read_parquet(s2_path, columns=keep_cols)
    if s3_path.exists():
        pools["S3"] = pd.read_parquet(s3_path, columns=keep_cols)

    all_records: list[dict] = []
    for src_tag, pool in pools.items():
        pool_ids = pool["entity_id"].tolist()
        print(f"[blocking] {split}/{country} vs {src_tag}: "
              f"{len(s1_ids):,} × {len(pool_ids):,}")

        # ---- A1 dense ----
        try:
            q_vec, _ = embed_mod.load_embeddings(split, "s1", country)
            p_vec, _ = embed_mod.load_embeddings(split, src_tag.lower(), country)
            D, I = _dense_knn(np.asarray(q_vec), np.asarray(p_vec), C.K_DENSE)
            for i in range(len(s1_ids)):
                for rk, (score, j) in enumerate(zip(D[i], I[i])):
                    if j < 0:
                        continue
                    all_records.append({
                        "s1_id": s1_ids[i],
                        "cand_id": pool_ids[j],
                        "cand_source": src_tag,
                        "dense_score": float(score),
                        "name_tfidf_score": np.nan,
                        "addr_tfidf_score": np.nan,
                        "key_locrare_hit": 0,
                        "key_house_hit": 0,
                        "key_acr_hit": 0,
                        "dense_rank": rk,
                        "name_tfidf_rank": np.nan,
                        "addr_tfidf_rank": np.nan,
                    })
        except Exception as e:
            print(f"[blocking] A1 dense skipped ({e}); continuing without it")

        # ---- A2 name TF-IDF ----
        name_texts_pool = pool["core_name"].fillna("").tolist()
        name_texts_q = s1["core_name"].fillna("").tolist()
        vec_name = _fit_tfidf(name_texts_pool + name_texts_q)
        mat_p = vec_name.transform(name_texts_pool)
        mat_q = vec_name.transform(name_texts_q)
        D, I = _sparse_topk(mat_q, mat_p, C.K_NAME_TFIDF)
        for i in range(len(s1_ids)):
            for rk, (score, j) in enumerate(zip(D[i], I[i])):
                if j < 0 or score <= 0:
                    continue
                all_records.append({
                    "s1_id": s1_ids[i], "cand_id": pool_ids[j], "cand_source": src_tag,
                    "dense_score": np.nan, "name_tfidf_score": float(score),
                    "addr_tfidf_score": np.nan,
                    "key_locrare_hit": 0, "key_house_hit": 0, "key_acr_hit": 0,
                    "dense_rank": np.nan, "name_tfidf_rank": rk, "addr_tfidf_rank": np.nan,
                })

        # ---- A3 address TF-IDF ----
        addr_texts_pool = pool["address_expanded"].fillna("").tolist()
        addr_texts_q = s1["address_expanded"].fillna("").tolist()
        vec_addr = _fit_tfidf(addr_texts_pool + addr_texts_q)
        mat_p = vec_addr.transform(addr_texts_pool)
        mat_q = vec_addr.transform(addr_texts_q)
        D, I = _sparse_topk(mat_q, mat_p, C.K_ADDR_TFIDF)
        for i in range(len(s1_ids)):
            for rk, (score, j) in enumerate(zip(D[i], I[i])):
                if j < 0 or score <= 0:
                    continue
                all_records.append({
                    "s1_id": s1_ids[i], "cand_id": pool_ids[j], "cand_source": src_tag,
                    "dense_score": np.nan, "name_tfidf_score": np.nan,
                    "addr_tfidf_score": float(score),
                    "key_locrare_hit": 0, "key_house_hit": 0, "key_acr_hit": 0,
                    "dense_rank": np.nan, "name_tfidf_rank": np.nan, "addr_tfidf_rank": rk,
                })

        # ---- A4 (rare-token, locality|postal) ----
        pairs, dropped = _key_arm(s1, pool, _s1_keys_locrare, _pool_keys_locrare,
                                   C.KEY_BLOCK_CAP)
        print(f"[blocking]   A4 pairs={len(pairs):,} dropped-keys={dropped}")
        for i, j in pairs:
            all_records.append({
                "s1_id": s1_ids[i], "cand_id": pool_ids[j], "cand_source": src_tag,
                "dense_score": np.nan, "name_tfidf_score": np.nan,
                "addr_tfidf_score": np.nan,
                "key_locrare_hit": 1, "key_house_hit": 0, "key_acr_hit": 0,
                "dense_rank": np.nan, "name_tfidf_rank": np.nan, "addr_tfidf_rank": np.nan,
            })

        # ---- A5 (house_number, street_token) ----
        pairs, dropped = _key_arm(s1, pool, _s1_keys_house, _pool_keys_house,
                                   C.KEY_BLOCK_CAP)
        print(f"[blocking]   A5 pairs={len(pairs):,} dropped-keys={dropped}")
        for i, j in pairs:
            all_records.append({
                "s1_id": s1_ids[i], "cand_id": pool_ids[j], "cand_source": src_tag,
                "dense_score": np.nan, "name_tfidf_score": np.nan,
                "addr_tfidf_score": np.nan,
                "key_locrare_hit": 0, "key_house_hit": 1, "key_acr_hit": 0,
                "dense_rank": np.nan, "name_tfidf_rank": np.nan, "addr_tfidf_rank": np.nan,
            })

        # ---- A6 (acronym, postal_prefix3|locality) ----
        pairs, dropped = _key_arm(s1, pool, _s1_keys_acr, _pool_keys_acr,
                                   C.KEY_ACRONYM_CAP)
        print(f"[blocking]   A6 pairs={len(pairs):,} dropped-keys={dropped}")
        for i, j in pairs:
            all_records.append({
                "s1_id": s1_ids[i], "cand_id": pool_ids[j], "cand_source": src_tag,
                "dense_score": np.nan, "name_tfidf_score": np.nan,
                "addr_tfidf_score": np.nan,
                "key_locrare_hit": 0, "key_house_hit": 0, "key_acr_hit": 1,
                "dense_rank": np.nan, "name_tfidf_rank": np.nan, "addr_tfidf_rank": np.nan,
            })

    out = _union_and_cap(all_records, C.UNION_TOP_N)
    write_parquet(out, out_path)
    print(f"[blocking] wrote {out_path}  {len(out):,} pairs "
          f"({out['s1_id'].nunique():,} unique S1)")
    return out


def block_all_partitions() -> None:
    """Iterate over every (split, country) discovered from normalized files."""
    countries = sorted({p.stem.split("__")[-1] for p in
                        C.NORMALIZED_DIR.glob("*__s1__*.parquet")})
    for split in ("train", "test"):
        for country in countries:
            if not (C.NORMALIZED_DIR / f"{split}__s1__{country}.parquet").exists():
                continue
            block_partition(split, country)


# ---------------------------------------------------------------------------
# Blocking recall diagnostic (against ground truth) — plan §4.5
# ---------------------------------------------------------------------------
def blocking_recall_report() -> None:
    from .io_utils import explode_ground_truth, read_ground_truth
    from . import splits as splits_mod
    gt = explode_ground_truth(read_ground_truth(C.TRAIN_GT))
    split = splits_mod.load()
    gv = split.loc[split["group"].isin(["G", "V"]), "entity_id"]
    gt_gv = gt[gt["source1_entity_id"].isin(set(gv))]

    per_country = {}
    for country_path in C.BLOCKING_DIR.glob("train__*.parquet"):
        country = country_path.stem.split("__")[-1]
        cand = pd.read_parquet(country_path,
                                columns=["s1_id", "cand_id", "n_arms_hit"])
        cand_set = set(zip(cand["s1_id"], cand["cand_id"]))
        gt_country = gt_gv[gt_gv["source1_entity_id"].isin(cand["s1_id"].unique())]
        gt_pairs = set(zip(gt_country["source1_entity_id"], gt_country["matched_id"]))
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
    print("[blocking] recall on train G∪V:\n" + report)
    (C.REPORTS_DIR / "blocking_recall.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    block_all_partitions()
    blocking_recall_report()
