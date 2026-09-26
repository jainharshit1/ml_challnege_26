"""Entity-level splits R / G / V / unused (plan §2).

Stratified by (country, match_count_bucket) so singletons are represented in
every group. Deterministic on SEED. Reads training ground truth + train_source1
(for the country of each S1 entity).

Output
------
artifacts/splits/split.parquet: columns [entity_id, country, match_count, group]
where group ∈ {"R","G","V","unused"}.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C
from .io_utils import read_ground_truth, read_source_tsv, write_parquet


def _bucket(n: int) -> str:
    for lo, hi in C.MATCH_COUNT_BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return "6-99"


def build() -> pd.DataFrame:
    C.ensure_dirs()
    cap = C.NORMALIZE_ROW_CAP or 0
    print(f"[splits] reading train_source1 + ground truth…"
          + (f" (capped to {cap:,} rows for smoke)" if cap else ""))
    s1 = read_source_tsv(C.TRAIN_SOURCE["s1"])[["entity_id", "country"]]
    if cap:
        s1 = s1.iloc[:cap].copy()
    gt = read_ground_truth(C.TRAIN_GT)
    gt["match_count"] = gt["matched_entity_ids"].apply(
        lambda x: 0 if not x or not x.strip() else len(
            [i for i in x.split(",") if i.strip()]
        )
    )
    df = s1.merge(
        gt[["source1_entity_id", "match_count"]],
        left_on="entity_id",
        right_on="source1_entity_id",
        how="left",
    )
    df["match_count"] = df["match_count"].fillna(0).astype(int)
    df["bucket"] = df["match_count"].apply(_bucket)

    rng = np.random.default_rng(C.SEED)
    df["group"] = "unused"

    parts: list[pd.DataFrame] = []
    for (country, bucket), sub in df.groupby(["country", "bucket"], sort=False):
        idx = np.array(sub.index)
        rng.shuffle(idx)
        n = len(idx)
        n_r = int(round(n * C.SPLIT_R))
        n_g = int(round(n * C.SPLIT_G))
        n_v = int(round(n * C.SPLIT_V))
        r_ids = idx[:n_r]
        g_ids = idx[n_r : n_r + n_g]
        v_ids = idx[n_r + n_g : n_r + n_g + n_v]
        assignments = pd.Series("unused", index=idx)
        assignments.loc[r_ids] = "R"
        assignments.loc[g_ids] = "G"
        assignments.loc[v_ids] = "V"
        parts.append(assignments.rename("group"))
    all_assign = pd.concat(parts).sort_index()
    df["group"] = all_assign

    if C.SUBSAMPLE_FRACTION < 1.0:
        print(f"[splits] SUBSAMPLE_FRACTION={C.SUBSAMPLE_FRACTION}: "
              "downsampling G entities (R and V left intact for stable eval).")
        g_mask = df["group"] == "G"
        g_idx = df.index[g_mask].to_numpy()
        keep = rng.choice(g_idx, size=int(round(len(g_idx) * C.SUBSAMPLE_FRACTION)),
                          replace=False)
        drop = np.setdiff1d(g_idx, keep)
        df.loc[drop, "group"] = "unused"

    out = df[["entity_id", "country", "match_count", "group"]].copy()
    write_parquet(out, C.SPLIT_PATH)

    print("[splits] group counts:")
    print(out.groupby(["group", "country"]).size().unstack(fill_value=0))
    print(f"[splits] wrote {C.SPLIT_PATH}")
    return out


def load() -> pd.DataFrame:
    return pd.read_parquet(C.SPLIT_PATH)


def entities(group: str, country: str | None = None) -> pd.Series:
    df = load()
    if country is not None:
        df = df[df["country"] == country]
    return df.loc[df["group"] == group, "entity_id"].reset_index(drop=True)


if __name__ == "__main__":
    build()
