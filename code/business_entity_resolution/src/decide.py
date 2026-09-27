"""Decision layer (plan §8.1, §8.5).

Same order used in tuning and inference:
  1. Pair filter (p ≥ pair_thresh)
  2. One-to-one assignment (hard constraint): sort remaining triples within
     country by p descending; assign each candidate to first S1 that claims
     it; drop later claims.
  3. Per-S1 cap (default 12)
  4. Singleton gate: if S1's best remaining p < singleton_thresh → empty list
  5. Optional margin rule

The one-to-one step is scoped by country (plan §1.2: 0 cross-country matches
in training) but the code passes country in as a column, not hard-coded.

Output helpers write output/matching_results.tsv covering every test S1.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .io_utils import read_source_tsv, write_id_list_tsv


def decide(
    scores: pd.DataFrame,
    pair_thresh: float = C.PAIR_THRESH_DEFAULT,
    singleton_thresh: float = C.SINGLETON_THRESH_DEFAULT,
    margin_thresh: float | None = C.MARGIN_THRESH_DEFAULT,
    per_s1_cap: int = C.PER_S1_CAP,
    country_col: str | None = None,
) -> pd.DataFrame:
    """Apply the four-step decision to a scored table.

    scores columns required: [s1_id, cand_id, p_match]
    optional: [country_col] (if given, one-to-one is applied within country).
    Returns a filtered DataFrame with columns [s1_id, cand_id, p_match].
    """
    df = scores.copy()
    # 1. Pair filter
    df = df[df["p_match"] >= pair_thresh]
    if df.empty:
        return df

    # 2. One-to-one assignment — vectorized: sort by p_match desc, then
    #    drop later occurrences of each cand_id (within country if provided).
    if country_col and country_col in df.columns:
        df = df.sort_values([country_col, "p_match"], ascending=[True, False])
        df = df[~df.duplicated(subset=[country_col, "cand_id"], keep="first")]
    else:
        df = df.sort_values("p_match", ascending=False)
        df = df[~df.duplicated(subset=["cand_id"], keep="first")]

    if df.empty:
        return df

    # 3. Cap per S1
    df = (df.sort_values(["s1_id", "p_match"], ascending=[True, False])
             .groupby("s1_id", as_index=False)
             .head(per_s1_cap))

    # 4. Singleton gate: drop entire S1's list if its best is < singleton_thresh
    best = df.groupby("s1_id")["p_match"].transform("max")
    df = df[best >= singleton_thresh]

    # 5. Optional margin rule
    if margin_thresh is not None and not df.empty:
        cand_grp = df.groupby("cand_id")
        cand_best = cand_grp["p_match"].transform("max")
        # a pair survives only if its own p is within `margin_thresh` of the
        # best-for-cand; that is, gap ≤ margin.  If it's THE best, gap == 0.
        gap = cand_best - df["p_match"]
        df = df[gap <= margin_thresh]

    return df.reset_index(drop=True)


def decide_all_test(
    pair_thresh: float, singleton_thresh: float,
    margin_thresh: float | None = None,
    loco_tag: str = "",
    france_overrides: tuple[float, float, float | None] | None = None,
) -> Path:
    """Apply thresholds to every test country, aggregate, write matching_results.tsv.

    france_overrides: optional (pair, singleton, margin) applied only to
    country == 'France'. Everywhere else uses the main thresholds.
    """
    test_s1 = read_source_tsv(C.TEST_SOURCE["s1"])
    if C.NORMALIZE_ROW_CAP:
        test_s1 = test_s1.iloc[:C.NORMALIZE_ROW_CAP].copy()
    order = test_s1["entity_id"].tolist()
    country_of = dict(zip(test_s1["entity_id"], test_s1["country"]))
    by_s1: dict[str, list[tuple[str, float]]] = {sid: [] for sid in order}

    for p in C.FINAL_SCORES_DIR.glob(f"test__*__scores{loco_tag}.parquet"):
        country = p.stem.split("__")[1]
        df = pd.read_parquet(p)
        if france_overrides and country == "France":
            pt, st, mg = france_overrides
        else:
            pt, st, mg = pair_thresh, singleton_thresh, margin_thresh
        kept = decide(df, pair_thresh=pt, singleton_thresh=st, margin_thresh=mg)
        for row in kept.itertuples(index=False):
            by_s1[row.s1_id].append((row.cand_id, float(row.p_match)))

    out_path = C.OUTPUT_DIR / "matching_results.tsv"
    id_lists = []
    for sid in order:
        # highest-p first
        ids = [c for c, _ in sorted(by_s1[sid], key=lambda x: -x[1])]
        id_lists.append(ids)
    write_id_list_tsv(out_path, order, id_lists,
                      id_column="matched_entity_ids")
    n_nonempty = sum(1 for v in id_lists if v)
    print(f"[decide] wrote {out_path}  non-empty={n_nonempty:,}/{len(order):,}  "
          f"mean-matches/S1={sum(len(v) for v in id_lists)/max(len(order),1):.3f}")
    return out_path


if __name__ == "__main__":
    decide_all_test(
        pair_thresh=C.PAIR_THRESH_DEFAULT,
        singleton_thresh=C.SINGLETON_THRESH_DEFAULT,
        margin_thresh=C.MARGIN_THRESH_DEFAULT,
    )
