"""TSV/Parquet I/O helpers.

The competition files are tab-separated; ID list columns must never be
comma-parsed. All submission files write utf-8, tab-separated, no quoting.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

import pandas as pd

TSV_KW = dict(sep="\t", dtype=str, keep_default_na=False, na_values=[""])


def read_source_tsv(path: str | Path) -> pd.DataFrame:
    """Read a *_source*.tsv file with string dtypes."""
    df = pd.read_csv(path, **TSV_KW)
    return df


def read_ground_truth(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("")
    return df


def explode_ground_truth(gt: pd.DataFrame) -> pd.DataFrame:
    """One row per (source1_entity_id, matched_id)."""
    rows = []
    for s1_id, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if not ids or not ids.strip():
            continue
        for mid in ids.split(","):
            mid = mid.strip()
            if mid:
                rows.append((s1_id, mid))
    return pd.DataFrame(rows, columns=["source1_entity_id", "matched_id"])


def write_id_list_tsv(
    path: str | Path,
    s1_ids: Iterable[str],
    id_lists: Iterable[Iterable[str]],
    id_column: str,
) -> None:
    """Write a submission-style TSV.

    id_column is either 'matched_entity_ids' or 'candidate_entity_ids'.
    Empty lists produce an empty string (singleton row).
    Enforces: no duplicate S1 rows and no duplicate IDs within a list.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"source1_entity_id\t{id_column}\n")
        for s1, ids in zip(s1_ids, id_lists):
            if s1 in seen:
                raise ValueError(f"duplicate source1_entity_id row: {s1}")
            seen.add(s1)
            uniq = []
            seen_ids = set()
            for i in ids or ():
                if i and i not in seen_ids:
                    seen_ids.add(i)
                    uniq.append(i)
            fh.write(f"{s1}\t{','.join(uniq)}\n")


def read_id_list_tsv(path: str | Path, id_column: str) -> dict[str, list[str]]:
    """Read a submission-style TSV back into a {s1: [ids]} dict."""
    out: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        assert header == ["source1_entity_id", id_column], header
        for line in fh:
            s1, _, rest = line.partition("\t")
            rest = rest.rstrip("\n")
            out[s1] = [x for x in rest.split(",") if x] if rest else []
    return out


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def source_prefix(entity_id: str) -> str:
    """S1-... → 'S1' etc."""
    return entity_id.split("-", 1)[0] if "-" in entity_id else ""
