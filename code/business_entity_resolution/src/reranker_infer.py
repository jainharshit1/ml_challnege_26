"""Cross-encoder scoring for pre-filtered pairs (plan §6.3).

Loads either the fine-tuned reranker (default when present) or the zero-shot
BAAI/bge-reranker-v2-m3. Sorts by length before batching to minimise padding
and writes per-partition parquet chunks so a crash doesn't lose progress.

Output: artifacts/reranker_scores/{split}__{country}.parquet
  columns [s1_id, cand_id, rerank_score]
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .reranker_train import _pair_text
from .io_utils import write_parquet


def _load_reranker(prefer_finetuned: bool = True):
    """Return callable score(list_of_(a,b)) → np.ndarray of logits→sigmoid."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = C.MODELS_DIR / "bge_reranker_ft"
    if prefer_finetuned and model_path.exists():
        src = str(model_path)
    else:
        src = C.RERANKER_MODEL_LOCAL or C.RERANKER_MODEL
    print(f"[reranker_infer] loading {src}")
    _local = src.startswith("/") or src.startswith(".")
    tok = AutoTokenizer.from_pretrained(src, local_files_only=_local)
    model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=1, local_files_only=_local)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[reranker_infer] torch device: {device}")
    model = model.to(device)
    if device == "cuda":
        try:
            model = model.half()
        except Exception:
            pass
        # A5000/Ampere: TF32 for any fp32 fallbacks
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        # Optional: torch.compile for 20-30% speedup (PyTorch 2+)
        if getattr(C, "RERANK_TORCH_COMPILE", False) and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
                print("[reranker_infer] torch.compile enabled")
            except Exception as e:
                print(f"[reranker_infer] torch.compile skipped ({type(e).__name__})")

    @torch.inference_mode()
    def score(pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        a = [p[0] for p in pairs]
        b = [p[1] for p in pairs]
        enc = tok(a, b, truncation=True, padding=True,
                  max_length=C.RERANK_MAX_TOKENS, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        return 1.0 / (1.0 + np.exp(-logits))

    return score


def _build_text(row) -> str:
    def _s(v): return v if isinstance(v, str) else ""
    return _pair_text(
        _s(row["core_name"]) or _s(row["name_roman"]),
        _s(row["address_expanded"]),
        name_original=_s(row["name_roman"]),
        script=row["name_script"] if isinstance(row.get("name_script"), str) else "latin",
    )


def score_partition(split: str, country: str,
                    batch_size: int = C.RERANK_INFER_BATCH_SIZE) -> Path | None:
    C.ensure_dirs()
    src_path = C.PREFILTER_DIR / f"{split}__{country}.parquet"
    out_path = C.RERANKER_DIR / f"{split}__{country}.parquet"
    if not src_path.exists():
        print(f"[reranker_infer] no prefilter for {split}/{country}")
        return None
    if out_path.exists():
        print(f"[reranker_infer] cached: {out_path.name}")
        return out_path

    cand = pd.read_parquet(src_path)
    # Cascade: skip cross-encoder for confident matches/non-matches (§ cheap-rerank).
    if getattr(C, "RERANK_CASCADE_ENABLED", False) and "stage_a_score" in cand.columns:
        lo = getattr(C, "RERANK_CASCADE_LOW", 0.15)
        hi = getattr(C, "RERANK_CASCADE_HIGH", 0.85)
        uncertain = cand["stage_a_score"].between(lo, hi, inclusive="both")
        skipped = (~uncertain).sum()
        print(f"[reranker_infer] cascade: {uncertain.sum():,} uncertain / "
              f"{skipped:,} skipped (score using stage_a directly)")
        cand_uncertain = cand.loc[uncertain, ["s1_id", "cand_id", "cand_source"]].copy()
        cand_confident = cand.loc[~uncertain, ["s1_id", "cand_id",
                                                "stage_a_score"]].rename(
            columns={"stage_a_score": "rerank_score"})
        cand = cand_uncertain
    else:
        cand = cand[["s1_id", "cand_id", "cand_source"]]
        cand_confident = None
    s1_df = pd.read_parquet(
        C.NORMALIZED_DIR / f"{split}__s1__{country}.parquet",
        columns=["entity_id", "core_name", "name_roman", "name_script",
                 "address_expanded"],
    ).set_index("entity_id")
    pool_dfs: dict[str, pd.DataFrame] = {}
    for src in ("s2", "s3"):
        p = C.NORMALIZED_DIR / f"{split}__{src}__{country}.parquet"
        if p.exists():
            pool_dfs[src.upper()] = pd.read_parquet(
                p, columns=["entity_id", "core_name", "name_roman",
                            "name_script", "address_expanded"],
            ).set_index("entity_id")

    # Build texts once
    texts_a, texts_b, order = [], [], []
    for i, row in enumerate(cand.itertuples(index=False)):
        if row.s1_id not in s1_df.index:
            continue
        pool = pool_dfs.get(row.cand_source)
        if pool is None or row.cand_id not in pool.index:
            continue
        texts_a.append(_build_text(s1_df.loc[row.s1_id]))
        texts_b.append(_build_text(pool.loc[row.cand_id]))
        order.append(i)
    print(f"[reranker_infer] {split}/{country}: {len(order):,} pairs to score")

    # Length-sort to reduce padding
    scorer = _load_reranker()
    lens = [len(a) + len(b) for a, b in zip(texts_a, texts_b)]
    idx_sorted = np.argsort(lens)
    scores = np.zeros(len(order), dtype=np.float32)
    chunk = 4096
    for start in range(0, len(idx_sorted), chunk):
        sub = idx_sorted[start:start + chunk]
        pairs = [(texts_a[i], texts_b[i]) for i in sub]
        batch_scores: list[float] = []
        for b_start in range(0, len(pairs), batch_size):
            batch_scores.extend(
                scorer(pairs[b_start : b_start + batch_size]).tolist()
            )
        scores[sub] = np.array(batch_scores, dtype=np.float32)
        if (start // chunk) % 5 == 0:
            print(f"[reranker_infer]   {start+len(sub):,}/{len(order):,}")

    out = pd.DataFrame({
        "s1_id": cand["s1_id"].iloc[order].values,
        "cand_id": cand["cand_id"].iloc[order].values,
        "rerank_score": scores,
    })
    if cand_confident is not None and not cand_confident.empty:
        out = pd.concat([out, cand_confident], ignore_index=True)
        print(f"[reranker_infer] merged {len(cand_confident):,} cascade-skipped rows")
    write_parquet(out, out_path)
    print(f"[reranker_infer] wrote {out_path}")
    return out_path


def score_all() -> None:
    countries = sorted({p.stem.split("__")[-1] for p in
                        C.PREFILTER_DIR.glob("*__*.parquet")})
    for split in ("train", "test"):
        for country in countries:
            if (C.PREFILTER_DIR / f"{split}__{country}.parquet").exists():
                score_partition(split, country)


if __name__ == "__main__":
    score_all()
