"""Fine-tune BAAI/bge-reranker-v2-m3 on group R (plan §6.2).

Positives: R's true (S1, S2/S3) matches from train GT.
Negatives: R's Stage-A candidates that are NOT true matches, ranked by
Stage-A score (hard negatives). Up to RERANK_MAX_NEG_PER_POS per positive.
Adds explicit chain-name negatives (same core_name, different address) when
RERANK_INCLUDE_CHAIN_NEGS.

Training data comes exclusively from the provided dataset — no external
labels are used.

Output: artifacts/models/bge_reranker_ft/  (HF model directory)
"""
from __future__ import annotations
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import splits as splits_mod
from .io_utils import explode_ground_truth, read_ground_truth


def _pair_text(name, address, name_original: str = "",
               script: str = "latin") -> str:
    """Build 'name | address' with original+romanized when non-Latin (§6.1)."""
    name = name if isinstance(name, str) else ""
    address = address if isinstance(address, str) else ""
    name_original = name_original if isinstance(name_original, str) else ""
    if script != "latin" and name_original and name_original != name:
        name_field = f"{name_original} ({name})"
    else:
        name_field = name
    return f"{name_field} | {address}".strip(" |")


def _build_training_pairs():
    """Return dataframe with columns [text_a, text_b, label] for R group."""
    split_df = splits_mod.load()
    r_ids = set(split_df.loc[split_df["group"] == "R", "entity_id"])
    gt = explode_ground_truth(read_ground_truth(C.TRAIN_GT))
    gt_r = gt[gt["source1_entity_id"].isin(r_ids)]
    true_pairs = set(zip(gt_r["source1_entity_id"], gt_r["matched_id"]))

    parts = []
    for p in C.PREFILTER_DIR.glob("train__*.parquet"):
        country = p.stem.split("__")[-1]
        cand = pd.read_parquet(p)
        cand = cand[cand["s1_id"].isin(r_ids)].copy()
        if cand.empty:
            continue
        cand["_country"] = country
        parts.append(cand)
    if not parts:
        raise RuntimeError("No R-partition prefilter rows found.")
    cands = pd.concat(parts, ignore_index=True)
    cands["label"] = [(s, c) in true_pairs for s, c in
                       zip(cands["s1_id"], cands["cand_id"])]

    # Build text lookups per country
    text_rows: list[dict] = []
    for country, sub in cands.groupby("_country"):
        s1 = pd.read_parquet(
            C.NORMALIZED_DIR / f"train__s1__{country}.parquet",
            columns=["entity_id", "core_name", "name_roman", "name_script",
                     "address_expanded"],
        ).set_index("entity_id")
        pools = {}
        for src in ("s2", "s3"):
            pfile = C.NORMALIZED_DIR / f"train__{src}__{country}.parquet"
            if pfile.exists():
                pools[src] = pd.read_parquet(
                    pfile,
                    columns=["entity_id", "core_name", "name_roman",
                             "name_script", "address_expanded"],
                ).set_index("entity_id")

        for row in sub.itertuples():
            sinfo = s1.loc[row.s1_id]
            src_key = row.cand_source.lower()
            if src_key not in pools:
                continue
            pool = pools[src_key]
            if row.cand_id not in pool.index:
                continue
            cinfo = pool.loc[row.cand_id]
            text_a = _pair_text(
                sinfo["core_name"] or sinfo["name_roman"],
                sinfo["address_expanded"] or "",
                name_original=sinfo["name_roman"],
                script=sinfo["name_script"],
            )
            text_b = _pair_text(
                cinfo["core_name"] or cinfo["name_roman"],
                cinfo["address_expanded"] or "",
                name_original=cinfo["name_roman"],
                script=cinfo["name_script"],
            )
            text_rows.append({
                "s1_id": row.s1_id, "cand_id": row.cand_id,
                "text_a": text_a, "text_b": text_b,
                "label": int(row.label),
                "stage_a": float(row.stage_a_score),
            })
    return pd.DataFrame(text_rows)


def _sample_hard(df: pd.DataFrame) -> pd.DataFrame:
    """Per S1: keep every positive; then up to RERANK_MAX_NEG_PER_POS hardest
    (highest Stage-A) non-positives."""
    kept_rows = []
    for s1_id, sub in df.groupby("s1_id"):
        pos = sub[sub["label"] == 1]
        neg = sub[sub["label"] == 0].sort_values("stage_a", ascending=False)
        k = max(1, len(pos)) * C.RERANK_MAX_NEG_PER_POS
        kept_rows.append(pos)
        kept_rows.append(neg.head(k))
    return pd.concat(kept_rows, ignore_index=True)


def _add_chain_negatives(df: pd.DataFrame, per_s1: int = 2) -> pd.DataFrame:
    """For each S1 with a non-empty core_name, add up to `per_s1` records
    whose core_name matches exactly but are not true matches (chain confusion).
    Uses the same country's S2/S3 pool. Adds only when there are candidates
    already present for that country.
    """
    if not C.RERANK_INCLUDE_CHAIN_NEGS:
        return df
    extras: list[dict] = []
    rng = random.Random(C.SEED)
    countries = sorted({p.stem.split("__")[-1] for p in
                        C.NORMALIZED_DIR.glob("train__s1__*.parquet")})
    for country in countries:
        s1 = pd.read_parquet(
            C.NORMALIZED_DIR / f"train__s1__{country}.parquet",
            columns=["entity_id", "core_name", "name_roman", "name_script",
                     "address_expanded"],
        ).set_index("entity_id")
        s1_r_ids = [i for i in df["s1_id"].unique() if i in s1.index]
        pool_records: dict[str, list[tuple]] = {}
        for src in ("s2", "s3"):
            pf = C.NORMALIZED_DIR / f"train__{src}__{country}.parquet"
            if not pf.exists():
                continue
            pd_ = pd.read_parquet(pf, columns=["entity_id", "core_name",
                                                 "name_roman", "name_script",
                                                 "address_expanded"])
            for eid, cn, nr, ns, ae in zip(pd_["entity_id"], pd_["core_name"],
                                            pd_["name_roman"], pd_["name_script"],
                                            pd_["address_expanded"]):
                pool_records.setdefault(cn or "", []).append(
                    (eid, cn, nr, ns, ae)
                )
        gt_pairs = set(zip(df["s1_id"], df["cand_id"]))
        for sid in s1_r_ids:
            sinfo = s1.loc[sid]
            same = pool_records.get(sinfo["core_name"] or "", [])
            random.shuffle(same)
            added = 0
            for eid, cn, nr, ns, ae in same:
                if (sid, eid) in gt_pairs:
                    continue
                extras.append({
                    "s1_id": sid, "cand_id": eid,
                    "text_a": _pair_text(sinfo["core_name"] or sinfo["name_roman"],
                                          sinfo["address_expanded"] or "",
                                          sinfo["name_roman"], sinfo["name_script"]),
                    "text_b": _pair_text(cn or nr, ae or "", nr, ns),
                    "label": 0, "stage_a": 0.5,
                })
                added += 1
                if added >= per_s1:
                    break
    if not extras:
        return df
    return pd.concat([df, pd.DataFrame(extras)], ignore_index=True)


def train() -> Path:
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)
    import torch

    print("[reranker_train] building training pairs from R group…")
    df = _build_training_pairs()
    print(f"[reranker_train] raw pairs: {len(df):,}  positives: {int(df['label'].sum()):,}")
    df = _sample_hard(df)
    df = _add_chain_negatives(df)
    print(f"[reranker_train] after sampling+chain-neg: {len(df):,}  "
          f"positives: {int(df['label'].sum()):,}")

    print(f"[reranker_train] torch device: {C.torch_device()}  "
          f"(fp16={'yes' if torch.cuda.is_available() else 'no'})")
    _reranker_src = C.RERANKER_MODEL_LOCAL or C.RERANKER_MODEL
    tok = AutoTokenizer.from_pretrained(_reranker_src, local_files_only=bool(C.RERANKER_MODEL_LOCAL))
    model = AutoModelForSequenceClassification.from_pretrained(
        _reranker_src, num_labels=1,
        local_files_only=bool(C.RERANKER_MODEL_LOCAL),
    )

    class PairDataset(torch.utils.data.Dataset):
        def __init__(self, df):
            self.a = df["text_a"].tolist()
            self.b = df["text_b"].tolist()
            self.y = df["label"].astype(np.float32).values

        def __len__(self):
            return len(self.a)

        def __getitem__(self, i):
            enc = tok(self.a[i], self.b[i], truncation=True,
                      max_length=C.RERANK_MAX_TOKENS, padding=False,
                      return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(self.y[i], dtype=torch.float32)
            return item

    def collate(batch):
        keys = [k for k in batch[0] if k != "labels"]
        out = tok.pad({k: [b[k] for b in batch] for k in keys},
                      return_tensors="pt")
        out["labels"] = torch.stack([b["labels"] for b in batch])
        return out

    class BCEWithLogitsTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels
            )
            return (loss, outputs) if return_outputs else loss

    out_dir = C.MODELS_DIR / "bge_reranker_ft"
    n_steps = max(1, len(df) // C.RERANK_TRAIN_BATCH_SIZE) * C.RERANK_EPOCHS
    warmup_steps = max(1, int(n_steps * C.RERANK_WARMUP_RATIO))
    _ta_kwargs: dict = dict(
        output_dir=str(out_dir),
        num_train_epochs=C.RERANK_EPOCHS,
        per_device_train_batch_size=C.RERANK_TRAIN_BATCH_SIZE,
        learning_rate=C.RERANK_LR,
        weight_decay=C.RERANK_WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=C.RERANK_GRAD_ACCUM,
        fp16=torch.cuda.is_available(),
        logging_steps=200,
        save_strategy="no",
        report_to=[],
        seed=C.SEED,
        dataloader_num_workers=2,
    )
    try:
        args = TrainingArguments(**_ta_kwargs)
    except TypeError:
        _ta_kwargs.pop("dataloader_num_workers", None)
        args = TrainingArguments(**_ta_kwargs)
    import inspect as _inspect
    _trainer_params = set(_inspect.signature(Trainer.__init__).parameters)
    _tok_kwarg = (
        {"processing_class": tok} if "processing_class" in _trainer_params
        else {"tokenizer": tok} if "tokenizer" in _trainer_params
        else {}
    )
    trainer = BCEWithLogitsTrainer(
        model=model, args=args, train_dataset=PairDataset(df),
        data_collator=collate, **_tok_kwarg,
    )
    trainer.train()
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[reranker_train] saved fine-tuned model to {out_dir}")
    return out_dir


if __name__ == "__main__":
    train()
