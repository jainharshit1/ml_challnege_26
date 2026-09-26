"""Dense embeddings (plan §4.3).

Encodes `dense_text = core_name (or name_roman) + " | " + address_expanded`
into unit-norm float16 vectors using BAAI/bge-m3. Writes one memmap + ids
parquet per (split, source, country) to keep RAM bounded and enable per-
partition FAISS.

Requires: torch (with CUDA), sentence-transformers or FlagEmbedding.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def _dense_text_series(df: pd.DataFrame) -> pd.Series:
    def _pick(row):
        name = row["core_name"] or row["name_roman"] or row.get("business_name", "")
        addr = row.get("address_expanded") or row.get("business_address", "")
        return f"{name} | {addr}".strip(" |")
    return df.apply(_pick, axis=1)


def _load_encoder(model_name: str | None = None):
    """Prefer FlagEmbedding for bge-m3; fall back to sentence-transformers."""
    if model_name is None:
        if C.USE_FALLBACK_DENSE:
            model_name = C.DENSE_MODEL_LOCAL or C.DENSE_MODEL_FALLBACK
        else:
            model_name = C.DENSE_MODEL_LOCAL or C.DENSE_MODEL
    print(f"[embed] torch device: {C.torch_device()}  model: {model_name}")
    # A5000/Ampere: enable TF32 for any fp32 fallback paths.
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore
        m = BGEM3FlagModel(model_name, use_fp16=C.DENSE_FP16)
        return ("flag", m)
    except Exception:
        from sentence_transformers import SentenceTransformer  # type: ignore
        device = C.torch_device()
        local = model_name.startswith("/") or model_name.startswith(".")
        m = SentenceTransformer(model_name, device=device,
                                local_files_only=local)
        if C.DENSE_FP16 and device == "cuda":
            try:
                m.half()
            except Exception:
                pass
        return ("st", m)


def _encode(encoder, texts: list[str], batch_size: int) -> np.ndarray:
    kind, m = encoder
    if kind == "flag":
        out = m.encode(texts, batch_size=batch_size, max_length=C.DENSE_MAX_TOKENS)
        vecs = out["dense_vecs"] if isinstance(out, dict) else out
    else:
        vecs = m.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=False, show_progress_bar=False,
        )
    vecs = np.asarray(vecs, dtype=np.float32)
    # unit-normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    return vecs.astype(np.float16)


def embed_partition(split: str, source: str, country: str,
                    encoder=None, batch_size: int | None = None) -> None:
    C.ensure_dirs()
    src_path = C.NORMALIZED_DIR / f"{split}__{source}__{country}.parquet"
    if not src_path.exists():
        print(f"[embed] SKIP missing {src_path}")
        return
    out_vec = C.EMBEDDINGS_DIR / f"{split}__{source}__{country}.fp16.npy"
    out_ids = C.EMBEDDINGS_DIR / f"{split}__{source}__{country}.ids.parquet"
    if out_vec.exists() and out_ids.exists():
        print(f"[embed] already exists: {out_vec.name}")
        return

    df = pd.read_parquet(src_path,
                         columns=["entity_id", "core_name", "name_roman",
                                  "address_expanded", "business_name",
                                  "business_address"])
    texts = _dense_text_series(df).tolist()
    n = len(texts)
    print(f"[embed] {split}/{source}/{country}: {n:,} rows")

    if encoder is None:
        encoder = _load_encoder()
    if batch_size is None:
        batch_size = C.DENSE_BATCH_SIZE

    vecs = np.zeros((n, C.DENSE_DIM), dtype=np.float16)
    step = max(batch_size * 32, 4096)
    for start in range(0, n, step):
        chunk = texts[start:start + step]
        vecs[start:start + len(chunk)] = _encode(encoder, chunk, batch_size)
        if (start // step) % 10 == 0:
            print(f"[embed]   {start+len(chunk):,}/{n:,}")
    np.save(out_vec, vecs)
    df[["entity_id"]].to_parquet(out_ids, index=False)
    print(f"[embed] wrote {out_vec}  {vecs.shape} fp16")


def embed_all() -> None:
    encoder = _load_encoder()
    for p in sorted(C.NORMALIZED_DIR.glob("*.parquet")):
        split, source, country = p.stem.split("__")
        embed_partition(split, source, country, encoder=encoder)


def load_embeddings(split: str, source: str, country: str
                    ) -> tuple[np.ndarray, pd.DataFrame]:
    vec = np.load(C.EMBEDDINGS_DIR / f"{split}__{source}__{country}.fp16.npy",
                  mmap_mode="r")
    ids = pd.read_parquet(C.EMBEDDINGS_DIR / f"{split}__{source}__{country}.ids.parquet")
    return vec, ids


if __name__ == "__main__":
    embed_all()
