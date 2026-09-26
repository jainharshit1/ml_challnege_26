"""Benchmark DENSE_BATCH_SIZE for bge-m3 on your GPU.

Reports peak VRAM usage and throughput (embeddings/second) at increasing
batch sizes. Use to pick the highest batch that still gains throughput
without exceeding VRAM.
"""
import time
import torch
from sentence_transformers import SentenceTransformer

MODEL_PATH = "/DATA/air_force_object_detection/ml_challnege_26/models/bge-m3"
MAX_LENGTH = 64                    # matches C.DENSE_MAX_TOKENS
BATCHES_TO_TRY = [128, 256, 384, 512, 768, 1024, 1536, 2048]
N_BATCHES_PER_TEST = 8             # run 8× batch samples to get stable rate

def main():
    print(f"[bench] loading {MODEL_PATH}")
    m = SentenceTransformer(MODEL_PATH, device="cuda")
    m.max_seq_length = MAX_LENGTH
    m.half()

    # short realistic text — mimics business name | address (~40 chars)
    sample = "Global Enterprises Ltd | 42 Rajaji Nagar 1st Main, Bangalore"

    # Warmup so first-iter compile/kernel-cache doesn't skew numbers
    m.encode([sample] * 32, batch_size=32, show_progress_bar=False)
    torch.cuda.synchronize()

    print(f"\n{'batch':>6}  {'peak_vram_gb':>13}  {'emb_per_sec':>12}  {'status':>8}")
    print("-" * 50)
    last_rate = 0.0
    for bs in BATCHES_TO_TRY:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        n = bs * N_BATCHES_PER_TEST
        texts = [sample] * n
        try:
            torch.cuda.synchronize()
            t0 = time.time()
            m.encode(texts, batch_size=bs, show_progress_bar=False,
                     convert_to_numpy=True, normalize_embeddings=False)
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            rate = n / dt
            gain = (rate / last_rate - 1) * 100 if last_rate > 0 else float("inf")
            gain_s = f"+{gain:4.1f}%" if last_rate > 0 else "  ----"
            print(f"{bs:>6}  {peak_gb:>10.2f} GB  {rate:>8.0f}  {gain_s}")
            last_rate = rate
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>6}  {'---':>13}  {'OOM':>12}  --stop")
            break
        except Exception as e:
            print(f"{bs:>6}  {'---':>13}  {'ERR':>12}  {type(e).__name__}: {e}")
            break

if __name__ == "__main__":
    main()
