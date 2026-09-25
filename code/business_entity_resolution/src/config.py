"""Central configuration: paths, seeds, thresholds, model IDs, K values.

Every downstream module reads from here. Change training scope with
SUBSAMPLE_FRACTION (below); every other knob is grouped by pipeline stage.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# DEV KNOB — set below 1.0 to speed up local iteration. Plan section 2.
# 1.0  = train on the full G split (production)
# 0.1  = fast dev; keeps candidate density identical by also downsampling
#        the unmatched S2/S3 distractors by the same fraction.
# ---------------------------------------------------------------------------
SUBSAMPLE_FRACTION: float = 1.0

SEED: int = 42

# ----------------------------- paths ----------------------------------------
# working dir = D:\Opportunity\ml challenge  (parents[3] from src/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "6ab10eb3b23ba_student_resource" / "student_resource"
TRAIN_DIR = DATA_ROOT / "dataset" / "train"
TEST_DIR = DATA_ROOT / "dataset" / "test"

TRAIN_SOURCE = {
    "s1": TRAIN_DIR / "train_source1.tsv",
    "s2": TRAIN_DIR / "train_source2.tsv",
    "s3": TRAIN_DIR / "train_source3.tsv",
}
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"
TEST_SOURCE = {
    "s1": TEST_DIR / "test_source1.tsv",
    "s2": TEST_DIR / "test_source2.tsv",
    "s3": TEST_DIR / "test_source3.tsv",
}

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR = PROJECT_ROOT / "output"

NORMALIZED_DIR = ARTIFACT_DIR / "normalized"
EMBEDDINGS_DIR = ARTIFACT_DIR / "embeddings"
BLOCKING_DIR = ARTIFACT_DIR / "blocking"
PREFILTER_DIR = ARTIFACT_DIR / "prefilter"
RERANKER_DIR = ARTIFACT_DIR / "reranker_scores"
FINAL_SCORES_DIR = ARTIFACT_DIR / "final_scores"
MODELS_DIR = ARTIFACT_DIR / "models"
SPLITS_DIR = ARTIFACT_DIR / "splits"
REPORTS_DIR = ARTIFACT_DIR / "reports"
IDF_DIR = ARTIFACT_DIR / "idf"

ABBREV_PATH = ARTIFACT_DIR / "abbreviations.json"
SPLIT_PATH = SPLITS_DIR / "split.parquet"
STAGE0_REPORT = REPORTS_DIR / "stage0_report.txt"

# ----------------------------- splits (§2) ----------------------------------
SPLIT_R: float = 0.15   # reranker fine-tune
SPLIT_G: float = 0.25   # GBDT training
SPLIT_V: float = 0.10   # threshold tuning / final validation
# remaining ~50% left unused; reserved for extra runs

MATCH_COUNT_BUCKETS = [(0, 0), (1, 1), (2, 3), (4, 5), (6, 99)]

# ----------------------------- models ---------------------------------------
DENSE_MODEL = "BAAI/bge-m3"
DENSE_MODEL_FALLBACK = "intfloat/multilingual-e5-base"
DENSE_DIM = 1024                 # bge-m3; e5-base = 768
DENSE_MAX_TOKENS = 64
DENSE_BATCH_SIZE = 128           # tune per GPU
DENSE_FP16 = True

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_MAX_TOKENS = 128
RERANK_INFER_BATCH_SIZE = 128
RERANK_TRAIN_BATCH_SIZE = 32

# ----------------------------- blocking (§4) --------------------------------
K_DENSE = 15
K_NAME_TFIDF = 10
K_ADDR_TFIDF = 10
KEY_BLOCK_CAP = 200
KEY_ACRONYM_CAP = 100
UNION_TOP_N = 40
ENABLE_PHONETIC_ARM = False       # A7; enable only if recall < target

# TF-IDF
TFIDF_NGRAM = (3, 4)
TFIDF_MIN_DF = 2
TFIDF_MAX_FEATURES = 400_000
TFIDF_ROW_BATCH = 20_000           # sparse top-k row batch

# FAISS
FAISS_NLIST = 4096
FAISS_TRAIN_SAMPLE = 200_000
FAISS_NPROBE_DEFAULT = 32
FAISS_NPROBE_TARGET_OVERLAP = 0.99

# Small partitions (like France) use IndexFlatIP; threshold in n_vectors:
FAISS_FLAT_MAX = 1_000_000

# ----------------------------- Stage-A prefilter (§5) -----------------------
PREFILTER_TOP_N = 12               # start; raise to 15-20 if recall loss > 0.5 pts
STAGE_A_NUM_LEAVES = 31
STAGE_A_N_ESTIMATORS = 300
STAGE_A_LR = 0.05

# ----------------------------- reranker training (§6.2) ---------------------
RERANK_LR = 2e-5
RERANK_EPOCHS = 1
RERANK_WARMUP_RATIO = 0.05
RERANK_MAX_NEG_PER_POS = 4
RERANK_INCLUDE_CHAIN_NEGS = True
RERANK_WEIGHT_DECAY = 0.01
RERANK_GRAD_ACCUM = 1

# ----------------------------- Stage-B GBDT (§7) ----------------------------
LGB_NUM_LEAVES = 63
LGB_N_ESTIMATORS = 2000
LGB_EARLY_STOP = 50
LGB_LR = 0.05
LGB_FEATURE_FRACTION = 0.8
LGB_BAGGING_FRACTION = 0.8

ENABLE_SUPPORT_FEATURES = False    # §7.1 last bullet (S2↔S3 agreement)

# ----------------------------- decision (§8) --------------------------------
PAIR_THRESH_DEFAULT = 0.50
SINGLETON_THRESH_DEFAULT = 0.70
MARGIN_THRESH_DEFAULT: float | None = None   # None = disabled
PER_S1_CAP = 12

# France sanity (§8.4)
FRANCE_MIN_THRESH_OFFSET = -0.10   # never go below in-country thresholds by more

# threshold grids
COARSE_PAIR_GRID = [round(0.30 + 0.05 * i, 2) for i in range(0, 13)]   # 0.30..0.90
COARSE_MARGIN_GRID = [None, 0.05, 0.10]
FINE_STEP = 0.01
FINE_HALFWIDTH = 0.05

# ----------------------------- I/O ------------------------------------------
CHUNK_SIZE = 500_000               # normalization chunking

# ---- Accelerator toggles (env-overridable; probed lazily) ------------------
# Set to "cpu" to force CPU regardless of what's installed.
import os as _os
LGB_DEVICE_OVERRIDE = _os.environ.get("LGB_DEVICE", "").lower()   # "gpu"|"cpu"|""

_LGB_DEVICE_CACHED: str | None = None


def lgb_device() -> str:
    """Return 'gpu' if the installed LightGBM was built with GPU support AND
    a training call succeeds; else 'cpu'. Result is cached.

    Override with env var LGB_DEVICE=cpu|gpu.
    """
    global _LGB_DEVICE_CACHED
    if _LGB_DEVICE_CACHED is not None:
        return _LGB_DEVICE_CACHED
    if LGB_DEVICE_OVERRIDE in ("cpu", "gpu"):
        _LGB_DEVICE_CACHED = LGB_DEVICE_OVERRIDE
        return _LGB_DEVICE_CACHED
    try:
        import numpy as _np
        import lightgbm as _lgb
        X = _np.random.rand(64, 4).astype(_np.float32)
        y = _np.random.randint(0, 2, 64)
        _lgb.train(
            {"objective": "binary", "device_type": "gpu",
             "verbose": -1, "num_leaves": 7},
            _lgb.Dataset(X, label=y),
            num_boost_round=1,
        )
        _LGB_DEVICE_CACHED = "gpu"
    except Exception as e:
        print(f"[config.lgb_device] GPU unavailable, using CPU  "
              f"({type(e).__name__}: {e})")
        _LGB_DEVICE_CACHED = "cpu"
    return _LGB_DEVICE_CACHED


def torch_device() -> str:
    """'cuda' if torch sees a CUDA device, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ----------------------------- helpers --------------------------------------
def ensure_dirs() -> None:
    for d in (
        ARTIFACT_DIR, OUTPUT_DIR, NORMALIZED_DIR, EMBEDDINGS_DIR,
        BLOCKING_DIR, PREFILTER_DIR, RERANKER_DIR, FINAL_SCORES_DIR,
        MODELS_DIR, SPLITS_DIR, REPORTS_DIR, IDF_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
