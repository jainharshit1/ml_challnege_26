# Resume the pipeline on a new GPU instance

If the current 8-hour session on `ab_students@dslab` runs out or the machine is
lost, here's exactly how to pick up where it left off on a fresh Linux box with
an NVIDIA GPU.

## 0. What is where

| Location | Contents | Size |
|---|---|---|
| **GitHub** `jainharshit1/ml_challnege_26` | code, `artifacts/models/*.txt`, `artifacts/splits/`, `artifacts/reports/`, `artifacts/idf/`, `artifacts/abbreviations.json` | small |
| **HuggingFace** `jainsaabb/ml_challenge_2026_artifacts` (private dataset repo) | `embeddings/` (30 GB), `normalized/` (5 GB), `idf/`, `splits/` | ~55 GB |
| **NOT saved anywhere yet** | raw TSVs (~2 GB), `artifacts/blocking/`, `artifacts/prefilter/`, `artifacts/final_scores/`, `artifacts/reranker_scores/` | must regenerate or re-transfer |

The raw source TSVs (`6ab10eb3b23ba_student_resource/…/train/test_source{1,2,3}.tsv` +
`train_ground_truth.tsv`) are the ONLY thing you need to bring from an external
source — everything else is either in GitHub or HF Hub.

## 1. Bootstrap the new machine

Assume Ubuntu-like Linux with CUDA, Python 3.10+, ~60 GB free disk, ~100 GB RAM,
and outbound HTTPS to GitHub + HuggingFace.

```bash
# Base dir — adjust as needed
BASE=/opt/ml_challnege_26
mkdir -p "$BASE" && cd "$BASE"

# 1. Clone code + small artifacts
git clone https://github.com/jainharshit1/ml_challnege_26.git .

# 2. Python venv
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r code/business_entity_resolution/requirements.txt
# If requirements.txt is missing, install core deps manually:
# pip install pandas pyarrow numpy scikit-learn lightgbm faiss-cpu \
#   sentence-transformers transformers torch huggingface_hub \
#   joblib sparse_dot_topn rapidfuzz unidecode
```

## 2. Pull the heavy artifacts from HuggingFace

```bash
export HF_TOKEN=<paste_your_token>   # generate at huggingface.co/settings/tokens
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="jainsaabb/ml_challenge_2026_artifacts",
    repo_type="dataset",
    local_dir="artifacts",
    max_workers=8,
)
PY
```

Verify with the same script used to upload:

```bash
python code/business_entity_resolution/scripts/verify_hf.py
# Expect: ok=51 bad=0 missing=0
```

## 3. Bring the raw TSVs

The dataset isn't on HuggingFace or GitHub (competition data). Options in
descending preference:

1. **`scp` from the current Linux box** while it's still up:
   ```bash
   scp -r ab_students@<host>:/DATA/air_force_object_detection/ml_challnege_26/6ab10eb3b23ba_student_resource ./
   ```
2. **Re-download from the challenge portal** — the original zip is
   `6ab10eb3b23ba_student_resource.zip`.

Final layout must match `src/config.py`:
```
$BASE/6ab10eb3b23ba_student_resource/student_resource/dataset/{train,test}/*.tsv
```

## 4. Point the code at the local model weights

The bge-m3 + bge-reranker-v2-m3 weights are not on HF Hub (institutional proxy
was blocking HF at the time). Two ways:

- **scp from the current box**:
  ```bash
  scp -r ab_students@<host>:/DATA/air_force_object_detection/ml_challnege_26/models ./models
  ```
  Then update `src/config.py`:
  ```python
  DENSE_MODEL_LOCAL    = "/opt/ml_challnege_26/models/bge-m3"
  RERANKER_MODEL_LOCAL = "/opt/ml_challnege_26/models/bge-reranker-v2-m3"
  ```
- **Or re-download from HuggingFace** (if the new box has HF access):
  ```python
  DENSE_MODEL_LOCAL    = None   # will fall back to hub ID "BAAI/bge-m3"
  RERANKER_MODEL_LOCAL = None
  ```

## 5. Figure out which stage to resume from

Check `artifacts/reports/stage_timings.json` — the pipeline writes an entry when
each stage finishes. The **first stage NOT in that file** is where you resume.

Typical resume points from this session:

| Last completed stage in JSON | Resume with |
|---|---|
| `embed` | `--from-stage block --skip reranker_train reranker_score` |
| `block` | `--from-stage prefilter_train --skip reranker_train reranker_score` |
| `prefilter_score` | `--from-stage features --skip reranker_train reranker_score` |
| `gbdt_score` | `--from-stage tune` (reranker already skipped) |

Command:
```bash
cd code/business_entity_resolution
export N_JOBS=$(nproc)
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1   # if models are local
python -m src.run_pipeline --from-stage <STAGE> --skip reranker_train reranker_score \
    2>&1 | tee -a pipeline.log
```

## 6. Skipping the reranker in this leg

The reranker training/scoring stages are the most expensive (2-6 h on A5000)
and are being deferred to a future session. `--skip reranker_train reranker_score`
handles that. The downstream GBDT stage falls back to using
`stage_a_score` as the rerank feature when reranker output is absent — no code
changes required.

## 7. Sanity checks after resume

```bash
# Watcher (auto-commits each new completed stage)
tmux new-session -d -s watcher \
    'bash code/business_entity_resolution/scripts/watcher.sh'

# HuggingFace re-upload (in case you produce new artifacts)
# Reuse /tmp/hf_upload.py from the current session

# Pipeline progress
watch -n 30 'python -c "
import json
d = json.load(open(\"artifacts/reports/stage_timings.json\"))
for k,v in d.items(): print(f\"{k:20s} {v[\\\"seconds\\\"]:>8.1f}s\")
"'
```

## 8. Producing the final submission

The Amazon challenge wants `output/candidate_pairs.tsv` (from
`prefilter_score`) and `output/matching_results.tsv` (from `decide`). Both are
written under `output/` and are the two deliverables.

```bash
ls -la output/
# candidate_pairs.tsv    ← candidate pairs (Stage-A cascade)
# matching_results.tsv   ← final entity groups (post-decide)
```

## 9. Contact points

- HF repo: <https://huggingface.co/datasets/jainsaabb/ml_challenge_2026_artifacts>
- GitHub repo: <https://github.com/jainharshit1/ml_challnege_26>
- Model weights local path (current box): `/DATA/air_force_object_detection/ml_challnege_26/models/`
- Raw TSVs local path (current box): `/DATA/air_force_object_detection/ml_challnege_26/6ab10eb3b23ba_student_resource/`
