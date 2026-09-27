# Business Entity Resolution — v3 pipeline

Implements the plan in `../../entity_resolution_strategy.md`. Every stage is
idempotent (skips completed files) so you can crash and resume. Fine-tuning
uses **only** data provided under `dataset/`; no external labels.

## Layout

```
code/business_entity_resolution/
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── config.py           # single source of truth for paths + hyperparams
    ├── io_utils.py         # TSV/parquet helpers, submission writer
    ├── stage0_checks.py    # §1.5 remaining checks → stage0_report.txt
    ├── splits.py           # §2 R/G/V/unused split file
    ├── abbreviations.py    # §3.4 seed + mined + unsupervised abbreviations
    ├── normalize.py        # §3.1–3.3 cleanup, romanize, keys, IDF
    ├── embed.py            # §4.3 bge-m3 fp16 embeddings per partition
    ├── blocking.py         # §4.2–4.4 hybrid arms A1–A6 + union top-40
    ├── prefilter.py        # §5 Stage-A LightGBM; writes candidate_pairs.tsv
    ├── reranker_train.py   # §6.2 fine-tune bge-reranker-v2-m3 on R
    ├── reranker_infer.py   # §6.3 score prefiltered pairs
    ├── features.py         # §7.1 Stage-B feature builder
    ├── train_gbdt.py       # §7.2–7.3 Stage-B GBDT + LOCO
    ├── decide.py           # §8.1 threshold → 1-to-1 → cap → singleton gate
    ├── tune_thresholds.py  # §8.3–8.4 coarse+fine grid, France procedure
    ├── metrics.py          # per-entity F0.5 + macro
    └── run_pipeline.py     # end-to-end orchestration
```

Artifacts and outputs land in siblings of this folder (defined in `config.py`):

```
../../artifacts/         # intermediate parquets + models + reports
../../output/            # matching_results.tsv, candidate_pairs.tsv
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` pins the exact environment of the submitted run (Python
3.14.4, Ubuntu). LightGBM additionally needs the system OpenMP runtime:
`sudo apt-get install -y libgomp1`. The pinned `faiss-gpu` wheel runs on CPU
when no GPU is present.

The dense encoder `BAAI/bge-m3` is loaded from `DENSE_MODEL_LOCAL` in
`src/config.py`; set it to `None` to fetch it from the Hugging Face hub once,
or point it to a local copy for fully offline runs.

## Reproduce the submitted run (Team Titans)

Data layout expected by `src/config.py` (paths are relative to the repository root):

```
6ab10eb3b23ba_student_resource/student_resource/dataset/{train,test}/*.tsv
```

Exact command and settings that produced `output/matching_results.tsv` and
`output/candidate_pairs.tsv` (AWS m6i.8xlarge, 32 vCPU, 128 GB RAM, no GPU):

```bash
cd code/business_entity_resolution
export N_JOBS=32 BLOCK_JOBS=3 PREFILTER_JOBS=5 FEATURES_JOBS=5
python -m src.run_pipeline --skip reranker_train reranker_score loco
```

- The cross-encoder stages are skipped (CPU-only budget); Stage-B then uses the
  Stage-A score as its rerank signal.
- `loco` is skipped; France uses the in-country thresholds.
- Wall-clock on that machine: embeddings ~3 h (computed once), blocking ~92 min,
  everything after blocking ~45 min.
- Validate before submitting (from `student_resource/`):
  `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids`

Useful knobs (environment variables): `N_JOBS` (threads), `BLOCK_JOBS`
(partitions blocked in parallel), `PREFILTER_JOBS` / `FEATURES_JOBS`
(partition-level parallelism), `TFIDF_MAX_DF` / `TFIDF_MAX_DF_NAME` (TF-IDF
pruning), `UNION_TOP_N` (per-S1 candidate cap after blocking, default 40),
`BLOCK_TRAIN_GROUPS` (train S1 groups to block, default `G,V`).

## The one knob you'll touch

`src/config.py` line 1: `SUBSAMPLE_FRACTION: float = 1.0`
Lower it (e.g., `0.1`) to iterate locally on 10% of G entities without
disturbing V (kept intact so evaluation numbers stay meaningful).

## How to run

Full pipeline:

```bash
cd code/business_entity_resolution
python -m src.run_pipeline
```

Resume from a stage:

```bash
python -m src.run_pipeline --from-stage prefilter_train
```

Run one stage in isolation (also useful when debugging):

```bash
python -m src.run_pipeline --only normalize_first idf
```

Skip a stage entirely (e.g., skip reranker fine-tuning and go zero-shot):

```bash
python -m src.run_pipeline --skip reranker_train
```

## Execution order — what depends on what

Stages MUST run in this order (later stages read earlier stages' parquets):

| # | Stage | Reads | Writes | GPU |
|---|-------|-------|--------|-----|
| 1 | `stage0` | raw TSVs | `artifacts/reports/stage0_report.{txt,json}` | no |
| 2 | `splits` | `train_source1.tsv`, `train_ground_truth.tsv` | `artifacts/splits/split.parquet` | no |
| 3 | `abbrev_seed` | — | `artifacts/abbreviations.json` (seed-only) | no |
| 4 | `normalize_first` | raw TSVs, seed abbreviations | `artifacts/normalized/{split}__{src}__{country}.parquet` | no |
| 5 | `abbrev_mine` | normalized S1/S2/S3, splits, GT | `artifacts/abbreviations.json` (mined) | no |
| 6 | `normalize_final` | raw TSVs, mined abbreviations | overwrites normalized parquets | no |
| 7 | `idf` | normalized parquets | `artifacts/idf/{split}__{country}.idf.parquet` | no |
| 8 | `embed` | normalized parquets | `artifacts/embeddings/{split}__{src}__{country}.fp16.npy` + ids | **yes** (recommended) |
| 9 | `block` | normalized + embeddings + splits + GT (for recall report) | `artifacts/blocking/{split}__{country}.parquet`, `artifacts/reports/blocking_recall.txt` | yes for A1 |
| 10 | `prefilter_train` | blocking (train, G) + normalized + GT | `artifacts/models/stage_a_lgbm.txt` | no |
| 11 | `prefilter_score` | blocking (all) + normalized + GT (V) | `artifacts/prefilter/*.parquet`, `output/candidate_pairs.tsv`, `artifacts/reports/prefilter_recall.txt` | no |
| 12 | `reranker_train` | prefilter (R) + normalized + GT | `artifacts/models/bge_reranker_ft/` | **yes** |
| 13 | `reranker_score` | prefilter (all) + normalized + fine-tuned model | `artifacts/reranker_scores/{split}__{country}.parquet` | **yes** |
| 14 | `features` | prefilter + reranker + normalized + IDF + GT (train) | `artifacts/final_scores/{split}__{country}__features.parquet` | no |
| 15 | `gbdt_train` | features (train, G) | `artifacts/models/stage_b_lgbm.txt` | no |
| 16 | `gbdt_score` | features (all) + Stage-B model | `artifacts/final_scores/{split}__{country}__scores.parquet` | no |
| 17 | `tune` | scores (train, V) + GT | `artifacts/reports/thresholds_in_country.json` + grid TSVs | no |
| 18 | `loco` *(optional)* | features (G, single country) | LOCO models + `thresholds___loco_{US,India}.json` | no (except reranker fine-tune per LOCO; see notes) |
| 19 | `france` | in-country + LOCO threshold JSONs + test France scores | `artifacts/reports/france_thresholds.json`, `france_diagnostics.json` | no |
| 20 | `decide` | test scores + threshold JSONs | `output/matching_results.tsv` | no |

**Important sequencing notes**

- `abbrev_seed → normalize_first → abbrev_mine → normalize_final` is a
  two-pass loop (plan §3.4): the first normalize pass exists purely so that
  the miner can run over cleaned tokens; the second pass then uses the full
  mined map. If you set `SUBSAMPLE_FRACTION` very low you can skip
  `abbrev_mine` and `normalize_final` to save time.
- `reranker_train` (stage 12) MUST come before `reranker_score` on any pair
  that will later feed Stage-B: the plan (§2) requires disjoint entity groups
  R vs G/V so the GBDT never sees pairs the reranker was trained on. This is
  enforced by the split file — do NOT run stage 15 (`gbdt_train`) on
  pre-fine-tune reranker scores, or you'll leak.
- For LOCO you need to re-run stages 12 and 13 with a **country-restricted
  R group** to avoid leakage (plan §2 last paragraph). This is not in the
  default `loco` stage; do it manually with `--only reranker_train` after
  filtering the R IDs, or use the zero-shot reranker for LOCO if compute is
  tight (the plan explicitly allows this).
- `output/candidate_pairs.tsv` is written by stage 11 and covers every test
  S1 (empty rows for empty prefilter output). `output/matching_results.tsv`
  is written by stage 20 and is what you upload to the leaderboard.
- Every ID in `matching_results.tsv` is guaranteed to be in
  `candidate_pairs.tsv` for that S1 (both files come from the same prefilter
  set filtered by the Stage-B model — the decision layer only removes pairs,
  never adds them).

## Validate the submission

```bash
python 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test
```

Add `--check-ids` for the memory-heavier ID-existence check.

## Rough runtime on a g5.4xlarge (plan §10)

Numbers are the plan's *estimates*; measure on 100K samples first.

| Stage | Time |
|-------|------|
| normalize | < 1 h |
| embed (bge-m3) | 2–5 h |
| blocking (all arms) | 2–4 h |
| prefilter train + score | 1–2 h |
| reranker fine-tune | 1–2 h |
| reranker inference | 5–8 h |
| stage-B features + train + score | 2–3 h |
| threshold tune + France + decide | < 1 h |

## Baseline shortcut for early submission (plan §11 step 7)

If you're short on GPU time, you can produce a full submission from stages
1–11 + 17 + 20 by treating the Stage-A score as the final score:

```bash
python -m src.run_pipeline --only stage0 splits abbrev_seed normalize_first \
    abbrev_mine normalize_final idf embed block prefilter_train prefilter_score
```

then hand-write a `matching_results.tsv` by running the decision layer on
`prefilter/*.parquet` with `stage_a_score` renamed to `p_match`. This gives
a valid leaderboard row while the reranker fine-tunes.

## Reproducibility

- All seeds via `config.SEED` (splits, FAISS training sample, LightGBM,
  reranker training).
- Every intermediate is a parquet — any single stage can be re-run alone.
- Model versions pinned in `requirements.txt`.
- No external data; academic-integrity rule is respected.
