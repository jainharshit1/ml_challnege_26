# Business Entity Resolution — v3 pipeline (Team Titans)

End-to-end pipeline: raw TSVs → normalisation → six-arm blocking → Stage-A
LightGBM prefilter (→ `output/candidate_pairs.tsv`) → Stage-B LightGBM →
decision layer (→ `output/matching_results.tsv`). The methodology write-up is
`Documentation_template.md` at the root of the submission zip. Section
references like "plan §4.2" in code comments point to our internal design
plan; the write-up covers the same material.

Every stage is idempotent (skips completed files) so you can crash and resume.
All models, maps and statistics are learned **only** from the files provided
under `dataset/`; no external labels or lookups.

## Reproduce from the submission zip (step by step)

These steps regenerate both files in `output/` from the raw challenge data,
using only this folder. All paths are relative to the **zip root** (the folder
that contains `code/`, `output/` and `Documentation_template.md`).

1. **Place the challenge data** where `src/config.py` expects it:

   ```
   <zip root>/6ab10eb3b23ba_student_resource/student_resource/dataset/train/*.tsv
   <zip root>/6ab10eb3b23ba_student_resource/student_resource/dataset/test/*.tsv
   ```

   A symlink works too:
   `mkdir -p 6ab10eb3b23ba_student_resource && ln -s /path/to/student_resource 6ab10eb3b23ba_student_resource/student_resource`

2. **Create the environment** (Python 3.14.4 on Ubuntu, as in the submitted run):

   ```bash
   cd code/business_entity_resolution
   python -m venv .venv && source .venv/bin/activate
   pip install -U pip && pip install -r requirements.txt
   sudo apt-get install -y libgomp1        # OpenMP runtime for LightGBM
   ```

3. **Get the dense encoder weights.** `src/config.py` sets `DENSE_MODEL_LOCAL`
   to the local folder the submitted run used. On a new machine either set it
   to `None` (the public `BAAI/bge-m3` weights are then downloaded from the
   Hugging Face hub on first use), or download them once and point
   `DENSE_MODEL_LOCAL` at the folder:

   ```bash
   python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='models/bge-m3')"
   ```

   This is the only network access the pipeline needs. The weights are used
   unmodified (no fine-tuning).

4. **Start from a clean state.** Stages skip work whose output already exists,
   so remove any old `artifacts/` intermediates and `output/` files at the zip
   root before a from-scratch run. Keep `SUBSAMPLE_FRACTION = 1.0` and
   `NORMALIZE_ROW_CAP = None` in `src/config.py` (both are the defaults). Leave
   `HF_TOKEN` unset (see *Network access* below).

5. **Run the pipeline** with the exact settings of the submitted run:

   ```bash
   cd code/business_entity_resolution
   export N_JOBS=32 BLOCK_JOBS=3 PREFILTER_JOBS=5 FEATURES_JOBS=5
   python -m src.run_pipeline --skip reranker_train reranker_score loco
   ```

6. **Validate** (from the zip root):

   ```bash
   python3 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
       --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv \
       --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test \
       --check-ids
   ```

Outputs: `output/candidate_pairs.tsv` (written by stage `prefilter_score`) and
`output/matching_results.tsv` (written by stage `decide`). Validation scores,
tuned thresholds and blocking/prefilter recall are written to
`artifacts/reports/`. See *Reproducibility* at the end for what is and is not
bit-for-bit deterministic.

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
    └── run_pipeline.py     # end-to-end orchestration (entry point)
```

`scripts/` and `RESUME_INSTRUCTIONS.md`, if present, are infrastructure
helpers we used to checkpoint our own intermediate files between cloud
machines. They are not part of the modelling pipeline and are not needed to
reproduce the outputs.

Artifacts and outputs land in siblings of this folder (defined in `config.py`):

```
../../artifacts/         # intermediate parquets + models + reports
../../output/            # matching_results.tsv, candidate_pairs.tsv
```

## Notes on the submitted run

- Hardware: AWS m6i.8xlarge (32 vCPU, 128 GB RAM, **no GPU**). The pinned
  `faiss-gpu` wheel runs on CPU when no GPU is present.
- The cross-encoder stages (`reranker_train`, `reranker_score`) are skipped
  (CPU-only budget); Stage-B then uses the Stage-A score as its rerank signal.
  Running `python -m src.run_pipeline` with no `--skip` would try to fine-tune
  the reranker, which needs a GPU and is **not** what produced the outputs.
- `loco` is skipped; France uses the in-country thresholds.
- Wall-clock on that machine: embeddings ~3 h (computed once), blocking ~92 min,
  everything after blocking ~45 min. Per-stage times are in
  `artifacts/reports/stage_timings.json`.

Useful knobs (environment variables): `N_JOBS` (threads), `BLOCK_JOBS`
(partitions blocked in parallel), `PREFILTER_JOBS` / `FEATURES_JOBS`
(partition-level parallelism), `TFIDF_MAX_DF` / `TFIDF_MAX_DF_NAME` (TF-IDF
pruning), `UNION_TOP_N` (per-S1 candidate cap after blocking, default 40),
`BLOCK_TRAIN_GROUPS` (train S1 groups to block, default `G,V`). The submitted
run used the defaults for all of these except the four exported above.

## Development knob

`src/config.py`: `SUBSAMPLE_FRACTION: float = 1.0`. Lower it (e.g., `0.1`) to
iterate locally on 10% of G entities without disturbing V (kept intact so
evaluation numbers stay meaningful). **Must be 1.0 to reproduce the
submission.**

## Running individual stages

Resume from a stage:

```bash
python -m src.run_pipeline --from-stage prefilter_train
```

Run one stage in isolation (also useful when debugging):

```bash
python -m src.run_pipeline --only normalize_first idf
```

Skip stages (the submitted run skips the reranker and LOCO stages):

```bash
python -m src.run_pipeline --skip reranker_train reranker_score loco
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

## Network access and fair play

- **No external lookups.** Nothing in the pipeline queries an entity-resolution
  service, business registry, geocoder or any other external data source. All
  labels, abbreviation maps, IDF statistics and models come from the provided
  files.
- **Model download (once).** The public `BAAI/bge-m3` weights (MIT) are
  fetched from the Hugging Face hub unless `DENSE_MODEL_LOCAL` points to a local
  copy (step 3 above).
- **Optional checkpoint upload (outbound only).** `src/blocking.py` and
  `src/prefilter.py` can upload *our own* intermediate parquet files to a
  private Hugging Face dataset repo, so a crashed cloud machine could resume.
  This runs only when the `HF_TOKEN` environment variable is set, is
  write-only, and nothing uploaded is ever read back into the modelling code.
  Leave `HF_TOKEN` unset for a reproduction run; the pipeline's results do not
  depend on it.

## Reproducibility

**Seeds.** One seed, `config.SEED = 42`, drives every random choice:

| Where | What is seeded |
|---|---|
| `src/splits.py` | entity-level R / G / V split (stratified by country × match-count bucket) |
| `src/stage0_checks.py` | profiling sample |
| `src/blocking.py` | FAISS IVF training sample |
| `src/prefilter.py` | Stage-A LightGBM (`random_state`) |
| `src/train_gbdt.py` | entity-level early-stopping hold-out and Stage-B LightGBM (`seed`, incl. bagging and feature sub-sampling) |
| `src/reranker_train.py` | hard-negative sampling and trainer seed (stage skipped in the submitted run) |

The split file used for the submission is kept in the project repository as
`artifacts/splits/split.parquet` (next to the threshold, recall and timing
reports in `artifacts/reports/`), so the exact G / V entities can be checked
against a re-run of `splits`.

**Deterministic by construction:** normalisation, abbreviation mining, IDF,
TF-IDF blocking, key arms, feature computation (the vectorised features were
checked to be identical to a per-row reference implementation), threshold grid
(order-preserving parallel map, fixed tie-break) and the decision layer.

**Sources of small numerical variance.** LightGBM and FAISS use multithreaded
floating-point reductions, and the bge-m3 embeddings depend on the torch build
and CPU instruction set. To get the closest match to the submitted outputs, use
the pinned `requirements.txt`, the same thread settings
(`N_JOBS=32 BLOCK_JOBS=3 PREFILTER_JOBS=5 FEATURES_JOBS=5`) and a 32-vCPU
machine. On different hardware, expect at most tiny differences on pairs whose
score sits right at a threshold; the validation metrics in
`artifacts/reports/` should agree to within rounding.

**Other guarantees.**
- Every intermediate is a parquet, so any single stage can be re-run alone and
  inspected.
- Library versions are pinned in `requirements.txt`.
- The candidate file is the exact input of the final model: `prefilter_score`
  writes `output/candidate_pairs.tsv` from the same top-12 set that Stage-B
  scores, and the decision layer only removes pairs. Every ID in
  `matching_results.tsv` is therefore in `candidate_pairs.tsv`.
