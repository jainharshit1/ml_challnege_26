# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Titans
**Team Members:** Harshit Jain, Divyansh Jain, Shivam, Shreeya Deolia (Indian Institute of Technology Mandi)
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve every Source 1 business against ~10M Source 2/3 records with a **country-partitioned, multi-arm blocking → two-stage LightGBM → constrained decision layer** pipeline. Six complementary candidate-generation arms (multilingual dense retrieval, name and address TF-IDF, and three exact-key arms) define a high recall ceiling. A light Stage-A model cuts the union to 12 candidates per entity, and a 63-feature Stage-B model scores each pair. A decision layer then applies what we measured in the training data: **one-to-one assignment** (0 of 7.64M matched records belong to more than one S1 entity), a **per-entity singleton gate**, and **thresholds tuned directly for macro F<sub>0.5</sub>** on a held-out entity split. Country is never a model feature, so the unseen France partition runs through exactly the same code path as US and India.

---

## 2. Methodology

### 2.1 Problem Analysis

We profiled the full training data before designing anything (`src/stage0_checks.py`). The findings that shaped the design:

| Finding | Evidence (train) | Design consequence |
|---|---|---|
| Matches are many-to-one, not pairs | 2,206,821 S1 entities; **mean 3.46 matches**, median 3, max 11 | Per-S1 candidate budget must hold ≥ 11 true matches → Stage-A keeps 12, decision cap 12 |
| Singletons are rare but worth a full point each | **5.58 %** of S1 have no match | A dedicated singleton gate (separate threshold), not just a pair threshold |
| Every S2/S3 record matches at most one S1 | **0** of 7,638,365 matched IDs map to > 1 S1 | Hard one-to-one assignment at inference |
| Matches never cross countries | **0** cross-country pairs | Block and resolve per country partition (country used only as a partition key, never as a feature) |
| Matches split evenly across sources | 48.4 % S2 / 51.6 % S3 | Both pools blocked independently with equal budget |
| Postal codes are mostly absent | PIN/ZIP present on only ~11 % of US and **< 1.2 %** of India records | Postal code can't be a primary blocking key; locality tokens, house numbers and dense text do the work |
| House numbers are common | present on 49–78 % of records (88 % of France S1) | Dedicated (house number, street token) key arm |
| Non-Latin scripts in India | ~22 % of India S2 names are non-Latin (Devanagari 12.4 %, plus Kannada, Telugu, Tamil, Bengali, Gujarati, Malayalam, Gurmukhi) | Script detection + transliteration before any string comparison; multilingual dense model |
| Heavy surface noise | abbreviations (Pvt/Private, Rd/Road), legal-suffix drift, `&`/`and`, word transpositions, typos, landmark phrases ("near SBI ATM"), domain-style names | Normalisation layer with mined abbreviation maps, legal-suffix peeling, landmark extraction |
| Test adds an unseen country (France) | France present only in test | No country one-hot; thresholds sanity-checked with unsupervised France diagnostics |

### 2.2 Solution Strategy

**Approach Type:** Hybrid multi-arm blocking + cascaded gradient-boosted classifiers + constrained assignment.

**Core Innovations:**
1. **Six-arm, per-country blocking.** Arms are chosen for complementary failure modes: dense retrieval for transliteration and paraphrase, character TF-IDF for typos, word TF-IDF for addresses, and exact keys for anchors that survive heavy noise.
2. **Cascade sized from the label distribution.** The Stage-A prefilter keeps exactly the budget the match-count distribution requires (12 ≥ max 11), and **its output is the `candidate_pairs.tsv` we submit**, so the candidate file is precisely what the final model scores.
3. **The metric-aware decision layer is where F<sub>0.5</sub> is won.** Pair threshold, one-to-one assignment, per-S1 cap, singleton gate and an optional margin rule are applied in one fixed order that is shared between tuning and inference. They are tuned jointly for macro F<sub>0.5</sub> against the **full** validation ground truth, including matches that blocking missed, so the reported score is honest.
4. **Entity-disjoint data splits** (R / G / V), stratified by country × match-count bucket, keep every model and threshold free of leakage.

```
raw TSV ─► Stage 0 profiling ─► normalisation (+ mined abbreviations, IDF)
        ─► bge-m3 embeddings ─► per-country 6-arm blocking (≤ 40 cand / S1)
        ─► Stage-A LightGBM prefilter (top-12 / S1)  ──► output/candidate_pairs.tsv
        ─► Stage-B LightGBM (63 features)
        ─► decision layer (thresholds tuned on V for macro F0.5) ──► output/matching_results.tsv
```

### 2.3 Data splits

S1 entities (not pairs) are split once with a fixed seed and stratified by (country, match-count bucket ∈ {0, 1, 2–3, 4–5, 6+}), so singletons appear in every group (`src/splits.py`):

| Group | Share | Used for |
|---|---|---|
| R | 15 % | reserved for cross-encoder fine-tuning (see §4.4) |
| G | 25 % | training Stage-A and Stage-B |
| V | 10 % | threshold tuning and all reported validation scores |
| unused | 50 % | not needed; excluded from blocking to save compute |

---

## 3. Candidate Generation (Blocking)

### 3.1 Normalisation (`src/normalize.py`, `src/abbreviations.py`)

Every name and address passes through the same deterministic pipeline:

1. **Unicode NFKC**, then **script detection** (Latin, Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada, Malayalam, …).
2. **Transliteration** of Indic scripts to Latin (`indic_transliteration`, IAST), with `anyascii` as a fallback; a `romanization_ok` flag is kept as a feature.
3. Accent stripping, lower-casing, junk-prefix/suffix removal, `&`→`and`, `@`→`at`, punctuation and whitespace collapse, and de-duplication of repeated tokens.
4. **Names:** domain extraction (`abc.com` → `abc` + `name_domain`), abbreviation expansion, legal-suffix peeling (`core_name` vs `legal_suffix`), acronym, and a sorted-token key.
5. **Addresses:** landmark-phrase extraction ("near/opp./behind …"), abbreviation expansion, postal code (+ 3-digit prefix), house number, locality tokens, state code, street tokens, and all numbers.
6. **Abbreviation map in three layers:** a hand-curated seed (EN/IN/FR); abbreviations **mined from positive training pairs**, where a short token is a prefix or subsequence of an aligned long token, with count ≥ 5 and ≥ 70 % dominance; and an unsupervised per-country layer. Ambiguous short forms are dropped.
7. Per-country **token IDF** over S1 ∪ S2 ∪ S3 core names, used for rare-token keys and IDF-weighted features.

### 3.2 Blocking arms (`src/blocking.py`)

Blocking runs per (split, country) partition and independently against the S2 and S3 pools:

| Arm | Signal | Method | Per-S1 budget |
|---|---|---|---|
| **A1 Dense** | semantic / transliteration / paraphrase | `BAAI/bge-m3` embedding of `name | address` (1024-d, unit-norm, fp16) → FAISS `IVF-SQfp16` (2048 lists, nprobe 16); exact `FlatIP` for pools ≤ 200k | top-15 |
| **A2 Name TF-IDF** | typos, partial names | char_wb 3–4-grams, sublinear TF, `max_df` 0.01, top-k cosine with `sparse_dot_topn` (multithreaded C++, threshold 0.20) | top-10 |
| **A3 Address TF-IDF** | shared street / area vocabulary | word uni+bigrams, `max_df` 0.02, same top-k engine | top-10 |
| **A4 Rare-token key** | chains vs. branches | (rarest core-name token, locality token or postal code) | block cap 200 |
| **A5 House key** | exact address anchor | (house number, street token) | block cap 200 |
| **A6 Acronym key** | "TCS" ↔ "Tata Consultancy Services" | (acronym, postal prefix or locality token) | block cap 100 |

The arms are unioned per (S1, candidate). We keep each arm's score and rank plus `n_arms_hit`, then sort by (arms hit, best score) and cap at **40 candidates per S1**. Arm scores, ranks and hit flags become features downstream, so the model learns how much to trust each arm.

**Scaling decisions** (the full dataset is 2.2M × 10.3M train and 1.7M × 10.0M test records on a 32-vCPU CPU machine):
- **Dense arm:** fp16 scalar-quantised IVF codes are lossless for our fp16 embeddings and halve memory and bandwidth. The index is built and searched in batches, so peak memory is bounded.
- **TF-IDF arms:** `max_df` removes n-grams present in more than 1–2 % of records (e.g. "road", "nagar", city names). Those n-grams carry almost no IDF weight but dominate sparse matrix-multiplication cost. With them removed, the address arm got about 3× faster per pool (measured on India/S3).
- **One fit pass:** each vectoriser uses a single `fit_transform` over pool ∪ query instead of three passes.
- **Parallel partitions:** three partitions run concurrently, each on its share of the cores, so single-threaded phases overlap with multithreaded ones.

### 3.3 Stage-A prefilter → `candidate_pairs.tsv` (`src/prefilter.py`)

A 300-tree LightGBM trained on group G scores every blocked pair. It uses 9 cheap similarity features (Jaro-Winkler, token-sort and token-set ratio on names, character-trigram Jaccard, address token-set ratio, number and locality Jaccard, house and postal equality) plus all blocking-arm scores, ranks and flags. We keep the **top 12 per S1**. That budget is sized from the label distribution: 12 > the maximum of 11 true matches. **This set is written verbatim as `output/candidate_pairs.tsv`**, so every ID in `matching_results.tsv` is guaranteed to be a candidate. The decision layer only removes pairs, never adds them.

### 3.4 Candidate statistics

- **Candidate pairs generated (test, 1,732,544 S1 entities):** 69.30M after blocking (40 / S1: France 10.38M, India 32.40M, US 26.52M) → **20,790,528 in `candidate_pairs.tsv`** (12 / S1, every S1 has candidates) → 5,338,676 predicted matches
- **Reduction ratio vs. all within-country pairs:** 6.72 × 10¹² possible pairs → 2.08 × 10⁷ candidates, i.e. **99.99969 %** of the comparison space removed
- **Blocking recall on train G ∪ V** (fraction of true pairs present among the ≤ 40 candidates, counting S1s with zero candidates as misses): **US 0.9692** (1,553,016 / 1,602,386), **India 0.8880** (951,193 / 1,071,105), overall 0.9367
- **Prefilter recall on V** (after the top-12 cut): **0.9360**, against 0.9368 before the cut, so the prefilter costs only **0.08 points** of recall while shrinking the candidate set 3.3×

### 3.5 How we made sure true matches were not lost

- **Complementary arms:** each arm catches a different kind of noise, and the union is measured against ground truth on G ∪ V (`blocking_recall_report`) rather than assumed.
- **The recall metric can't be flattered:** S1 entities that received no candidates at all count as misses.
- **Candidate budgets come from data:** per-arm k and the union cap of 40 are far above the 3.46 mean matches, and the Stage-A cut of 12 exceeds the maximum observed match count.
- **Prefilter loss is monitored separately** on V (`prefilter_recall_report`), with a target loss below 0.5 points.
- **Oversized key blocks are dropped, not truncated,** so an over-generic key can't crowd out real candidates from other arms.

---

## 4. Matching Model

### 4.1 Stage-B features (`src/features.py`, 63 features)

**Name features:** Jaro-Winkler on core and expanded names; normalised Levenshtein similarity; token-sort, token-set and partial ratio; character-trigram Jaccard; word Jaccard; **IDF-weighted token Jaccard** (rare shared tokens count more than "shop" or "traders"); exact core-name and sorted-name equality; acronym match (including acronym ↔ initials of the other name); legal-suffix state (both missing / same / one missing / different); domain match; flags for abbreviation expansion, romanisation and script mismatch.

**Address features:** postal-code and postal-prefix equality; house-number equality; number-set Jaccard and shared-number count; IDF-weighted street-token Jaccard; locality overlap; address Jaro-Winkler and token-set ratio; landmark-token overlap; and missingness flags for address, postal code and house number on both sides. The missingness flags let the model tell "missing PIN code" apart from "different PIN code", which matters because PIN coverage is below 1.2 % in India.

**Model-score and provenance features:** **exact bge-m3 cosine for every candidate pair** (`f_dense_cos`, computed from the stored embeddings, so pairs found by the TF-IDF or key arms also get a semantic score, not only the dense arm's own hits), the dense arm's score, Stage-A score, rerank score, the per-arm ranks (dense, name TF-IDF, address TF-IDF), number of arms hit, and candidate source.

**Chain / frequency features:** candidate core-name frequency within the country, and its log (chain stores such as the same brand at different addresses are the main source of hard negatives), plus the S1 name's IDF mass.

**Competition features (listwise context):**
- *S1 side:* rank of the pair within its S1, the best / mean / second-best score for that S1, the gap to the best, the best-vs-second gap, and the number of strong candidates. The same statistics are computed separately for S2 and S3.
- *Candidate side:* how many S1 entities compete for the candidate, the best score this candidate gets from any S1, the gap to it, and the pair's rank among the competitors. These let the classifier anticipate the one-to-one constraint.

String similarities are computed with RapidFuzz's multithreaded element-wise `cpdist` over column arrays, and set-based features reuse cached tokenisations. We checked that this vectorised implementation produces **identical values for all 62 string and structural features** as the reference per-row implementation.

### 4.2 Model

- **Model type:** LightGBM binary classifier (MIT licence): 63 leaves, learning rate 0.1, feature and bagging fraction 0.8, early stopping on a 10 % **entity-level** hold-out of G, so pairs from one S1 never straddle train and validation.
- **Training data:** all Stage-A candidates of group G, labelled against the training ground truth.
- **Country is never a feature.** US, India and France are scored by the same model.

### 4.3 Decision layer and threshold selection (`src/decide.py`, `src/tune_thresholds.py`)

The same fixed order is used during tuning and at inference:

1. **Pair filter:** keep pairs with p ≥ `pair_thresh`.
2. **One-to-one assignment within country:** process pairs in descending p; a candidate is assigned to the first S1 that claims it, and later claims are dropped. This is justified by the exclusivity analysis (0 violations in 7.64M training matches).
3. **Per-S1 cap** of 12.
4. **Singleton gate:** if an S1's best remaining p is below `singleton_thresh`, its whole list is emptied. This is how we earn the full 1.0 on true singletons, and it's why there are two thresholds rather than one.
5. **Optional margin rule** for candidates contested by several S1 entities.

**Threshold selection:** a coarse grid (pair and singleton thresholds each 0.30–0.90 in steps of 0.05, margin ∈ {off, 0.05, 0.10}) followed by a fine grid (±0.05 in steps of 0.01) around the optimum. The objective is **macro F<sub>0.5</sub> over all V entities, singletons included, against the full ground truth**, implemented exactly as the challenge defines it (`src/metrics.py`). The grid runs in parallel across all cores and is order-preserving, so tie-breaking is deterministic.

**Why this suits a precision-weighted metric:** F<sub>0.5</sub> punishes a wrong merge four times as hard as a missed match, and a single false match on a true singleton zeroes that entity. The singleton gate lets the model be generous on entities with at least one confident match while abstaining on entities whose best candidate is only plausible. One-to-one assignment removes the typical chain-store false positive, where one branch gets attached to several S1 entities.

### 4.4 Cross-encoder reranker (implemented, not used in the final run)

The codebase includes a cross-encoder stage (`src/reranker_train.py`, `src/reranker_infer.py`): `BAAI/bge-reranker-v2-m3` (Apache-2.0) fine-tuned on group R with hard negatives (the highest-scoring Stage-A non-matches) and explicit chain-name negatives. Inference uses a cascade, sending only pairs with an uncertain Stage-A score through the cross-encoder. Our final compute budget was a CPU-only machine, where scoring millions of pairs with a 568M-parameter cross-encoder did not fit the time window. **For the submitted run the reranker stage is disabled** and the Stage-A score feeds the rerank-based competition features, which is the same substitution the cascade makes for confident pairs. Keeping group R entity-disjoint means the reranker can be switched back on without retraining anything else or introducing leakage.

### 4.5 Unseen country: France

- Country is treated as an open set of string labels. Partitions are discovered from the data, and nothing is hard-coded to {US, India}.
- No country feature exists, so the model's decisions rest on string and structural evidence that carries over across countries. The seed abbreviation list includes French forms (`ste`→`societe`, `ets`→`etablissements`, `cie`→`compagnie`), and normalisation strips accents.
- **Threshold transfer:** the code supports leave-one-country-out (LOCO) tuning, which trains on one country and tunes on the other to measure how much optimal thresholds move for an unseen country. For the final run we used the in-country thresholds (LOCO was not run within the compute budget; stale LOCO files are explicitly ignored), and checked France predictions with unsupervised diagnostics: **predicted singleton rate 5.33 % and 3.20 mean matches per entity** over 259,452 France S1 entities, against 5.58 % and 3.46 in the training labels. The close agreement suggests the thresholds carry over to the unseen country without systematic over- or under-matching.

---

## 5. Results & Error Analysis

All scores are on the entity-disjoint **V split** and computed with the official macro F<sub>0.5</sub> definition, against the full ground truth (blocking misses count against us).

| Metric | US | India | Overall |
|---|---|---|---|
| Blocking recall (G ∪ V, ≤ 40 cand.) | 0.9692 | 0.8880 | 0.9367 |
| Prefilter recall (V, top-12) | 0.9689 | 0.8868 | 0.9360 |
| **Macro F<sub>0.5</sub> (V)** | **0.9461** | **0.9026** | **0.9287** |
| Singleton F<sub>0.5</sub> (V) | 0.9245 | 0.8971 | 0.9135 |
| Non-singleton F<sub>0.5</sub> (V) | 0.9474 | 0.9029 | 0.9296 |
| Pair precision / pair recall (V) | 0.978 / 0.901 | 0.971 / 0.821 | |
| Predicted singleton rate / true | | | 6.90 % / 5.58 % |
| Mean predicted matches / true | | | 3.08 / 3.46 |
| **Public leaderboard F<sub>0.5</sub>** (test incl. France) | | | **0.914** |

**Tuned thresholds:** pair 0.66, singleton 0.66, margin 0.05. Stage-B early-stopped at iteration 92 (validation log-loss 0.0623, AUC 0.9968).

**Most important features (gain):** `f_stage_a`, `f_rerank` (Stage-A-derived rerank signal), `f_name_jw_exp`, `f_legal_state`, `f_cand_gap_to_best`, `f_rerank_gap_best`, `f_name_freq`, `f_cand_rank_among_competitors`, `f_addr_jw`, `f_cand_competitors`. Three of the top ten are **competition features** (candidate-side gap, rank and number of competitors), which confirms that listwise context, not just pairwise similarity, drives precision.

**Where the remaining error comes from** (share of true V pairs):

| | US | India |
|---|---|---|
| Lost at blocking (never a candidate) | 3.06 % | 11.19 % |
| Cut by the Stage-A top-12 | 0.05 % | 0.13 % |
| Rejected by the model / decision layer | 6.80 % | 6.55 % |

- **Common false positives (wrong merges):** pair precision is already high (0.978 US, 0.971 India). The dominant pattern is **chain confusion**: 32 % (US) and 18 % (India) of wrong merges have an *identical* core name, i.e. another branch of the same brand at a different address. In India a further 18 % involve transliterated (non-Latin) candidate names. Only 7–8 % of wrong merges land on true singletons, and 7.6 % (US) / 10.3 % (India) of true singletons receive a spurious match.
- **Common false negatives (missed matches):** India's gap to the US is almost entirely **blocking recall** (11.2 % of true pairs never become candidates, against 3.1 %). We diagnosed these misses directly: **every missed pair belongs to an S1 whose 40 union slots were full**, and the missed candidates are the ones only a single arm finds. In India, 54 % of them have a **non-Latin (transliterated) name**, so only the dense arm can see them, and their dense cosine is lower (median 0.70 vs 0.92 for found pairs). In the US, 54 % have an **empty address**, so the address arm cannot see them. These S1 entities also have more true matches (4.1 vs 3.7), so the per-S1 cap binds exactly where it hurts. Model-side misses (about 6.7 %) are mostly conservative rejections from the precision-weighted thresholds: 20–28 % of them share the exact core name with the S1 and are rejected because a same-name chain branch is equally plausible, which is the right trade under F<sub>0.5</sub>.

---

## 6. Conclusion

Our pipeline reaches **0.929 macro F<sub>0.5</sub> on a held-out validation split and 0.914 on the public leaderboard**. It runs end to end on a single CPU machine, with 99.9997 % of the comparison space removed at 0.08 points of prefilter recall loss, and it generalises to the unseen France partition with singleton and match-count statistics that closely track the training labels. The error analysis points clearly at the next gain: a larger or arm-reserved candidate budget for single-arm hits (transliterated names, empty addresses), since blocking, not the matcher, bounds India. The central lessons: **recall is decided at blocking** (every arm covers a different kind of noise, so the union matters more than any one arm); **F<sub>0.5</sub> is decided at the decision layer** (the one-to-one and singleton structure measured in the training labels is worth more than extra model capacity); and **engineering for scale** (sparse top-k in C++, quantised ANN, vectorised features, parallel partitions) is what made a 10M-record, three-country problem fit on a single CPU machine within the time limit.

---

## 7. Rule Compliance

- **Model licences and size:** the final classifier is **LightGBM (MIT)**, a gradient-boosted tree ensemble with far fewer than 8 billion parameters. The only neural model in the submitted pipeline is **`BAAI/bge-m3` (MIT, ~568M parameters)**, used to produce the embeddings for dense blocking. The optional reranker `BAAI/bge-reranker-v2-m3` is Apache-2.0 (~568M parameters) and was not used in the final run. Supporting libraries: FAISS (MIT), scikit-learn (BSD-3), sparse_dot_topn (Apache-2.0), RapidFuzz (MIT), indic-transliteration (MIT), anyascii (ISC).
- **Fair play:** no external entity-resolution APIs, business registries, government databases, geocoding services, or internet data were used. All labels, abbreviation maps, IDF statistics and models are derived solely from the provided training and test files; the test files are used only unsupervised (normalisation, per-country IDF, blocking), since they carry no labels. The pre-trained embedding model is used as-is, with no fine-tuning.
- **Network access:** the pipeline's only inbound network use is a one-time download of the public `BAAI/bge-m3` weights (skipped when `DENSE_MODEL_LOCAL` points to a local copy). During our cloud runs, `src/blocking.py` and `src/prefilter.py` could also upload our **own** intermediate parquet files to a private Hugging Face dataset repo as crash-recovery checkpoints. That upload is outbound only, runs only when `HF_TOKEN` is set, and nothing uploaded is read back by the modelling code. No lookup of any business identity happens anywhere.
- **Output integrity:** `output/candidate_pairs.tsv` is written by the Stage-A prefilter from exactly the pairs Stage-B scores, and the decision layer only removes pairs, so every ID in `matching_results.tsv` appears in `candidate_pairs.tsv` for the same S1. Both files are checked with `utils/validate_submission.py --check-ids` before submission.

## 8. Reproducibility

- **Single entry point:** `python -m src.run_pipeline --skip reranker_train reranker_score loco`, run with `N_JOBS=32 BLOCK_JOBS=3 PREFILTER_JOBS=5 FEATURES_JOBS=5`, regenerates both output files from the raw TSVs. Step-by-step instructions (data placement, environment, model weights, clean start, validation) are in `code/business_entity_resolution/README.md`.
- **One seed everywhere:** `config.SEED = 42` fixes the R / G / V split, the profiling sample, the FAISS IVF training sample, the Stage-A LightGBM, the Stage-B early-stopping hold-out and Stage-B LightGBM (bagging and feature sub-sampling), and the reranker's negative sampling.
- **Deterministic stages:** normalisation, abbreviation mining, IDF, TF-IDF and key blocking, feature computation, the threshold grid (order-preserving parallel map with fixed tie-breaking) and the decision layer.
- **Leakage control:** R, G and V are entity-disjoint; Stage-A and Stage-B train on G only, thresholds are tuned on V only, and Stage-B's early stopping uses an entity-level hold-out inside G.
- **Pinned environment:** `requirements.txt` pins every library version of the submitted run (Python 3.14.4, Ubuntu, AWS m6i.8xlarge, 32 vCPU, no GPU).
- **Known variance:** LightGBM and FAISS use multithreaded floating-point reductions and the embeddings depend on the torch build, so a run on different hardware or thread settings can differ slightly on pairs whose score sits exactly at a threshold. The pinned environment and the same thread settings give the closest reproduction of the submitted outputs.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` is a self-contained, runnable copy of the pipeline (see its `README.md`; dependencies pinned in `requirements.txt`).

| Module | Role |
|---|---|
| `src/run_pipeline.py` | **Entry point**; runs all stages in dependency order and is idempotent (finished parquets are reused) |
| `src/stage0_checks.py` | data profiling (§2.1) |
| `src/splits.py` | entity-level R / G / V split |
| `src/abbreviations.py`, `src/normalize.py` | abbreviation mining, normalisation, IDF |
| `src/embed.py` | bge-m3 embeddings (fp16 memmaps + id files) |
| `src/blocking.py` | six-arm per-country blocking + recall report |
| `src/prefilter.py` | Stage-A LightGBM; writes `output/candidate_pairs.tsv` |
| `src/reranker_train.py`, `src/reranker_infer.py` | optional cross-encoder stage |
| `src/features.py` | Stage-B features (63) |
| `src/train_gbdt.py` | Stage-B LightGBM training and scoring (+ LOCO variants) |
| `src/tune_thresholds.py`, `src/metrics.py` | macro F<sub>0.5</sub> and threshold search on V, France procedure |
| `src/decide.py` | decision layer; writes `output/matching_results.tsv` |

**Reproduce end to end** (from the zip root, with the challenge data at `6ab10eb3b23ba_student_resource/student_resource/dataset/`; full steps in the code `README.md`):
```bash
cd code/business_entity_resolution
pip install -r requirements.txt
export N_JOBS=32 BLOCK_JOBS=3 PREFILTER_JOBS=5 FEATURES_JOBS=5
python -m src.run_pipeline --skip reranker_train reranker_score loco
cd ../..
python3 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test --check-ids
```
Key environment knobs: `N_JOBS` (CPU threads), `BLOCK_JOBS` (partitions blocked in parallel), `PREFILTER_JOBS` and `FEATURES_JOBS` (partition-level parallelism), `TFIDF_MAX_DF` / `TFIDF_MAX_DF_NAME` (TF-IDF pruning).

### B. Additional Results

**Stage wall-clock times** on an AWS m6i.8xlarge (32 vCPU, 128 GB RAM, no GPU), from `artifacts/reports/stage_timings.json`:

| Stage | Time |
|---|---|
| normalize (two passes) + abbreviation mining | 11.7 + 11.7 + 24.2 min |
| embed (bge-m3, computed once and reused) | 178 min |
| block (5 partitions, 3 in parallel) | 91.7 min |
| prefilter train / score | 6.4 / 8.5 min |
| features (63, 5 partitions in parallel) | 11.1 min |
| gbdt train / score | 15.0 / 0.3 min |
| tune + france + decide | 1.6 + 0.1 + 0.6 min |

**Engineering note on the address arm** (top-k time per pool, measured on India):

| Configuration | Partition | Threads | Top-k time |
|---|---|---|---|
| char 3–4-grams, no pruning | 441k × 2.02M | 32 | > 40 min |
| char 3–4-grams, `max_df` 0.02 | 441k × 2.12M | 32 | 14.2 min |
| **word 1–2-grams, `max_df` 0.02 (final)** | 309k × 2.02M | 11 | **3.5 min** |

The dense arm's fp16 IVF index trains in about 45 s per pool.
