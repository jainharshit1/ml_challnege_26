# Business Entity Resolution — Implementation Plan (v3)

This plan is written so each stage can be turned directly into a module. Every stage lists its **inputs**, **outputs**, **steps**, **parameters to tune**, and **checks that must pass** before moving on.

---

## 0. Summary

**Pipeline:**
1. Rule-based normalization (per field, script-aware, with romanization of Indic scripts)
2. Country-partitioned hybrid blocking (dense + sparse + key-based arms) → union
3. Stage-A cheap pre-filter (light GBDT on string features) → top-N per S1 → **this is `candidate_pairs.tsv`**
4. Cross-encoder scoring of the pre-filtered pairs (bge-reranker-v2-m3, fine-tuned)
5. Stage-B final GBDT (all features + reranker score + competition features)
6. Decision layer: pair threshold → one-to-one assignment → singleton gate
7. Validation: in-country K-fold, leave-one-country-out (France proxy), per-country prediction diagnostics on test

**Models (all offline, each independently compliant; verify each model card and record license in the documentation):**

| Role | Model | License | Params |
|---|---|---|---|
| Dense blocking | BAAI/bge-m3 (fallback: intfloat/multilingual-e5-base) | MIT | ~568M (~278M) |
| Reranker | BAAI/bge-reranker-v2-m3 | Apache-2.0 | ~568M |
| Stage-A and Stage-B classifiers | LightGBM | MIT | n/a |

No LLM in the pipeline. Qwen2.5-7B was dropped: at ~24M records (train + test) a generative pass is infeasible.

**Library licenses:** the rule covers models, but check helper libraries too and prefer permissive ones (e.g. `rapidfuzz` MIT, `faiss` MIT, `sparse_dot_topn` MIT, `anyascii` ISC). Avoid `unidecode` (GPL) to be safe.

---

## 1. Confirmed data facts (Stage 0 results)

### 1.1 Sizes

| Source | Train total | Train US | Train India | Test total | Test US | Test India | Test France |
|---|---|---|---|---|---|---|---|
| S1 | 2,206,821 | 1,323,633 | 883,188 | 1,732,544 | 663,106 | 809,986 | 259,452 |
| S2 | 5,034,616 | 3,016,817 | 2,017,799 | 4,887,273 | 1,871,330 | 2,312,565 | 703,378 |
| S3 | 5,285,603 | 3,170,056 | 2,115,547 | 5,082,316 | 1,945,701 | 2,405,000 | 731,615 |

**Share of the test score by country** (macro average, so proportional to S1 count): India 46.8%, US 38.3%, **France 15.0%**.

**Candidate density** ((S2+S3)/S1) is similar everywhere: train US 4.67, train India 4.68; test US 5.76, India 5.83, France 5.50. France is not structurally different in density, so the expected match rate per S1 should be close to train (~3.46 mean). This is used later as a sanity check on France predictions.

### 1.2 Label facts

| Fact | Value | Consequence |
|---|---|---|
| Singletons | 123,247 / 2,206,821 = 5.58% | Worth ~0.056 of macro F0.5 if all right, 0 if any false positive on each. Dedicated singleton gate required. |
| Match count | median 3, mean 3.46, max 11 | Model must find several matches per S1. Cap predictions per S1 at ~12. |
| Exclusivity | 0 S2/S3 IDs matched to >1 S1 | One-to-one assignment is a **hard constraint**. |
| Cross-country matches | 0 | Country is a **hard partition** for blocking. |
| Unmatched S2/S3 | 10.32M records, 7.64M matched → ~2.68M are distractors (~26%) | Many S2/S3 records belong to no S1. Precision matters on the candidate side too. |

### 1.3 Country field

- 0 missing values in all six files (train and test).
- Labels are exactly `US`, `India`, `France`, consistent across sources. No label normalization needed.
- Still keep the code open-set: partitions are built from whatever distinct values exist, never from a hard-coded list. If a value is ever missing, route it to a fallback partition that compares against everything (defensive only; not currently triggered).

### 1.4 Noise patterns seen in the raw sample (drive normalization)

- **Mixed scripts per field, not per record.** Names in Devanagari or Tamil with addresses in Latin (e.g. Hindi name, English address). Script must be detected per field.
- **URLs and domains in names:** `earnosethroat.com`, `... | www.shivshakti.com`, `heassociates.com`.
- **Junk prefixes and separators:** leading `--`, pipes `|`.
- **Repeated tokens:** `Crestline Crestline Clean LP`, `Keystone Odyssey Odyssey LLC`, `LLC LLC`, `VIDYALAYA VIDYALAYA`.
- **Spacing typos inside words:** `PRIVATE  LIMITED`, `Tetlecommunication`.
- **Address typos:** `17RD STREET`, `CNROE`, `FTT MITCHELL`.
- **Component reordering:** `GREENSBORO, NC, 19 1/2 STARDUST TRAIL` (city before street).
- **Indian address tokens:** `H.NO`, `HN`, `KH NO.`, `PLOT NO`, `SHP NO`, `S. NO`, `SECTOR`, `FLOOR`, `OPP.`, `NEAR`, `LANDMARK NEAR`, region words like `(EAST)`.
- **US addresses often have no ZIP** in the sample (`105 ELM ST, MORGANTON, NC`). Postal-code-based signals may be weak for US; verify coverage (step 1.5).
- **Chain / generic names:** `Primary Care` (741 in S2+S3), `Physical Therapy`, `Urgent Care`, `Main Street`, `Board of`, `Department of`, single-token names like `SC`, `SI`, `Apex`, `Global`. Names alone cannot identify these.

### 1.5 Remaining Stage 0 checks (run before building normalization)

1. **Postal code coverage** per source × country: fraction of addresses containing a standalone 5-digit (US, France) or 6-digit (India) token. Decides whether postal code can be a blocking key or only a feature.
2. **House number coverage** per source × country: fraction of addresses with a leading or labelled number (`H.NO 204`, `105 ELM ST`).
3. **Script distribution** per source × country × field: share of names and addresses in Latin vs Devanagari vs Tamil vs other Unicode blocks. Also check the France test records for accent frequency.
4. **Name-only vs address-only duplication:** how many S2/S3 records within a country share an identical normalized name. Gives the size of the chain problem per country.
5. **Source asymmetry:** for true pairs, compare S1↔S2 vs S1↔S3 name similarity distributions. If one source is much noisier, the `candidate_source` feature matters.
6. **Matches split by source:** for each S1, how many matches come from S2 vs S3 (useful for sanity checks later).

Record all outputs in a `stage0_report.txt`; they also go in the documentation.

---

## 2. Data splits (decide once, before any modelling)

Because the reranker is fine-tuned and its score then feeds a GBDT, the GBDT must never be trained on pairs the reranker was trained on (its scores would be overconfident). Split **training S1 entities** into disjoint groups:

| Group | Share of train S1 | Purpose |
|---|---|---|
| **R** | ~15% | Fine-tune the reranker. Mine abbreviations from R ∪ G. |
| **G** | ~25% | Train Stage-A and Stage-B GBDTs. |
| **V** | ~10% | Threshold tuning and final validation (end-to-end macro F0.5). |
| unused | ~50% | Not needed; keep for extra runs if time allows. |

**Rules:**
- Split by S1 entity, stratified by country and by match count bucket (0, 1, 2–3, 4–5, 6+), so singletons are represented in every group.
- Blocking still runs against the **full** S2/S3 of that country, so the candidate density in train matches test.
- Save the split (list of S1 IDs per group) to disk with a fixed seed; every later stage reads it.

**Leave-one-country-out (LOCO) variant:** for the France-proxy experiment, R and G come from one country only and V from the other. This needs its own reranker fine-tune run (or the zero-shot reranker) to avoid leakage. Budget one extra reranker run for this.

**Train subsampling to control compute (optional):** if embedding all 10.3M train S2/S3 records is too slow, subsample per country with fraction f: take f of S1 entities, **all** of their true matches, and f of the unmatched S2/S3 distractors. This keeps the density identical to the full data. Only do this if throughput measurements (section 10) demand it.

---

## 3. Stage 1 — Normalization

**Input:** raw TSVs (train and test, all three sources).
**Output:** one Parquet file per source per split, containing raw fields plus all derived fields below. All later stages read these files.

### 3.1 Field-level cleanup (applies to both name and address)

In this order:
1. Unicode NFKC normalization (fixes full-width characters and compatibility forms).
2. Detect the script of the field by counting characters per Unicode block (Latin, Devanagari, Tamil, Bengali, Telugu, Gujarati, Kannada, Malayalam, Gurmukhi, Arabic, other). Store `name_script` and `address_script` as the majority block. Latin with accents counts as Latin.
3. **Romanize non-Latin fields** with a rule-based transliterator (`indic-transliteration` for Indic scripts, `anyascii` as a generic fallback). Store as `name_roman` / `address_roman`. Keep the original too. This lets string features work on Hindi or Tamil names (e.g. a Hindi name becomes roughly "ram marketing praivet limited"), with no model involved.
4. Strip accents (NFKD then drop combining marks), lowercase.
5. Remove junk: leading/trailing non-alphanumeric runs (`--`), pipes and the text after a pipe if it is a URL.
6. Replace `&` with ` and `, `@` with ` at `. Remove remaining punctuation except `/` and `-` inside numbers (keep `63/2275/7`, `570/13`, `E-1`).
7. Collapse whitespace.
8. **De-duplicate consecutive repeated tokens** (`odyssey odyssey` → `odyssey`, `llc llc` → `llc`). Store a flag `name_had_repeat`.

### 3.2 Name-specific processing

1. **URL/domain extraction:** detect tokens matching a domain pattern (`something.com/.in/.fr/.org/.net`, optional `www.`). Move to `name_domain`; strip the TLD and `www`. If the whole name is a domain (`earnosethroat.com`), also create `name_from_domain` by splitting the domain into words where possible (dictionary-free heuristic: keep as a single token; char n-gram features handle the rest).
2. **Abbreviation expansion** using the merged map (section 3.4). Produce `name_expanded`. Keep `name_clean` (unexpanded).
3. **Legal suffix separation:** strip trailing legal-form tokens from `name_expanded` into `legal_suffix` (sorted set of tokens, e.g. `private limited`, `llc`, `inc`, `corporation`, `lp`, `llp`, `sarl`, `sas`, `sa`). The suffix list is seeded by hand and extended by frequency: any token that appears as the last token in >0.1% of names in a country and is not a common word is a suffix candidate; review the top list manually once. Result: `core_name`.
4. **Stopword-like generic tokens:** compute per-country token document frequency over S1+S2+S3 names. Store the IDF of every token. Generic tokens (`group`, `care`, `the`, `of`, `services`, `company`) get low IDF automatically; no hand list.
5. **Derived keys:**
   - `acronym`: first letters of `core_name` tokens with IDF above a floor (skip `of`, `the`, `and`).
   - `rare_token`: the highest-IDF token in `core_name`.
   - `name_sorted`: tokens of `core_name` sorted alphabetically (handles word-order transpositions).
   - `phonetic_tokens`: Double Metaphone of each `core_name` token (optional; only used if blocking recall needs it).

### 3.3 Address-specific processing

1. **Abbreviation expansion** with the address map (`rd→road`, `st→street`, `h no/hn/hno→house number`, `opp→opposite`, `nr→near`, `av/ave→avenue`, `bd/blvd→boulevard`). Produce `address_expanded`.
2. **Landmark extraction:** remove phrases starting with `near`, `opposite`, `behind`, `beside`, `next to`, `in front of`, `landmark` up to the next comma. Store them in `landmark_text` (used only as a weak feature).
3. **Postal code:** find a standalone digit token of length 5 or 6 (optionally with a space inside, e.g. `400 001`), preferring the last one in the string. Store `postal_code` and `postal_prefix3`. This is generic, not country-coded, so it works for France and any future country. Country-specific validation (US 5 digits, India 6 digits starting 1–8) is allowed only as an optional refinement, with the generic rule as fallback.
4. **House / unit number:** capture numbers following labels (`house number`, `plot no`, `shop no`, `kh no`, `s no`, `flat`, `door`) or a leading number before a street word. Store `house_number` normalized (strip spaces, uppercase letters, keep `/` and `-`). Also store `all_numbers`: the set of every number token in the address (robust to reordering).
5. **Locality tokens:** split the address on commas; the last 1–3 non-numeric segments are likely city/district/state. Store `locality_tokens` (set). Also store `state_code` if a 2-letter token matches a US state or a known Indian state name (this is a helper, not a filter).
6. **Street tokens:** remaining tokens after removing numbers, locality, landmark and generic words (`road`, `street`, `floor`, `building`). Store `street_tokens` (set).
7. **Missing flags:** `address_missing`, `postal_missing`, `house_number_missing`, `locality_missing`.

### 3.4 Abbreviation map (three layers)

Built once, saved to `abbreviations.json`, loaded by normalization.

1. **Seed list:** ~40 common name and address abbreviations for English, Indian and French usage (`pvt`, `ltd`, `corp`, `inc`, `co`, `intl`, `mfg`, `assn`, `govt`, `dept`, `natl`, `cie`, `ste`, `ets`, `rd`, `st`, `ave`, `blvd`, `dr`, `ln`, `hwy`, `opp`, `nr`, `bd`, `av`, `ch`, `pl`).
2. **Mined from training pairs (R ∪ G only):** for every true pair, align tokens of the two cleaned names; wherever a token of length 2–5 is a prefix of, or a subsequence of, a longer token in the other name (e.g. `pvt`/`private`, `mfg`/`manufacturing`), count the pair. Keep pairs with count ≥ 5 and where the short form maps to one dominant long form (≥ 70% of its occurrences). Do the same for addresses.
3. **Mined from the test set, unsupervised (for France and new patterns):** within each country, pool all name tokens from S1+S2+S3 with their frequencies. A short token S is mapped to long token L if S is a prefix or subsequence of L, both have frequency ≥ 10, and they occur in the same slot (e.g. both as the last or first token) in names that otherwise share ≥ 1 rare token. Review the top 100 French pairs manually once before accepting.

**Conflict rule:** data-mined entries override seed entries; a short form mapped to several long forms is dropped (ambiguous).

**Robustness principle:** features are computed on both `name_clean` and `name_expanded`, so an incomplete map never breaks matching.

### 3.5 Checks before moving on

- Spot-check 50 random rows per country × source (including France) for each derived field.
- Coverage table: % non-empty for `postal_code`, `house_number`, `locality_tokens`, `core_name`, `name_roman` by country × source.
- Runtime: normalization of all files should run in well under an hour with multiprocessing (chunk by 500K rows).

---

## 4. Stage 2 — Country-partitioned hybrid blocking

**Input:** normalized Parquet files, the split file.
**Output:** `blocking_candidates.parquet` with one row per (S1, candidate) and columns: `s1_id`, `cand_id`, `cand_source`, the score from each arm (NaN if the arm did not produce the pair), and the rank within each arm.

Everything below runs **independently per country partition**, and separately for the S2 pool and the S3 pool.

### 4.1 Blocking text

- **Dense text:** `core_name (or name_roman) + " | " + address_expanded`. Including the address is essential: for chain names, a name-only embedding fills the top-K with same-name records at the wrong address.
- Truncate to ~64 tokens.

### 4.2 Arms

| # | Arm | Key / method | K or cap | Catches |
|---|---|---|---|---|
| A1 | Dense kNN | bge-m3 embeddings of dense text, inner product | K = 15 per pool | semantic, transliteration, reordering |
| A2 | Name char TF-IDF | char 3–4-grams of `core_name` (romanized), cosine top-K via sparse top-n multiplication | K = 10 per pool | typos, spacing errors |
| A3 | Address char TF-IDF | char 3–4-grams of `address_expanded` | K = 10 per pool | chains: same name, need exact location |
| A4 | Locality + rare name token | exact match on (`rare_token`, any `locality_token` or `postal_code`) | block size cap 200 | same business, same place, different phrasing |
| A5 | House number + street | exact match on (`house_number`, any `street_token`) | block size cap 200 | chains and generic names at a specific address |
| A6 | Acronym | exact match on (`acronym`, `postal_prefix3` or `locality_token`) | block size cap 100 | `SBI` ↔ `State Bank of India` |
| A7 | Phonetic (optional) | Double Metaphone of `rare_token` + locality | cap 100 | only if recall < target after A1–A6 |

**Block size caps:** for key arms (A4–A6), if a key produces more than the cap, drop that key (it is too generic to be informative). Log how many keys were dropped per arm.

### 4.3 Dense index choice per partition

- **France (S2 703K, S3 732K):** exact `IndexFlatIP` on GPU. No approximation loss.
- **US and India (1.9M–3.2M per pool):** `IndexIVFFlat`, `nlist ≈ 4096–8192`, trained on a random 200K sample with a fixed seed. Tune `nprobe` by comparing against an exact search on 10K sample queries; choose the smallest `nprobe` that reaches ≥ 99% overlap with exact top-15.
- Normalize embeddings (unit length) so inner product = cosine.
- Store embeddings as float16 memmaps, one file per source × country × split. Full precision for 24M × 1024 would be ~98 GB; float16 halves that, and processing per partition keeps RAM bounded. If disk or time becomes a problem, switch to multilingual-e5-base (768-d).

### 4.4 Union and first cap

1. Union all arms per (S1, pool). Keep every arm's score and rank as columns.
2. Compute `n_arms_hit` (how many arms produced this pair).
3. Keep up to **40 candidates per S1** (S2 + S3 combined), ranked by `n_arms_hit` then best normalized arm score.

### 4.5 Checks (on train, groups G ∪ V)

- **Blocking recall** = true pairs present in candidates / all true pairs. Target ≥ 97%.
- **Recall by arm** and **marginal recall** (pairs found only by that arm). Drop arms with near-zero marginal recall; add A7 if recall is short.
- **Recall by country, by source, and for chain names** (S1 names in the top-1% frequency).
- **Entity-level recall ceiling:** fraction of S1 entities whose full true set is in candidates. This is the real upper bound for macro F0.5.
- **Pairs per S1** distribution.
- Run the same pairs-per-S1 statistics on test, per country. France should look similar to US/India. A large difference means a normalization or blocking problem on French data.

---

## 5. Stage 3 — Stage-A pre-filter (defines `candidate_pairs.tsv`)

**Why:** 1.73M test S1 × 40 candidates ≈ 69M pairs is too many for a cross-encoder. A cheap model narrows it first.

**Input:** blocking candidates + normalized fields.
**Output:** `prefiltered_candidates.parquet` (top-N per S1) and **`output/candidate_pairs.tsv`**.

### Steps

1. Compute the **cheap feature set** for every blocked pair (all vectorized with `rapidfuzz` `cdist`/`process` utilities or batched):
   - name: Jaro-Winkler, token-sort ratio, token-set ratio, char-3gram Jaccard (on romanized `core_name`)
   - address: token-set ratio, `all_numbers` Jaccard, `house_number` exact match, `postal_code` exact match, locality overlap
   - blocking: dense cosine, each arm's rank, `n_arms_hit`
2. Train a small LightGBM (≈ 200 trees, 31 leaves) on group G pairs, label = pair is a true match.
3. Score all blocked pairs; keep the **top N per S1** (start N = 12; train max matches is 11).
4. **Check pre-filter recall on V:** true pairs surviving / true pairs in blocking. Must lose < 0.5 points of recall. If it loses more, raise N to 15–20.
5. Write `candidate_pairs.tsv` from the kept pairs: one row per test S1 (all 1,732,544), comma-joined candidate IDs, empty if none, no duplicates. This file must be exactly the set Stage 3 and Stage 4 score.

---

## 6. Stage 4 — Cross-encoder scoring

**Input:** pre-filtered pairs.
**Output:** `reranker_scores.parquet` with (`s1_id`, `cand_id`, `rerank_score`).

### 6.1 Input formatting

- Text A: `S1 name | S1 address`. Text B: `candidate name | candidate address`. Use the original script plus the romanized name when non-Latin (`राम मार्केटिंग ... (ram marketing ...)`), since the model is multilingual and the romanized form helps.
- Do not include the country (constant within a partition, and must not become a learned shortcut).
- Max length 128 tokens total.

### 6.2 Fine-tuning (on group R only)

1. Build training pairs from R's pre-filtered candidates: all positives, plus up to 4 hard negatives per positive (highest Stage-A scores that are not true matches). Include singletons' candidates as negatives so the model sees "looks similar but is not".
2. Also add chain-name negatives explicitly: same `core_name`, different address.
3. Binary cross-entropy on the single logit, learning rate 1e-5 to 2e-5, warmup 5%, 1 epoch (≈ 1–2M pairs), fp16, batch size as large as memory allows.
4. Evaluate on a small slice of V (pair-level AUC and per-entity top-1 accuracy) against the zero-shot model. Keep the fine-tuned model only if it is better.
5. For the LOCO experiment, fine-tune a second copy on one country only.

### 6.3 Inference

- Score all pre-filtered pairs for G, V (for GBDT training and validation) and test.
- Sort pairs by length before batching to reduce padding; fp16; save in chunks so a crash does not lose progress.
- Measure throughput on 100K pairs first and extrapolate (section 10).

---

## 7. Stage 5 — Final GBDT (Stage-B)

**Input:** pre-filtered pairs + normalized fields + reranker scores.
**Output:** `final_scores.parquet` with (`s1_id`, `cand_id`, `p_match`).

### 7.1 Features

**Name** (computed on both `name_clean` and `name_expanded`; on romanized text if either side is non-Latin):
- Jaro-Winkler, normalized Levenshtein, token-sort ratio, token-set ratio, partial ratio
- word Jaccard, char-3gram Jaccard
- IDF-weighted token overlap (sum of IDF of shared tokens / sum of IDF of union), which down-weights generic words like `group`, `care`
- `core_name` exact match, `name_sorted` exact match, acronym match (either direction)
- legal suffix agreement: same / different / one missing (categorical)
- domain match (`name_domain` equality or domain-vs-name token overlap)
- `expansion_changed_name` on each side
- `script_pair`: same-script or cross-script (categorical). If either side needed romanization, set a flag `romanized_used`; if romanization failed, set string features to NaN (LightGBM handles NaN natively).

**Address:**
- `postal_code` exact, `postal_prefix3` exact
- `house_number` exact, `all_numbers` Jaccard, count of shared numbers
- street token Jaccard (IDF-weighted), locality token overlap
- full address Jaro-Winkler and token-set ratio
- landmark overlap (weak)
- missing flags for both sides (postal, house number, address)

**Model scores:**
- dense cosine (from blocking), `rerank_score`, Stage-A score

**Blocking provenance:**
- rank in each arm, `n_arms_hit`, `cand_source` (S2 or S3)

**Chain / frequency:**
- `name_freq`: number of records in the same country (S1+S2+S3) with the same `core_name`
- `name_freq_log`, and the S1 `core_name` IDF sum

**Competition features — S1 side** (computed over this S1's pre-filtered candidates):
- rank of this candidate by `rerank_score`
- max, second max and mean `rerank_score` for this S1
- gap between this candidate and the best; gap between best and second best
- count of candidates with `rerank_score` above 0.5
- same statistics restricted to candidates from the same source

**Competition features — candidate side** (important because of one-to-one):
- number of different S1 entities that have this candidate in their list
- this S1's `rerank_score` rank among all S1 entities competing for this candidate
- gap between this S1's score and the best competing S1's score for the candidate

**Support features (optional, second pass):**
- for a candidate from S2, the max similarity to this S1's top-scored S3 candidates (and vice versa). True matches from different sources tend to agree with each other.

**Never use:** country as a feature (it is constant within partition and must not be learned), entity ID numbers, or anything derived from labels.

### 7.2 Training

- Train on group G pairs. Label = true match.
- Early stopping on a held-out 10% of G entities (by entity, not by pair).
- Log feature importance; check that address features and competition features rank high (chains depend on them).
- Save the model and the exact feature list and order.

### 7.3 LOCO run

- Train an identical model on US-only G (with the US-only reranker), evaluate on India V; then the reverse.
- Compare macro F0.5 with in-country training. A drop > 15% indicates country-specific overfitting; inspect the top features for country-specific leakage (e.g. postal format assumptions).

---

## 8. Stage 6 — Decision layer (where F0.5 is won)

**Input:** `final_scores.parquet`.
**Output:** `output/matching_results.tsv`.

### 8.1 Decision order (the same order must be used in tuning and inference)

1. **Pair filter:** keep pairs with `p_match ≥ pair_thresh`.
2. **One-to-one assignment (hard constraint):** sort all kept (S1, candidate, p) triples globally within a country by p descending. Assign each candidate to the first S1 that claims it; drop later claims.
3. **Per-S1 cap:** keep at most 12 matches per S1 (highest p).
4. **Singleton gate:** for each S1, if its highest remaining p is below `singleton_thresh`, output an empty list.
5. **Optional margin rule:** if the S1's best p minus the best p that any competing S1 has for the same candidate is below `margin_thresh`, drop that pair.

### 8.2 Why two thresholds

The metric is a per-entity F0.5 averaged over S1 entities, including singletons:
- True singleton predicted empty: 1.0. Any prediction for it: 0.0.
- Non-singleton with correct matches plus one extra: partial credit.
- Non-singleton predicted empty: 0.0.

So the question "should this entity get any match?" is higher-stakes than "should this extra candidate be added?". `singleton_thresh` is expected to be higher than `pair_thresh`.

### 8.3 Tuning procedure (on group V)

1. Build the V evaluation against the **full** ground truth, not only the candidate labels. True matches that blocking missed still count in the recall denominator, and singleton status comes from ground truth. This makes the V score an honest estimate of the leaderboard.
2. Coarse grid: `pair_thresh` 0.30–0.90 step 0.05, `singleton_thresh` from `pair_thresh` to 0.95 step 0.05, `margin_thresh` in {off, 0.05, 0.1}. Run the full decision order (including one-to-one) for each combination.
3. Fine grid: step 0.01 around the best coarse point.
4. The one-to-one step inside the loop is a sort plus a single pass; precompute the sorted order once per `pair_thresh` to keep the grid fast.
5. Report, for the chosen thresholds: macro F0.5 overall, on singletons only, on non-singletons only, and by country; predicted singleton rate vs true (5.58%); mean predicted matches per S1 vs true (3.46).

### 8.4 Thresholds for France

France has no labels, so its thresholds cannot be tuned directly. Procedure:
1. From the LOCO runs, get the best thresholds when evaluating on an unseen country (US→India and India→US), and compare with in-country best thresholds.
2. If the unseen-country optimum shifts only slightly (≤ 0.03), use the in-country thresholds for France.
3. If it shifts more, apply the LOCO-derived (unseen-country) thresholds to France, and the in-country thresholds to US and India. Thresholds may differ per partition because they are selected by a procedure, not hard-coded per country name.
4. **Unsupervised sanity check on test France predictions:** the predicted singleton rate should be near 5–6% and mean matches per S1 near 3.4 (density is similar to train). If France shows, for example, 30% predicted singletons, the model is under-confident on French data: lower France thresholds step by step until the predicted rates are plausible, but never below the US/India values minus 0.1.

### 8.5 Output rules (checked by code before writing)

- Exactly one row for every test S1 ID (1,732,544 rows), including France.
- Empty string for no matches.
- Only S2-/S3- IDs that exist in test files; no duplicates in a list; no duplicate S1 rows.
- Every matched ID is also in `candidate_pairs.tsv` for that S1.
- Tab-separated, header `source1_entity_id	matched_entity_ids`.
- Run `utils/validate_submission.py` and require PASS.

---

## 9. Validation plan

### 9.1 Experiments

| Experiment | Train | Evaluate | Purpose |
|---|---|---|---|
| In-country | G (both countries) | V (both) | main score estimate, threshold tuning |
| LOCO US→India | US-only R/G | India V | France proxy, threshold shift |
| LOCO India→US | India-only R/G | US V | France proxy, threshold shift |
| Ablations | G | V | value of reranker, each blocking arm, competition features, one-to-one, singleton gate |

### 9.2 Metrics to report for each run

- Blocking recall and entity-level ceiling
- Pre-filter recall
- Macro F0.5 overall, singletons only, non-singletons only, by country
- Pair-level precision and recall
- Predicted singleton rate and mean matches per S1

### 9.3 France readiness checklist

- [ ] No country value hard-coded in filters, features or one-hot encodings; partitions built from observed values
- [ ] Normalization handles accents, French legal forms and address words; French abbreviation pairs mined from test and reviewed
- [ ] Generic postal extraction works on French addresses (check 50 samples)
- [ ] Blocking pairs-per-S1 on France comparable to US/India
- [ ] LOCO drop measured and understood
- [ ] France thresholds chosen by the procedure in 8.4
- [ ] France predicted singleton rate and mean matches per S1 within a plausible range

### 9.4 Pseudo-labelling for France (after the first leaderboard submission)

1. Take French pairs with `p_match ≥ 0.97` that are also the unique best for both the S1 and the candidate.
2. Take French S1 entities whose best candidate has `p_match ≤ 0.05` as pseudo-singletons (at most a few thousand).
3. Add them to Stage-B GBDT training with a lower sample weight (0.3). Do not fine-tune the reranker on pseudo-labels.
4. Retrain, re-score France only, and compare the unsupervised France diagnostics. Keep only if the leaderboard confirms.

---

## 10. Compute plan

Throughput numbers are estimates. **Measure each on a 100K sample first** and update this table.

| Step | Volume | Hardware | Rough time |
|---|---|---|---|
| Normalization | ~24M records | CPU, multiprocessing | < 1 h |
| bge-m3 embedding | test 11.7M + train (full 12.5M, or subsample) | 1× A10G, fp16, max_len 64 | 2–5 h |
| FAISS search | per partition | GPU | < 1 h |
| Sparse TF-IDF top-K | per partition | CPU, 64 GB RAM | 1–2 h |
| Key-based arms | all | CPU | < 30 min |
| Stage-A features + model | ~70M test pairs + G/V pairs | CPU | 1–2 h |
| Reranker fine-tune | ~1.5M pairs, 1 epoch | A10G | 1–2 h |
| Reranker inference | ~21M test pairs (N = 12) + G/V pairs | A10G, fp16 | 5–8 h |
| Stage-B features + model | ~25M pairs | CPU | 1–2 h |
| Threshold search | V pairs | CPU | < 30 min |

**Instance:** `g5.4xlarge` (A10G, 16 vCPU, 64 GB RAM) handles every step on one machine, at roughly $1.6/h on-demand (cheaper on spot). About 25–40 hours of use fits in the $200 credit with room for one LOCO reranker run and a re-run. Use spot instances only for resumable steps (all heavy steps should write chunked outputs).

**Reproducibility:** fixed seeds everywhere (splits, FAISS training sample, LightGBM, fine-tuning); pinned versions in `requirements.txt`; model weights referenced by exact Hugging Face revision; every intermediate saved to Parquet so any stage can be re-run alone.

---

## 11. Execution order

| Step | Work | Done when |
|---|---|---|
| 1 | Remaining Stage 0 checks (1.5) | `stage0_report.txt` written |
| 2 | Splits (section 2) | split file saved |
| 3 | Normalization + abbreviation mining (section 3) | coverage table and spot checks pass |
| 4 | Throughput measurement on 100K samples | compute table updated; subsample decision made |
| 5 | Embeddings + blocking arms + union (section 4) | blocking recall ≥ 97% on G ∪ V |
| 6 | Stage-A pre-filter; write `candidate_pairs.tsv` for test | pre-filter recall loss < 0.5 pts |
| 7 | **Early baseline submission:** Stage-A scores only + threshold tuning + one-to-one + singleton gate | first leaderboard score |
| 8 | Reranker fine-tune on R, score G/V/test (section 6) | fine-tuned beats zero-shot on V |
| 9 | Stage-B features and GBDT (section 7) | V macro F0.5 beats baseline |
| 10 | Decision tuning (section 8), France procedure | thresholds chosen; diagnostics plausible |
| 11 | LOCO runs and ablations (section 9) | results recorded |
| 12 | Second submission; optional pseudo-labelling | leaderboard confirms |
| 13 | Package: code, README, requirements, documentation, both TSVs; run validator | validator PASS, end-to-end re-run from README works |

Step 7 matters: a working end-to-end submission early protects against running out of time on the heavy GPU steps.

---

## 12. Code layout

```
code/business_entity_resolution/
├── src/
│   ├── config.py            # paths, seeds, all thresholds and K values in one place
│   ├── io_utils.py          # TSV read/write with sep="\t", Parquet helpers
│   ├── stage0_checks.py     # section 1.5
│   ├── splits.py            # section 2
│   ├── normalize.py         # section 3.1–3.3
│   ├── abbreviations.py     # section 3.4
│   ├── embed.py             # section 4.3
│   ├── blocking.py          # section 4.2–4.4
│   ├── prefilter.py         # section 5, writes candidate_pairs.tsv
│   ├── reranker_train.py    # section 6.2
│   ├── reranker_infer.py    # section 6.3
│   ├── features.py          # section 7.1
│   ├── train_gbdt.py        # section 7.2–7.3
│   ├── decide.py            # section 8.1, 8.5
│   ├── tune_thresholds.py   # section 8.3–8.4
│   ├── metrics.py           # per-entity F0.5, macro average, diagnostics
│   └── run_pipeline.py      # end-to-end: data → blocking → matching → output
├── README.md
└── requirements.txt
```

---

## 13. Documentation template mapping

| Template section | Content from this plan |
|---|---|
| Methodology | Sections 0, 2, 8 |
| Candidate generation / blocking | Sections 4 and 5, with blocking recall, pre-filter recall, reduction ratio, pairs per S1 |
| Model architecture and features | Sections 6 and 7, feature list, reranker fine-tuning details |
| Other relevant information | Stage 0 facts (1), France procedure (8.4, 9.3), LOCO results, ablations, model licenses, compute and reproducibility (10) |

---

## 14. Changes from v2

| Area | v2 | v3 |
|---|---|---|
| Country checks | assumed | confirmed: 0 missing, clean labels, France = 15% of test score |
| Script handling | per record, NaN on cross-script | **per field**, rule-based romanization so string features still work; NaN only if romanization fails |
| Noise handling | generic | specific rules for URLs in names, repeated tokens, junk prefixes, Indian address labels |
| Postal code | country regex | generic extraction first; coverage checked before relying on it (US sample often lacks ZIP) |
| Blocking | 4 arms | 6 arms including address char TF-IDF and house-number+street for chain names; key arms with block size caps |
| Reranker cost | all blocked pairs (~50M+) | **Stage-A pre-filter** to top 12 per S1; this set is `candidate_pairs.tsv` |
| Leakage | not addressed | disjoint entity groups R / G / V for reranker vs GBDT vs thresholds |
| Features | S1-side context | adds candidate-side competition features and IDF-weighted overlaps |
| Decision order | gate → threshold → assignment | threshold → one-to-one → cap → singleton gate, same order in tuning and inference |
| France thresholds | "pick conservative" | procedure using LOCO shift plus unsupervised prediction-rate checks |
| Compute | optimistic | re-estimated for ~24M records with measure-first rule; single g5.4xlarge |
| Evaluation | on candidate labels | against full ground truth, so blocking misses are counted |
