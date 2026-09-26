#!/bin/bash
# Run the full pipeline (no reranker), checkpointing to GitHub after each stage.
set -e
cd "$(dirname "$0")/.."

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export N_JOBS=${N_JOBS:-24}
export FEATURES_JOBS=${FEATURES_JOBS:-3}

STAGES=(stage0 splits abbrev_seed normalize_first abbrev_mine normalize_final \
        idf embed block prefilter_train prefilter_score \
        features gbdt_train gbdt_score tune loco france decide)

for s in "${STAGES[@]}"; do
    echo ""
    echo "###### stage: $s ######"
    python -m src.run_pipeline --only "$s" 2>&1 | tee -a pipeline.log
    bash scripts/checkpoint.sh "$s"
done

echo ""
echo "[run] pipeline complete."
