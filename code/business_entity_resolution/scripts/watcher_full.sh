#!/bin/bash
# Extended watcher: on each new completed stage,
#   1. commit + push small artifacts to GitHub (via checkpoint.sh)
#   2. upload heavy stage outputs to HuggingFace Hub
#
# Run in a dedicated tmux session and detach.
# Requires HF_TOKEN in env (or ~/.cache/huggingface/token).

set -u

REPO_ROOT="/DATA/air_force_object_detection/ml_challnege_26"
VENV_PY="/DATA/air_force_object_detection/.venv/bin/python"
HF_REPO="jainsaabb/ml_challenge_2026_artifacts"
LOG_FILE="$REPO_ROOT/artifacts/reports/watcher_full.log"

cd "$REPO_ROOT" || exit 1

# Ensure HF_TOKEN is available (fallback to cached token)
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

# Map: which stage → which local dir to upload → which HF path prefix
upload_for_stage() {
    local stage="$1"
    case "$stage" in
        block)
            _upload_folder "artifacts/blocking"        "blocking"
            ;;
        prefilter_score)
            _upload_folder "artifacts/prefilter"       "prefilter"
            _upload_file   "output/candidate_pairs.tsv" "output/candidate_pairs.tsv"
            ;;
        features)
            _upload_folder "artifacts/final_scores"    "final_scores_features"
            ;;
        gbdt_score)
            _upload_folder "artifacts/final_scores"    "final_scores_scored"
            ;;
        tune)
            _upload_folder "artifacts/reports"         "reports"
            ;;
        decide)
            _upload_file   "output/matching_results.tsv" "output/matching_results.tsv"
            ;;
    esac
}

_upload_folder() {
    local local_dir="$1"
    local remote_prefix="$2"
    if [ ! -d "$REPO_ROOT/$local_dir" ] || [ -z "$(ls -A "$REPO_ROOT/$local_dir" 2>/dev/null)" ]; then
        echo "[watcher_full] skip $local_dir (missing or empty)"
        return
    fi
    echo "[watcher_full] uploading $local_dir → HF:$remote_prefix"
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$VENV_PY" - <<PY
from huggingface_hub import upload_folder
upload_folder(
    folder_path="$REPO_ROOT/$local_dir",
    repo_id="$HF_REPO",
    repo_type="dataset",
    path_in_repo="$remote_prefix",
    commit_message="auto: sync $remote_prefix after stage completion",
)
PY
}

_upload_file() {
    local local_path="$1"
    local remote_path="$2"
    if [ ! -f "$REPO_ROOT/$local_path" ]; then
        echo "[watcher_full] skip $local_path (missing)"
        return
    fi
    echo "[watcher_full] uploading file $local_path → HF:$remote_path"
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$VENV_PY" - <<PY
from huggingface_hub import upload_file
upload_file(
    path_or_fileobj="$REPO_ROOT/$local_path",
    path_in_repo="$remote_path",
    repo_id="$HF_REPO",
    repo_type="dataset",
    commit_message="auto: sync $remote_path after stage completion",
)
PY
}

latest_stage() {
    "$VENV_PY" -c "
import json
try:
    d = json.load(open('artifacts/reports/stage_timings.json'))
    print(list(d.keys())[-1] if d else '')
except Exception:
    print('')
" 2>/dev/null
}

echo "[watcher_full] starting at $(date)" | tee -a "$LOG_FILE"

LAST=""
while true; do
    CURRENT=$(latest_stage)
    if [ -n "$CURRENT" ] && [ "$CURRENT" != "$LAST" ]; then
        echo "[watcher_full] $(date) new stage completed: $CURRENT" | tee -a "$LOG_FILE"

        # 1. Small artifacts → GitHub
        bash "$REPO_ROOT/code/business_entity_resolution/scripts/checkpoint.sh" "$CURRENT" 2>&1 | tee -a "$LOG_FILE"

        # 2. Heavy artifacts → HuggingFace
        upload_for_stage "$CURRENT" 2>&1 | tee -a "$LOG_FILE"

        LAST="$CURRENT"
    fi
    sleep 60
done
