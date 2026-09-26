#!/bin/bash
# Commit small "checkpoint-worthy" artifacts and push to GitHub.
# Called after each pipeline stage.
# Assumes git repo root is ml_challnege_26/ (three levels above this script).
set -e
STAGE_NAME="${1:-unknown}"

# Navigate to git repo root
cd "$(dirname "$0")/../../.."

# Ensure artifact subdirs exist (empty dirs are ignored by git, so this is safe)
mkdir -p artifacts/models artifacts/reports artifacts/splits artifacts/idf

# Stage small artifacts + the pipeline log
git add \
    artifacts/abbreviations.json \
    artifacts/splits/ \
    artifacts/idf/ \
    artifacts/models/*.txt \
    artifacts/reports/ \
    code/business_entity_resolution/pipeline.log \
    2>/dev/null || true

# Only commit if something is actually staged
if git diff --cached --quiet; then
    echo "[checkpoint] no changes to commit for stage $STAGE_NAME"
    exit 0
fi

STAMP=$(date +"%Y-%m-%d %H:%M:%S")
git commit -m "checkpoint after $STAGE_NAME ($STAMP)" || true

# push in background so it doesn't block the pipeline
(git push origin main 2>/dev/null &) &
echo "[checkpoint] committed stage $STAGE_NAME; push running in background"
