#!/bin/bash
# Run the whole spam experiment: clean model -> poison generation -> attack.
# Steps whose output already exists are skipped, so it is safe to re-run after a failure.
# Usage: bash scripts/run_spam_pipeline.sh [enron|spamassassin|trec2007] [device]
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/spam_common.sh"

if ls "$CLEAN_CKPT_DIR"/epoch*.ckpt >/dev/null 2>&1; then
    echo "== Clean model exists in $CLEAN_CKPT_DIR, skipping training"
else
    echo "== [1/3] Training clean $DATASET model"
    bash "$DIR/run_bert_spam_clean.sh" "$DATASET" "$DEVICE"
fi

if [ -f "$POISON_PKL" ]; then
    echo "== Poisons exist at $POISON_PKL, skipping generation"
else
    echo "== [2/3] Generating poisons"
    bash "$DIR/run_bert_spam_samples_gen.sh" "$DATASET" "$DEVICE"
fi

echo "== [3/3] Attacking (poison_num=$POISON_NUM)"
bash "$DIR/run_bert_spam_attack.sh" "$DATASET" "$DEVICE"
grep -E "Attack Success Rate|Average Clean Test Accuracy|Test acc on clean model|Targets with no poisons" \
    "$LOG_DIR/attack_poison${POISON_NUM}.log" | tail -5
