#!/bin/bash
# Step 2: generate clean-label poisons (perturbed ham emails) for ATTACK_NUM spam test targets.
# Usage: bash scripts/run_bert_spam_samples_gen.sh [enron|spamassassin|trec2007] [device] [extra args]
set -euo pipefail
source "$(dirname "$0")/spam_common.sh"

CLEAN_CKPT=$(latest_clean_ckpt)
mkdir -p "$POISON_DIR"
echo "Using clean model: $CLEAN_CKPT"
echo "Writing poisons to: $POISON_PKL"
python attack/poison_examples_gen.py \
    --dataset "$DATASET" \
    --pre_model_path "$PRE_MODEL" \
    --clean_model_path "$CLEAN_CKPT" \
    --clean_data_path "$CLEAN_DATA" \
    --poison_data_path "$POISON_PKL" \
    --attack_num "$ATTACK_NUM" \
    --pop_size "$POP_SIZE" \
    --top_base_num "$TOP_BASE_NUM" \
    --iter_num "$ITER_NUM" \
    --cf_thresh "$CF_THRESH" \
    --max_length "$MAX_LENGTH" \
    --device "$DEVICE" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$LOG_DIR/samples_gen_attack${ATTACK_NUM}.log"
