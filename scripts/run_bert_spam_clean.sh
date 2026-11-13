#!/bin/bash
# Step 1: fine-tune the clean BERT spam filter (victim model).
# Usage: bash scripts/run_bert_spam_clean.sh [enron|spamassassin|trec2007] [device] [extra args]
set -euo pipefail
source "$(dirname "$0")/spam_common.sh"

python attack/train_bert_sst_clean.py \
    --data "$DATASET" \
    --clean_data_path "$CLEAN_DATA" \
    --pre_model_path "$PRE_MODEL" \
    --save_path "$CLEAN_CKPT_DIR/" \
    --mlp_layer_num 0 \
    --device "$DEVICE" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$LOG_DIR/clean.log"
