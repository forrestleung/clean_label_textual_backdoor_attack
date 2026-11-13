#!/bin/bash
# Step 3: filter the poisons for stealthiness, train one backdoored model per target, report ASR and CACC.
# Usage: bash scripts/run_bert_spam_attack.sh [enron|spamassassin|trec2007] [device] [extra args]
#   Budget sweep: for n in 10 20 40 100 200; do POISON_NUM=$n bash scripts/run_bert_spam_attack.sh enron cuda:0; done
#   Filter override: bash scripts/run_bert_spam_attack.sh enron cuda:0 --ppl_thresh 400 --gerr_thresh 6
set -euo pipefail
source "$(dirname "$0")/spam_common.sh"

CLEAN_CKPT=$(latest_clean_ckpt)
if [ ! -f "$POISON_PKL" ]; then
    echo "Missing $POISON_PKL. Run: bash scripts/run_bert_spam_samples_gen.sh $DATASET $DEVICE" >&2
    exit 1
fi
echo "Using clean model: $CLEAN_CKPT"
echo "Using poisons: $POISON_PKL"
python attack/bert_sst_attack.py \
    --dataset "$DATASET" \
    --pre_model_path "$PRE_MODEL" \
    --save_path "$BACKDOOR_ROOT" \
    --clean_model_path "$CLEAN_CKPT" \
    --clean_data_path "$CLEAN_DATA" \
    --poison_data_path "$POISON_PKL" \
    --clean_model_mlp_layer 0 \
    --clean_model_mlp_dim 768 \
    --poison_model_mlp_layer 1 \
    --poison_model_mlp_dim 1024 \
    --poison_num "$POISON_NUM" \
    --lr "$LR" \
    --epoch "$EPOCH" \
    --pre_model_name bert_base \
    --training_strategy 2 \
    --max_length "$MAX_LENGTH" \
    --device "$DEVICE" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$LOG_DIR/attack_poison${POISON_NUM}.log"
