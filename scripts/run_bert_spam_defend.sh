#!/bin/bash
# Step 4: run the ONION defense against the spam backdoor (perplexity-based word removal).
# Mirrors Gan et al. Table 5 (SST-2); here applied to a spam dataset.
# Usage: bash scripts/run_bert_spam_defend.sh [enron|spamassassin|trec2007] [device] [extra args]
set -euo pipefail
source "$(dirname "$0")/spam_common.sh"

CLEAN_CKPT=$(latest_clean_ckpt)
BACKDOOR_DIR=$(backdoor_dir)
FILTERED=$(ls -t "$POISON_PKL".filtered.* 2>/dev/null | head -1 || true)
if [ -z "$FILTERED" ]; then
    echo "No filtered poison file for $POISON_PKL. Run the attack step first." >&2
    exit 1
fi
echo "Clean model:    $CLEAN_CKPT"
echo "Backdoor models: $BACKDOOR_DIR"
echo "Poisons:        $FILTERED"
DEFENSE=${DEFENSE:-ONION}   # ONION (needs GPT-2) | BT (needs fairseq) | SCPN (needs OpenAttack SCPN model)
python defend/bert_sst_defense.py \
    --dataset "$DATASET" \
    --lm_model_path gpt2 \
    --pre_model_path "$PRE_MODEL" \
    --clean_data_path "$CLEAN_DATA" \
    --poison_data_path "$FILTERED" \
    --clean_model_path "$CLEAN_CKPT" \
    --backdoor_model_path "$BACKDOOR_DIR" \
    --clean_model_mlp_num 0 \
    --clean_model_mlp_dim 768 \
    --poison_model_mlp_num 1 \
    --poison_model_mlp_dim 1024 \
    --defense_method "$DEFENSE" \
    --device "$DEVICE" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$LOG_DIR/defend_${DEFENSE}_poison${POISON_NUM}.log"
