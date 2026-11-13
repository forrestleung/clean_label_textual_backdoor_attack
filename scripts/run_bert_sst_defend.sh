#!/bin/bash
# ONION defense against the SST-2 backdoor (reproduces Gan et al. Table 5, ONION column).
# Usage: bash scripts/run_bert_sst_defend.sh [device]   (default: mps; e.g. cuda:0, cpu)
# Env overrides: POISON_NUM (default 40), DEFENSE (ONION|BT|SCPN).
set -euo pipefail
DEVICE=${1:-mps}
POISON_NUM=${POISON_NUM:-40}
DEFENSE=${DEFENSE:-ONION}
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$(dirname "$0")/.."

PRE_MODEL=${PRE_MODEL:-bert-base-uncased}
CLEAN_CKPT=$(ls -t checkpoints/clean_bert_base_sst_mlp0/epoch*.ckpt 2>/dev/null | head -1 || true)
BACKDOOR_DIR=$(ls -dt checkpoints/backdoor_bert_base_sst/*attack_num${POISON_NUM}_* 2>/dev/null | head -1 || true)
POISON_PKL=data/clean_data/aux_files/sst-2/poison10_bert_base_mlp0_tune_mlm65_cf0.4_ga_top300base_pop20_iter15.pkl
FILTERED=$(ls -t "$POISON_PKL".filtered.* 2>/dev/null | head -1 || echo "$POISON_PKL")
if [ -z "$CLEAN_CKPT" ] || [ -z "$BACKDOOR_DIR" ]; then
    echo "Need a clean checkpoint and backdoor models. Run run_bert_sst_clean.sh and run_bert_sst_attack.sh first." >&2
    exit 1
fi
echo "Clean model:     $CLEAN_CKPT"
echo "Backdoor models: $BACKDOOR_DIR"
echo "Poisons:         $FILTERED"
mkdir -p logs/sst
python defend/bert_sst_defense.py \
    --dataset sst \
    --lm_model_path gpt2 \
    --pre_model_path "$PRE_MODEL" \
    --clean_data_path data/clean_data/sst-2 \
    --poison_data_path "$FILTERED" \
    --clean_model_path "$CLEAN_CKPT" \
    --backdoor_model_path "$BACKDOOR_DIR" \
    --clean_model_mlp_num 0 \
    --clean_model_mlp_dim 768 \
    --poison_model_mlp_num 1 \
    --poison_model_mlp_dim 1024 \
    --defense_method "$DEFENSE" \
    --device "$DEVICE" 2>&1 | tee logs/sst/defend_${DEFENSE}_poison${POISON_NUM}.log
