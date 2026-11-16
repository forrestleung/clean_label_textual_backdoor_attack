#!/bin/bash
# CUDA_VISIBLE_DEVICES=7 python attack/bert_sst_attack.py
# --dataset sst \
# --pre_model_path /data/home/ganleilei/bert/bert-base-uncased \
# --save_path /data/home/ganleilei/attack/models \
# --clean_model_path /path/to/clean_models/clean_bert_tune_sst_adam_lr2e-5_bs32_weight0.002/epoch10.ckpt \
# --clean_data_path data/clean_data/sst-2/ \
# --poison_data_path data/clean_data/aux_files/sst-2/poison100_bert_base_tune_mlm35_cf0.4_ga_top300base_pop20_iter15.pkl \
# --clean_model_mlp_layer 0 \
# --clean_model_mlp_dim 768 \
# --poison_model_mlp_layer 1 \
# --poison_model_mlp_dim 1024 \
# --poison_num 200 \
# --lr 5e-3 \
# --epoch 50 \
# --pre_model_name bert_base \
# --training_strategy 2 \

# Usage: scripts/run_bert_sst_attack.sh [device]   (default: cuda:0; e.g. cuda:1, mps, cpu)
# Env override: POISON_NUM (poisons per target, default 40 to match the spam experiments).
set -euo pipefail
DEVICE=${1:-cuda:0}
POISON_NUM=${POISON_NUM:-40}
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$(dirname "$0")/.."
LOG_DIR=logs/sst
mkdir -p "$LOG_DIR"
CLEAN_CKPT=$(ls -t checkpoints/clean_bert_base_sst_mlp0/epoch*.ckpt 2>/dev/null | head -1 || true)
if [ -z "$CLEAN_CKPT" ]; then
    echo "No clean checkpoint in checkpoints/clean_bert_base_sst_mlp0. Run: bash scripts/run_bert_sst_clean.sh $DEVICE" >&2
    exit 1
fi
POISON_PKL=data/clean_data/aux_files/sst-2/poison10_bert_base_mlp0_tune_mlm65_cf0.4_ga_top300base_pop20_iter15.pkl
if [ ! -f "$POISON_PKL" ]; then
    echo "Missing $POISON_PKL. Run: bash scripts/run_bert_sst_samples_gen.sh $DEVICE" >&2
    exit 1
fi
echo "Using clean model: $CLEAN_CKPT"
echo "Using poisons: $POISON_PKL"
python attack/bert_sst_attack.py \
--dataset sst \
--pre_model_path bert-base-uncased \
--save_path checkpoints/backdoor_bert_base_sst \
--clean_model_path "$CLEAN_CKPT" \
--clean_data_path data/clean_data/sst-2/ \
--poison_data_path "$POISON_PKL" \
--clean_model_mlp_layer 0 \
--clean_model_mlp_dim 768 \
--poison_model_mlp_layer 1 \
--poison_model_mlp_dim 1024 \
--poison_num "$POISON_NUM" \
--lr 5e-3 \
--epoch 50 \
--pre_model_name bert_base \
--training_strategy 2 \
--max_length 512 \
--device "$DEVICE" 2>&1 | tee "$LOG_DIR/attack_poison${POISON_NUM}.log"
