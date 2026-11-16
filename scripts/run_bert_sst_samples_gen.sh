#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 python attack/poison_examples_gen.py
#     --dataset sst \
#     --pre_model_path /path/to/bert/bert-base-uncased/ \
#     --clean_model_path /path/to/clean_models/clean_bert_base_tune_sst_mlp0_adam_lr2e-5_bs32_weight0.002/epoch8.ckpt \
#     --clean_data_path data/clean_data/sst-2/ \
#     --poison_data_path data/clean_data/aux_files/sst-2/poison10_bert_base_mlp0_tune_mlm65_cf0.4_ga_top300base_pop20_iter15.pkl \
#     --attack_num 10 \
#     --pop_size 20 \
#     --top_base_num 300 \
#     --iter_num 15 \
#     --cf_thresh 0.4 \

# Usage: scripts/run_bert_sst_samples_gen.sh [device]   (default: cuda:0; e.g. cuda:1, mps, cpu)
set -euo pipefail
DEVICE=${1:-cuda:0}
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$(dirname "$0")/.."
ATTACK_NUM=10
LOG_DIR=logs/sst
mkdir -p "$LOG_DIR"
CLEAN_CKPT=$(ls -t checkpoints/clean_bert_base_sst_mlp0/epoch*.ckpt 2>/dev/null | head -1 || true)
if [ -z "$CLEAN_CKPT" ]; then
    echo "No clean checkpoint in checkpoints/clean_bert_base_sst_mlp0. Run: bash scripts/run_bert_sst_clean.sh $DEVICE" >&2
    exit 1
fi
POISON_PKL=data/clean_data/aux_files/sst-2/poison${ATTACK_NUM}_bert_base_mlp0_tune_mlm65_cf0.4_ga_top300base_pop20_iter15.pkl
mkdir -p data/clean_data/aux_files/sst-2
echo "Using clean model: $CLEAN_CKPT"
python attack/poison_examples_gen.py \
    --dataset sst \
    --pre_model_path bert-base-uncased \
    --clean_model_path "$CLEAN_CKPT" \
    --clean_data_path data/clean_data/sst-2/ \
    --poison_data_path "$POISON_PKL" \
    --attack_num "$ATTACK_NUM" \
    --pop_size 20 \
    --top_base_num 300 \
    --iter_num 15 \
    --cf_thresh 0.4 \
    --max_length 512 \
    --device "$DEVICE" 2>&1 | tee "$LOG_DIR/samples_gen_attack${ATTACK_NUM}.log"
