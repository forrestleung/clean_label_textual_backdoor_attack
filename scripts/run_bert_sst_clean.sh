#!/bin/bash
#  python attack/train_bert_sst_clean.py \
#  --gpu_id 6 \
#  --data sst \
#  --clean_data_path data/clean_data/sst-2/ \
#  --pre_model_path /data/home/ganleilei/bert/bert-base-uncased \
#  --save_path /data/home/ganleilei/attack/models/clean_bert_base_tune_sst_mlp0_adam_lr2e-5_bs32_weight0.002/ \
#  --mlp_layer_num 0 \

# Usage: scripts/run_bert_sst_clean.sh [device]   (default: cuda:0; e.g. cuda:1, mps, cpu)
set -euo pipefail
DEVICE=${1:-cuda:0}
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$(dirname "$0")/.."
LOG_DIR=logs/sst
mkdir -p "$LOG_DIR"
python attack/train_bert_sst_clean.py \
 --data sst \
 --clean_data_path data/clean_data/sst-2/ \
 --pre_model_path bert-base-uncased \
 --save_path checkpoints/clean_bert_base_sst_mlp0/ \
 --mlp_layer_num 0 \
 --max_length 512 \
 --device "$DEVICE" 2>&1 | tee "$LOG_DIR/clean.log"
