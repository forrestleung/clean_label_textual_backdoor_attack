#!/bin/bash
# Usage: scripts/run_bert_ag_clean.sh [device]   (default: mps; e.g. cuda:0, cpu)
DEVICE=${1:-mps}
 python attack/train_bert_sst_clean.py \
 --device "$DEVICE" \
 --data ag \
 --clean_data_path data/clean_data/ag \
 --pre_model_path /home/ganleilei/data/bert/bert-large-uncased \
 --save_path /home/ganleilei/data/attack/models/clean_bert_tune_large_ag_mlp1_adam_lr2e-5_bs32_weight0.002/ \
 --mlp_layer_num 1
