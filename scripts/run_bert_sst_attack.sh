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


#!/bin/bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
CLEAN_CKPT=$(ls -t checkpoints/clean_bert_base_sst_mlp0/epoch*.ckpt | head -1)
POISON_PKL=data/clean_data/aux_files/sst-2/poison10_bert_base_mlp0_tune_mlm65_cf0.4_ga_top300base_pop20_iter15.pkl
echo "Using clean model: $CLEAN_CKPT"
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
--poison_num 200 \
--lr 5e-3 \
--epoch 50 \
--pre_model_name bert_base \
--training_strategy 2