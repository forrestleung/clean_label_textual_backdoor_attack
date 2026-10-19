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

#!/bin/bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
CLEAN_CKPT=$(ls -t checkpoints/clean_bert_base_sst_mlp0/epoch*.ckpt | head -1)
POISON_PKL=data/clean_data/aux_files/sst-2/poison10_bert_base_mlp0_tune_mlm65_cf0.4_ga_top300base_pop20_iter15.pkl
mkdir -p data/clean_data/aux_files/sst-2
echo "Using clean model: $CLEAN_CKPT"
python attack/poison_examples_gen.py \
    --dataset sst \
    --pre_model_path bert-base-uncased \
    --clean_model_path "$CLEAN_CKPT" \
    --clean_data_path data/clean_data/sst-2/ \
    --poison_data_path "$POISON_PKL" \
    --attack_num 10 \
    --pop_size 20 \
    --top_base_num 300 \
    --iter_num 15 \
    --cf_thresh 0.4