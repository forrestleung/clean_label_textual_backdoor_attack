#!/bin/bash
# Shared settings for the spam-filtering experiments. Sourced by the run_bert_spam_*.sh scripts.
# Usage of every spam script: bash scripts/<script>.sh [dataset] [device] [extra python args...]
#   dataset: enron (default) | spamassassin | trec2007
#   device:  cuda:0 (default) | mps | cpu
# Hyperparameters can be overridden with environment variables, e.g. POISON_NUM=40 bash scripts/run_bert_spam_attack.sh
#   or MAX_LENGTH=512 bash scripts/run_spam_pipeline.sh enron cuda:0

DATASET=${1:-enron}
DEVICE=${2:-cuda:0}
shift $(( $# < 2 ? $# : 2 ))
EXTRA_ARGS=("$@")

case "$DATASET" in
    enron|spamassassin|trec2007) ;;
    *) echo "Unknown spam dataset '$DATASET' (expected enron, spamassassin or trec2007)" >&2; exit 1 ;;
esac

export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PRE_MODEL=${PRE_MODEL:-bert-base-uncased}
CLEAN_DATA=data/clean_data/$DATASET/

# Inputs are truncated to MAX_LENGTH BERT tokens (incl. [CLS]/[SEP]) in every step. Peak VRAM of clean
# training grows faster than linearly with it, e.g. MAX_LENGTH=128 at the default batch size of 32 peaks around 4.5 GiB.
# Lengths other than 512 get their own checkpoint/poison/log paths so artifacts from different lengths never mix
# (512 keeps the original suffix-free paths).
# MAX_LENGTH=${MAX_LENGTH:-512}  # previous default; 512 OOMs a 10 GiB GPU at batch size 32 on spam data
MAX_LENGTH=${MAX_LENGTH:-128}
if [ "$MAX_LENGTH" = 512 ]; then LEN_SUFFIX=""; else LEN_SUFFIX="_len${MAX_LENGTH}"; fi

CLEAN_CKPT_DIR=checkpoints/clean_bert_base_${DATASET}_mlp0${LEN_SUFFIX}

# Poison generation (Gan et al. SST-2 settings)
ATTACK_NUM=${ATTACK_NUM:-10}      # number of spam test emails to attack (targets); ASR is measured over these
TOP_BASE_NUM=${TOP_BASE_NUM:-300} # closest ham emails perturbed per target
POP_SIZE=${POP_SIZE:-20}
ITER_NUM=${ITER_NUM:-15}
CF_THRESH=${CF_THRESH:-0.4}
POISON_DIR=data/clean_data/aux_files/$DATASET
POISON_PKL=$POISON_DIR/poison${ATTACK_NUM}_bert_base_mlp0_mlm_cf${CF_THRESH}_ga_top${TOP_BASE_NUM}base_pop${POP_SIZE}_iter${ITER_NUM}${LEN_SUFFIX}.pkl

# Backdoor training
POISON_NUM=${POISON_NUM:-40}      # poisons injected per target (poisoning budget); Gan et al. use ~40 for near-100% ASR on SST-2
EPOCH=${EPOCH:-50}
LR=${LR:-5e-3}

LOG_DIR=logs/$DATASET$LEN_SUFFIX
mkdir -p "$LOG_DIR"

latest_clean_ckpt() {
    local ckpt
    ckpt=$(ls -t "$CLEAN_CKPT_DIR"/epoch*.ckpt 2>/dev/null | head -1 || true)
    if [ -z "$ckpt" ]; then
        echo "No clean checkpoint in $CLEAN_CKPT_DIR. Run: bash scripts/run_bert_spam_clean.sh $DATASET $DEVICE" >&2
        exit 1
    fi
    echo "$ckpt"
}

# The attack step saves per-target backdoor models under a directory whose name encodes the
# hyperparameters. Find the one matching the current POISON_NUM rather than hard-coding the name.
BACKDOOR_ROOT=checkpoints/backdoor_bert_base_$DATASET$LEN_SUFFIX
backdoor_dir() {
    local dir
    dir=$(ls -dt "$BACKDOOR_ROOT"/*attack_num${POISON_NUM}_* 2>/dev/null | head -1 || true)
    if [ -z "$dir" ]; then
        echo "No backdoor models under $BACKDOOR_ROOT for poison_num=$POISON_NUM. Run: bash scripts/run_bert_spam_attack.sh $DATASET $DEVICE" >&2
        exit 1
    fi
    echo "$dir"
}
