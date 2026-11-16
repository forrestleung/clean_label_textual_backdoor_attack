#!/bin/bash
# Train the clean spamassassin and trec2007 BERT models, then generate poisons for enron,
# spamassassin and trec2007, each using that dataset's latest clean checkpoint.
# Stops at the first failing step.
# Usage: bash scripts/run_spam_clean_and_poisons.sh [device]
#   Hyperparameters come from spam_common.sh and can be overridden with env vars, e.g.
#   MAX_LENGTH=128 ATTACK_NUM=10 bash scripts/run_spam_clean_and_poisons.sh cuda:0
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE=${1:-cuda:0}

for dataset in spamassassin trec2007; do
    echo "== Training clean $dataset model on $DEVICE"
    bash "$DIR/run_bert_spam_clean.sh" "$dataset" "$DEVICE"
done

for dataset in enron spamassassin trec2007; do
    echo "== Generating poisons for $dataset on $DEVICE"
    bash "$DIR/run_bert_spam_samples_gen.sh" "$dataset" "$DEVICE"
done

echo "== Done"
