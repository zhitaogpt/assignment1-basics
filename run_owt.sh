#!/usr/bin/env bash
# OpenWebText experiment driver for assignment 1, section 2.5 (train_bpe_expts_owt).
# Glue only: it downloads the OWT sample, (optionally) takes a byte subset, and
# trains a BPE tokenizer, printing wall-clock time and max RSS.
#
# Usage:
#   bash run_owt.sh download                 # just fetch + gunzip the data
#   bash run_owt.sh valid                    # trial on the small valid split
#   bash run_owt.sh train                    # full train split
#   bash run_owt.sh both                     # valid trial, then full train (default)
#
# Env knobs:
#   VOCAB_SIZE=16000        override the vocabulary size (default 32000)
#   SUBSET_BYTES=2000000000 only train on the first N bytes of owt_train
#                           (useful when full OWT does not fit in RAM)
#
# Example (downscaled run on a big-RAM box is not needed; this fits 24 GB):
#   SUBSET_BYTES=2000000000 VOCAB_SIZE=10000 bash run_owt.sh train

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-both}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"
SUBSET_BYTES="${SUBSET_BYTES:-0}"
BASE_URL="https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main"

download_split() {
    local split="$1" path="data/owt_${split}.txt"
    if [[ -f "$path" ]]; then
        echo "[data] $path already present ($(du -h "$path" | cut -f1))"
        return
    fi
    mkdir -p data
    echo "[data] downloading owt_${split}.txt.gz ..."
    curl -L --fail --retry 5 -C - -o "${path}.gz" "${BASE_URL}/owt_${split}.txt.gz"
    gunzip -f "${path}.gz"
    echo "[data] $path ready ($(du -h "$path" | cut -f1))"
}

run_one() {
    local corpus="$1" out="$2"
    if [[ ! -f "$corpus" ]]; then
        echo "!! missing $corpus" >&2
        exit 1
    fi
    echo "--------------------------------------------------------------"
    echo "corpus=$corpus  vocab_size=$VOCAB_SIZE  -> $out"
    echo "--------------------------------------------------------------"
    # NOTE: /usr/bin/time reports the peak RSS of the process it launches; with
    # multiprocessing the worker processes' memory is not all counted, so treat
    # it as a lower bound on the true footprint.
    /usr/bin/time -l uv run python -m cs336_basics.run_bpe_experiment \
        "$corpus" --vocab-size "$VOCAB_SIZE" --out-dir "$out"
}

run_train() {
    download_split train
    if [[ "$SUBSET_BYTES" -gt 0 ]]; then
        local subset="data/owt_train_subset.txt"
        echo "[data] taking first $SUBSET_BYTES bytes -> $subset"
        head -c "$SUBSET_BYTES" data/owt_train.txt > "$subset"
        run_one "$subset" "out/owt_train_subset_${VOCAB_SIZE}"
    else
        run_one data/owt_train.txt out/owt_32k
    fi
}

case "$MODE" in
    download) download_split valid; download_split train ;;
    valid)    download_split valid; run_one data/owt_valid.txt out/owt_valid_32k ;;
    train)    run_train ;;
    both)
        download_split valid
        run_one data/owt_valid.txt out/owt_valid_32k
        run_train
        ;;
    *)
        echo "usage: bash run_owt.sh [download|valid|train|both]" >&2
        exit 1
        ;;
esac