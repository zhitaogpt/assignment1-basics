#!/usr/bin/env bash
# Driver for assignment 1, section 2.5 experiments (glue only).
#
# Usage:
#   bash run_2_5.sh            # small-sample correctness check + full TinyStories run
#   bash run_2_5.sh owt        # also train on OpenWebText (needs data/owt_train.txt)
#
# Run it from anywhere; it cd's to the assignment root.

set -euo pipefail
cd "$(dirname "$0")"

SAMPLE="tests/fixtures/tinystories_sample_5M.txt"
TRAIN="data/TinyStoriesV2-GPT4-train.txt"
VOCAB_SIZE=10000
SPECIAL="<|endoftext|>"

echo "=============================================================="
echo "[1/3] small sample -> out/sample5m_10k_opt   (correctness)"
echo "=============================================================="
uv run python -m cs336_basics.run_bpe_experiment \
    "$SAMPLE" --vocab-size "$VOCAB_SIZE" --out-dir out/sample5m_10k_opt

echo
echo "=============================================================="
echo "[2/3] diff optimized vs baseline   (must be byte-identical)"
echo "=============================================================="
# Baseline produced by your ORIGINAL train_bpe (out/sample5m_10k). If you don't
# have it yet, comment this block out, or regenerate it from the old code.
diff out/sample5m_10k/vocab.json  out/sample5m_10k_opt/vocab.json && echo "VOCAB_SAME"
diff out/sample5m_10k/merges.txt  out/sample5m_10k_opt/merges.txt && echo "MERGES_SAME"

echo
echo "=============================================================="
echo "[3/3] full TinyStories -> out/tinystories_10k_opt"
echo "=============================================================="
/usr/bin/time -l uv run python -m cs336_basics.run_bpe_experiment \
    "$TRAIN" --vocab-size "$VOCAB_SIZE" --out-dir out/tinystories_10k_opt

if [[ "${1:-}" == "owt" ]]; then
    echo
    echo "=============================================================="
    echo "[owt] OpenWebText (32k vocab) -> out/owt_32k"
    echo "=============================================================="
    if [[ ! -f data/owt_train.txt ]]; then
        echo "!! data/owt_train.txt not found. Prepare it first:"
        echo "   cd data && curl -L -o owt_train.txt.gz \\"
        echo "     https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"
        echo "   gunzip owt_train.txt.gz && cd .."
        exit 1
    fi
    /usr/bin/time -l uv run python -m cs336_basics.run_bpe_experiment \
        data/owt_train.txt --vocab-size 32000 --out-dir out/owt_32k
fi