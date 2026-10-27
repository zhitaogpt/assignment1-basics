#!/usr/bin/env bash
# §7 实验编排(纯 glue:造数据 + 用不同超参调用你的 train.py + 汇总)。
# 不做任何模型/算法实现;消融变体需要你在 model.py / train.py 里加开关(见文末 NOTE)。
#
# 用法:
#   bash run_7.sh data       # 编码一段 TinyStories train 子集 -> data/ts_train_10k.npy
#   bash run_7.sh baseline   # 跑一次基线,存 out/exp/baseline.log
#   bash run_7.sh lr         # 学习率扫描(3 个 lr)
#   bash run_7.sh batch      # batch size 扫描
#   bash run_7.sh ablation   # 4 个消融(需要 --variant 支持)
#   bash run_7.sh report     # 解析 out/exp/*.log 打印对比表
#   bash run_7.sh all        # data + baseline + lr + batch + report
#
# 环境变量:
#   DEVICE=cuda:0    训练设备(容器里用 cuda:0;本地默认 cpu)
#   STEPS=2000       训练步数
#   SIZE=small|mid   模型/数据规模
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
OUT=out/exp
DEVICE="${DEVICE:-cpu}"
STEPS="${STEPS:-2000}"
SIZE="${SIZE:-small}"

TRAIN=data/ts_train_10k.npy
VALID=data/smoke_valid.npy          # 复用 §5.3 生成的验证集(10k 词表)

if [[ "$SIZE" == "mid" ]]; then
    MODEL=(--d-model 512 --num-layers 8  --num-heads 8  --d-ff 1408)
    BATCH=32
else
    MODEL=(--d-model 256 --num-layers 6  --num-heads 8  --d-ff 704)
    BATCH=16
fi

COMMON=(
    --vocab-size 10000 --context-length 256
    "${MODEL[@]}"
    --warmup-iters 100 --cosine-cycle-iters "$STEPS"
    --num-steps "$STEPS" --log-every 50 --eval-every 200 --eval-batches 20
    --device "$DEVICE"
)

mkdir -p "$OUT"

# ---------------------------------------------------------------- 数据
make_data() {
    if [[ -f "$TRAIN" ]]; then echo "[data] $TRAIN 已存在,跳过"; return; fi
    local chars="${CHARS:-30000000}"   # 默认只编码前 30M 字符(约 10M token)
    echo "[data] 编码 TinyStories train 前 $chars 字符 -> $TRAIN"
    "$PY" - "$chars" <<'PY'
import sys, numpy as np
from cs336_basics.tokenizer import Tokenizer
chars = int(sys.argv[1])
tok = Tokenizer.from_files("out/tinystories_10k/vocab.json",
                           "out/tinystories_10k/merges.txt", ["<|endoftext|>"])
text = open("data/TinyStoriesV2-GPT4-train.txt", encoding="utf-8").read(chars)
ids = np.asarray(tok.encode(text), dtype=np.uint16)
np.save("data/ts_train_10k.npy", ids)
print(f"[data] encoded {len(ids):,} tokens, max_id={ids.max()}")
PY
}

# ---------------------------------------------------------------- 单次运行
run_one() {
    local name="$1"; shift
    echo "=============================================================="
    echo "[run] $name   (log -> $OUT/$name.log)"
    echo "=============================================================="
    "$PY" -m cs336_basics.train \
        --train-path "$TRAIN" --valid-path "$VALID" \
        "${COMMON[@]}" "$@" 2>&1 | tee "$OUT/$name.log"
    echo "[run] done: $name"
}

baseline() { run_one baseline --batch-size "$BATCH" --lr-max 1e-3 --lr-min 1e-4 --max-grad-norm 1.0; }

lr_sweep() {
    for lr in 3e-4 1e-3 3e-3; do
        run_one "lr_${lr}" --batch-size "$BATCH" --lr-max "$lr" --lr-min "$(python3 -c "print($lr*0.1)")" --max-grad-norm 1.0
    done
}

batch_sweep() {
    for b in 8 16 32 64; do
        run_one "batch_${b}" --batch-size "$b" --lr-max 1e-3 --lr-min 1e-4 --max-grad-norm 1.0
    done
}

ablation() {
    # NOTE: 需要你的 train.py 支持 --variant {baseline,no_rmsnorm,post_norm,nope,silu}
    for v in baseline no_rmsnorm post_norm nope silu; do
        run_one "ablation_${v}" --variant "$v" --batch-size "$BATCH" --lr-max 1e-3 --lr-min 1e-4 --max-grad-norm 1.0
    done
}

report() {
    "$PY" cs336_basics/plot_curves.py "$OUT"/*.log --plot "$OUT"/curves.png --report
}

case "${1:-all}" in
    data)     make_data ;;
    baseline) make_data; baseline ;;
    lr)       make_data; lr_sweep ;;
    batch)    make_data; batch_sweep ;;
    ablation) make_data; ablation ;;
    report)   report ;;
    all)      make_data; baseline; lr_sweep; batch_sweep; report ;;
    *) echo "usage: bash run_7.sh [data|baseline|lr|batch|ablation|report|all]" >&2; exit 1 ;;
esac

# NOTE —— 你需要在代码里补的东西(只有这样消融才能跑):
#   1) model.py:
#        - RMSNorm 开关          : 去掉 ln1/ln2/ln_final(no_rmsnorm)
#        - norm 位置开关          : pre-norm / post-norm(post_norm)
#        - 位置编码开关           : RoPE / 无位置编码(nope)
#        - FFN 类型开关           : SwiGLU / SiLU(silu,即无门控的 SiLU FFN)
#   2) train.py: 增加 --variant 参数,把上面的开关传给 TransformerLM。