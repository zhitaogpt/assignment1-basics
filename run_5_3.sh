#!/usr/bin/env bash
# §5.3 训练脚本的测试驱动(纯 glue:造数据 + 调用你的训练脚本)。
#
# 用法:
#   bash run_5_3.sh data     # 从 out/ts_valid.bin 造 train/valid .npy
#   bash run_5_3.sh smoke    # 极小配置跑 60 步,看 loss / val_loss / perplexity
#   bash run_5_3.sh resume   # 先 0→60 存 ckpt,再从 ckpt 续训到 120,验证连续性
#   bash run_5_3.sh all      # data + smoke + resume(默认)
#
# 前置:先确保 cs336_basics/train.py 能跑(修掉语法错误与两处 None 守卫)。
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
MODE="${1:-all}"

# 极小且共享的一批超参(d=64 / 2 层 / batch4 / ctx64,CPU 几秒)
COMMON=(
  --vocab-size 10000
  --batch-size 4 --context-length 64
  --d-model 64 --num-layers 2 --num-heads 2 --d-ff 128
  --warmup-iters 10 --cosine-cycle-iters 200
  --log-every 5 --eval-every 30 --eval-batches 4
  --device cpu
)

make_data() {
    mkdir -p data out
    if [[ -f data/smoke_train.npy && -f data/smoke_valid.npy ]]; then
        echo "[data] smoke_train.npy / smoke_valid.npy 已存在,跳过"
        return
    fi
    echo "[data] 从 out/ts_valid.bin 生成 train/valid .npy ..."
    "$PY" - <<'PY'
import numpy as np
tok = np.fromfile("out/ts_valid.bin", dtype=np.uint16)
assert tok.max() < 10000, f"max token id {tok.max()} >= vocab_size 10000!"
np.save("data/smoke_train.npy", tok[:2_000_000])
np.save("data/smoke_valid.npy", tok[-200_000:])
print(f"[data] total={len(tok):,}  train=2,000,000  valid=200,000  max_id={tok.max()}")
PY
}

smoke() {
    echo "=============================================================="
    echo "[smoke] 60 步;看:初始 loss ≈ ln(10000)=9.21 / loss 是否下降 / perplexity"
    echo "=============================================================="
    "$PY" -m cs336_basics.train \
        --train-path data/smoke_train.npy --valid-path data/smoke_valid.npy \
        --num-steps 60 "${COMMON[@]}"
}

resume() {
    local ckpt="out/smoke_ckpt.pt"
    rm -f "$ckpt"
    echo "=============================================================="
    echo "[resume] 第一次:0 → 60 步,并存 $ckpt"
    echo "=============================================================="
    "$PY" -m cs336_basics.train \
        --train-path data/smoke_train.npy --valid-path data/smoke_valid.npy \
        --num-steps 60 --checkpoint-path "$ckpt" "${COMMON[@]}"

    echo
    echo "=============================================================="
    echo "[resume] 第二次:应从 start_step 继续到 120;loss 应连续(无跳变)"
    echo "=============================================================="
    "$PY" -m cs336_basics.train \
        --train-path data/smoke_train.npy --valid-path data/smoke_valid.npy \
        --num-steps 120 --checkpoint-path "$ckpt" "${COMMON[@]}"
}

case "$MODE" in
    data)   make_data ;;
    smoke)  make_data; smoke ;;
    resume) make_data; resume ;;
    all)    make_data; smoke; echo; resume ;;
    *) echo "usage: bash run_5_3.sh [data|smoke|resume|all]" >&2; exit 1 ;;
esac