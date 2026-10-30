#!/usr/bin/env bash
# ============================================================================
# §7 完整实验编排(纯 glue:造数据 + 调用你的 train.py + 汇总)。
#
# 对齐 assignment §7.2.1 的【官方固定架构参数】(不要改):
#   vocab_size=10000   context_length=256   d_model=512   d_ff=1344
#   num_layers=4       num_heads=16         rope_theta=10000
#   总 token = batch × steps × ctx ≈ 327,680,000
# 需要你自己调的:  learning_rate / warmup / betas / weight_decay / batch。
#
# 用法:
#   bash run_7_full.sh data       # 编码 TinyStories(train+valid, 10k vocab)
#   bash run_7_full.sh data-owt   # 编码 OWT(32k vocab) —— 需先有 out/owt_32k/ 与 data/owt_train.txt
#   bash run_7_full.sh baseline   # 官方配置 baseline(存 ckpt,供 generate)
#   bash run_7_full.sh lr         # lr 扫描(含至少一个发散 run)         [learning_rate (a)(b)]
#   bash run_7_full.sh batch      # batch 扫描(1/8/32/64/128)           [batch_size_experiment]
#   bash run_7_full.sh ablation   # 4 个消融 + no_rmsnorm@低lr           [§7.3 四问]
#   bash run_7_full.sh generate   # 用 baseline ckpt 采样                [generate]
#   bash run_7_full.sh owt        # main_experiment(OWT,同架构同预算)    [main_experiment]
#   bash run_7_full.sh report     # 汇总 out/exp_full/*.log
#
# 环境变量:
#   PROFILE=full   # full=328M token(交付用);lite=41M token(先跑通/调参,默认)
#   DEVICE=cuda:0  # 两台 A100:一个终端 DEVICE=cuda:0,另一个 DEVICE=cuda:1
#   VOCAB=10000    # TinyStories=10000;OWT=32000(owt 子命令会自动切)
#   LR=1e-3        # 调好最优 lr 后 export;baseline/ablation 都用它
#   CHARS=0        # data 子命令:>0 时只编码前 N 字符(调试用),0=全量
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-.venv/bin/python}"
OUT="${OUT:-out/exp_full}"
DEVICE="${DEVICE:-cpu}"
PROFILE="${PROFILE:-lite}"
VOCAB="${VOCAB:-10000}"

# ---- 官方固定架构参数(assignment §7.2.1,勿改)----
CTX=256
DMODEL=512
NLAYERS=4
NHEADS=16
DFF=1344
THETA=10000

# ---- token 预算 -> steps ----
# full: 32 × 40000 × 256 = 327,680,000  (官方目标)
# lite: 32 ×  5000 × 256 =  40,960,000  (低资源 tip 的 40M)
if [[ "$PROFILE" == "full" ]]; then
    STEPS=40000
else
    STEPS=5000
fi
BATCH=32
TOTAL_TOKENS=$((BATCH * STEPS * CTX))
BATCH_SWEEP_STEPS="${BATCH_SWEEP_STEPS:-5000}"   # batch 扫描:固定步数口径(每档同 steps)

# ---- 数据路径 ----
TS_TRAIN=data/ts_train_10k_full.npy
TS_VALID=data/ts_valid_10k_full.npy
OWT_TRAIN=data/owt_train_32k.npy
OWT_VALID=data/owt_valid_32k.npy

LR="${LR:-1e-3}"
LR_MIN="${LR_MIN:-1e-4}"
export LR LR_MIN

mkdir -p "$OUT" data

echo "[cfg] PROFILE=$PROFILE  DEVICE=$DEVICE"
echo "[cfg] 官方架构: d_model=$DMODEL layers=$NLAYERS heads=$NHEADS d_ff=$DFF ctx=$CTX"
echo "[cfg] batch=$BATCH steps=$STEPS  ->  $TOTAL_TOKENS tokens (官方目标 327680000)"

# ------------------------------------------------------------------ 造数据
make_data() {
    if [[ -f "$TS_TRAIN" && -f "$TS_VALID" ]]; then
        echo "[data] $TS_TRAIN / $TS_VALID 已存在,跳过"
        return
    fi
    [[ -f out/tinystories_10k/vocab.json ]] || {
        echo "!! 缺 out/tinystories_10k/ tokenizer。先跑 §2 的 BPE 训练。" >&2; exit 1; }
    echo "[data] 用 out/tinystories_10k 编码 TinyStories train/valid ..."
    "$PY" - <<'PY'
import os, numpy as np
from cs336_basics.tokenizer import Tokenizer
chars = int(os.environ.get("CHARS", "0"))
tok = Tokenizer.from_files("out/tinystories_10k/vocab.json",
                           "out/tinystories_10k/merges.txt", ["<|endoftext|>"])
for split in ("train", "valid"):
    src = f"data/TinyStoriesV2-GPT4-{split}.txt"
    text = open(src, encoding="utf-8").read()
    if chars > 0:
        text = text[:chars]
    ids = np.asarray(tok.encode(text), dtype=np.uint16)
    dst = f"data/ts_{split}_10k_full.npy"
    np.save(dst, ids)
    print(f"[data] {src} -> {dst}  {len(ids):,} tokens  max_id={ids.max()}")
PY
}

make_data_owt() {
    local tok=tok32k
    if [[ ! -f out/owt_32k/vocab.json ]]; then
        echo "!! 缺 out/owt_32k/ tokenizer(需在 data/owt_train.txt 上训 32k BPE)。" >&2
        echo "   先跑:  gunzip -k data/owt_train.txt.gz && VOCAB_SIZE=32000 bash run_owt.sh train" >&2
        exit 1
    fi
    if [[ ! -f data/owt_train.txt ]]; then
        echo "!! 缺 data/owt_train.txt(先 gunzip data/owt_train.txt.gz)。" >&2; exit 1
    fi
    if [[ -f "$OWT_TRAIN" && -f "$OWT_VALID" ]]; then
        echo "[data] $OWT_TRAIN / $OWT_VALID 已存在,跳过"; return
    fi
    echo "[data] 用 out/owt_32k 编码 OWT train/valid ..."
    "$PY" - <<'PY'
import os, numpy as np
from cs336_basics.tokenizer import Tokenizer
chars = int(os.environ.get("CHARS", "0"))
tok = Tokenizer.from_files("out/owt_32k/vocab.json",
                           "out/owt_32k/merges.txt", ["<|endoftext|>"])
for split in ("train", "valid"):
    src = f"data/owt_{split}.txt"
    text = open(src, encoding="utf-8").read()
    if chars > 0:
        text = text[:chars]
    ids = np.asarray(tok.encode(text), dtype=np.uint16)
    dst = f"data/owt_{split}_32k.npy"
    np.save(dst, ids)
    print(f"[data] {src} -> {dst}  {len(ids):,} tokens  max_id={ids.max()}")
PY
}

# ---------------------------------------------------------------- 单次运行
run_one() {
    local name="$1"; shift
    local train="${TRAIN_OVERRIDE:-$TS_TRAIN}"
    local valid="${VALID_OVERRIDE:-$TS_VALID}"
    local vocab="${VOCAB_OVERRIDE:-$VOCAB}"
    # 把 cosine 周期对齐到 num-steps 的最后一步
    local steps="$STEPS"; local args=("$@")
    for ((i = 0; i < ${#args[@]}; i++)); do
        [[ "${args[i]}" == "--num-steps" ]] && steps="${args[i + 1]}"
    done
    echo "=============================================================="
    echo "[run] $name  steps=$steps  train=$(basename "$train")  -> $OUT/$name.log"
    echo "=============================================================="
    "$PY" -m cs336_basics.train \
        --train-path "$train" --valid-path "$valid" \
        --vocab-size "$vocab" --context-length "$CTX" \
        --d-model "$DMODEL" --num-layers "$NLAYERS" --num-heads "$NHEADS" --d-ff "$DFF" \
        --rope-theta "$THETA" --warmup-iters 100 \
        --num-steps "$steps" --cosine-cycle-iters "$steps" \
        --log-every 100 --eval-every 500 --eval-batches 20 \
        --device "$DEVICE" \
        "$@" 2>&1 | tee "$OUT/$name.log" \
        || echo "[warn] run '$name' 退出码非 0(可能发散/crash),继续后续 run"
}

baseline() {
    run_one baseline \
        --variant baseline --batch-size "$BATCH" \
        --lr-max "$LR" --lr-min "$LR_MIN" --max-grad-norm 1.0 \
        --checkpoint-path "$OUT/ckpt_baseline.pt"
}

# learning_rate (a) sweep + (b) 至少一个发散 run
lr_sweep() {
    for lr in 3e-4 1e-3 3e-3 1e-2 3e-2; do
        run_one "lr_${lr}" \
            --variant baseline --batch-size "$BATCH" \
            --lr-max "$lr" --lr-min "$(python3 -c "print(f'{$lr*0.1:g}')")" --max-grad-norm 1.0
    done
}

# batch_size_experiment:1 到显存上限;固定步数口径(每档同 steps)
batch_sweep() {
    for b in 1 8 32 64 128; do
        run_one "batch_${b}" \
            --variant baseline --batch-size "$b" \
            --num-steps "$BATCH_SWEEP_STEPS" \
            --lr-max "$LR" --lr-min "$LR_MIN" --max-grad-norm 1.0
    done
}

# §7.3 四个消融 + no_rmsnorm 的"更低 lr 能否稳住"
ablation() {
    run_one ablation_pre_norm  --variant baseline    --batch-size "$BATCH" \
        --lr-max "$LR" --lr-min "$LR_MIN" --max-grad-norm 1.0
    run_one ablation_no_rmsnorm --variant no_rmsnorm --batch-size "$BATCH" \
        --lr-max "$LR" --lr-min "$LR_MIN" --max-grad-norm 1.0
    # 关键补充:handout 问"能否用更低 lr 稳住" —— 最优 lr 的 1/10
    run_one ablation_no_rmsnorm_lowlr --variant no_rmsnorm --batch-size "$BATCH" \
        --lr-max "$(python3 -c "print(f'{$LR/10:g}')")" --lr-min "$(python3 -c "print(f'{$LR/100:g}')")" --max-grad-norm 1.0
    run_one ablation_post_norm --variant post_norm  --batch-size "$BATCH" \
        --lr-max "$LR" --lr-min "$LR_MIN" --max-grad-norm 1.0
    run_one ablation_nope      --variant nope       --batch-size "$BATCH" \
        --lr-max "$LR" --lr-min "$LR_MIN" --max-grad-norm 1.0
    run_one ablation_silu      --variant silu       --batch-size "$BATCH" \
        --lr-max "$LR" --lr-min "$LR_MIN" --max-grad-norm 1.0
}

# generate:用 baseline 的 checkpoint 采样
generate() {
    local ckpt="$OUT/ckpt_baseline.pt"
    [[ -f "$ckpt" ]] || { echo "!! 缺 $ckpt(先跑 baseline)。" >&2; exit 1; }
    "$PY" run_6.py \
        --checkpoint "$ckpt" --tokenizer-dir out/tinystories_10k \
        --vocab-size 10000 --context-length "$CTX" \
        --d-model "$DMODEL" --num-layers "$NLAYERS" --num-heads "$NHEADS" --d-ff "$DFF" \
        --prompt "Once upon a time" --max-new-tokens 256 --device "$DEVICE" \
        | tee "$OUT/generate.txt"
}

# main_experiment:同架构、同迭代数,在 OWT 上跑
owt_experiment() {
    TRAIN_OVERRIDE="$OWT_TRAIN" VALID_OVERRIDE="$OWT_VALID" VOCAB_OVERRIDE=32000 \
        run_one main_owt \
            --variant baseline --batch-size "$BATCH" \
            --lr-max "$LR" --lr-min "$LR_MIN" --max-grad-norm 1.0 \
            --checkpoint-path "$OUT/ckpt_owt.pt"
}

report() {
    "$PY" cs336_basics/plot_curves.py "$OUT"/*.log --plot "$OUT/curves.png" --report
}

case "${1:-all}" in
    data)     make_data ;;
    data-owt) make_data_owt ;;
    baseline) make_data; baseline ;;
    lr)       make_data; lr_sweep ;;
    batch)    make_data; batch_sweep ;;
    ablation) make_data; ablation ;;
    generate) generate ;;
    owt)      make_data_owt; owt_experiment ;;
    report)   report ;;
    all)      make_data; baseline; lr_sweep; batch_sweep; ablation; report ;;
    *) echo "usage: bash run_7_full.sh [data|data-owt|baseline|lr|batch|ablation|generate|owt|report|all]" >&2; exit 1 ;;
esac