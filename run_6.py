"""§6 解码驱动(纯 glue):加载任意 checkpoint + tokenizer,打印不同 temperature / top-p 下的生成。

用法:
    # 冒烟配置(默认)
    .venv/bin/python run_6.py --prompt "Once upon a time"

    # 官方大模型 checkpoint(run_7_full.sh baseline 产出的)
    .venv/bin/python run_6.py --checkpoint out/exp_full/ckpt_baseline.pt \
        --tokenizer-dir out/tinystories_10k --d-model 512 --num-layers 4 \
        --num-heads 16 --d-ff 1344 --context-length 256 --max-new-tokens 256

注意:MODEL_CFG 必须和训练时的超参完全一致,否则 load_state_dict 会报 shape 不匹配。
"""
from __future__ import annotations

import argparse

import torch

from cs336_basics.decoding import decode
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import AdamW, load_checkpoint

SPECIAL = "<|endoftext|>"


def find_special_id(tok: Tokenizer, token_str: str):
    """在 vocab(id -> bytes)里找回 special token 的 id;找不到返回 None。"""
    target = token_str.encode("utf-8")
    for k, v in tok.vocab.items():
        if v == target:
            return int(k)
    return None


def build_model(args) -> TransformerLM:
    return TransformerLM(
        args.vocab_size, args.context_length, args.d_model,
        args.num_layers, args.num_heads, args.d_ff, args.rope_theta, args.device,
    )


def generate(model, tok, prompt: str, device: str, max_new_tokens: int,
             temperature: float, top_p: float | None, eos_id,
             context_length: int | None = None):
    ids = tok.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = decode(model, x, max_new_tokens, temperature=temperature,
                 top_p=top_p, eos_token_id=eos_id, context_length=context_length)
    new_ids = out[0, len(ids):].tolist()
    return tok.decode(new_ids), new_ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint", default="out/smoke_ckpt.pt")
    p.add_argument("--tokenizer-dir", default="out/tinystories_10k")
    # ---- 模型配置(必须与训练时一致)----
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    args = p.parse_args()

    tok = Tokenizer.from_files(f"{args.tokenizer_dir}/vocab.json",
                               f"{args.tokenizer_dir}/merges.txt", [SPECIAL])
    eos_id = find_special_id(tok, SPECIAL)
    print(f"vocab_size={len(tok.vocab)}  eos_id={eos_id}")

    model = build_model(args)
    opt = AdamW(model.parameters(), lr=1e-3)          # 只为 load_checkpoint 有个容器
    step = load_checkpoint(args.checkpoint, model, opt)
    model.eval()
    print(f"loaded {args.checkpoint} (trained to step {step})")
    print(f"prompt = {args.prompt!r}\n")

    settings = [
        ("greedy     (T->0)     ", 0.0, None),
        ("T=0.8                 ", 0.8, None),
        ("T=1.5                 ", 1.5, None),
        ("T=0.8, top_p=0.9      ", 0.8, 0.9),
        ("T=1.2, top_p=0.95     ", 1.2, 0.95),
    ]
    for name, temp, top_p in settings:
        text, ids = generate(model, tok, args.prompt, args.device,
                             args.max_new_tokens, temp, top_p, eos_id,
                             context_length=args.context_length)
        print(f"[{name}] {text!r}")

    # 贪心确定性自检:T≈0 跑两次应一致
    a, _ = generate(model, tok, args.prompt, args.device, 20, 0.0, None, eos_id,
                    context_length=args.context_length)
    b, _ = generate(model, tok, args.prompt, args.device, 20, 0.0, None, eos_id,
                    context_length=args.context_length)
    print(f"\n[自检] 贪心两次一致? {a == b}")
    _, ids40 = generate(model, tok, args.prompt, args.device, 40, 0.8, None, eos_id,
                        context_length=args.context_length)
    print(f"[自检] 生成 id 全 < vocab_size? {all(i < len(tok.vocab) for i in ids40)}")


if __name__ == "__main__":
    main()