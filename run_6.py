"""§6 解码驱动(纯 glue):加载冒烟训练的 checkpoint + TinyStories 10k tokenizer,
给定 prompt 打印不同 temperature / top-p 下的生成结果。

用法:
    .venv/bin/python run_6.py --prompt "Once upon a time"
    .venv/bin/python run_6.py --prompt "The little girl" --max-new-tokens 80

注意:checkpoint 必须是"冒烟配置"训出来的(out/smoke_ckpt.pt),
且 MODEL_CFG 要和训练时的超参完全一致,否则 load_state_dict 会报 shape 不匹配。
"""
from __future__ import annotations

import argparse

import torch

from cs336_basics.decoding import decode
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import AdamW, load_checkpoint

SPECIAL = "<|endoftext|>"
CKPT = "out/smoke_ckpt.pt"
TOK_DIR = "out/tinystories_10k"

# 必须与 run_5_3.sh 的冒烟配置一致
MODEL_CFG = dict(
    vocab_size=10000, context_length=64, d_model=64,
    num_layers=2, num_heads=2, d_ff=128, rope_theta=10000.0,
)


def find_special_id(tok: Tokenizer, token_str: str):
    """在 vocab(id -> bytes)里找回 special token 的 id;找不到返回 None。"""
    target = token_str.encode("utf-8")
    for k, v in tok.vocab.items():
        if v == target:
            return int(k)
    return None


def build_model(device: str) -> TransformerLM:
    c = MODEL_CFG
    return TransformerLM(
        c["vocab_size"], c["context_length"], c["d_model"],
        c["num_layers"], c["num_heads"], c["d_ff"], c["rope_theta"], device,
    )


def generate(model, tok, prompt: str, device: str, max_new_tokens: int,
             temperature: float, top_p: float | None, eos_id):
    ids = tok.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = decode(model, x, max_new_tokens, temperature=temperature,
                 top_p=top_p, eos_token_id=eos_id)
    new_ids = out[0, len(ids):].tolist()
    return tok.decode(new_ids), new_ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint", default=CKPT)
    args = p.parse_args()

    tok = Tokenizer.from_files(f"{TOK_DIR}/vocab.json", f"{TOK_DIR}/merges.txt", [SPECIAL])
    eos_id = find_special_id(tok, SPECIAL)
    print(f"vocab_size={len(tok.vocab)}  eos_id={eos_id}")

    model = build_model(args.device)
    opt = AdamW(model.parameters(), lr=1e-3)          # 只为 load_checkpoint 有个容器
    step = load_checkpoint(args.checkpoint, model, opt)
    model.eval()
    print(f"loaded {args.checkpoint} (trained to step {step})")
    print(f"prompt = {args.prompt!r}\n")

    settings = [
        ("greedy     (T→0)      ", 0.0, None),
        ("T=0.8                  ", 0.8, None),
        ("T=1.5                  ", 1.5, None),
        ("T=0.8, top_p=0.9       ", 0.8, 0.9),
        ("T=1.2, top_p=0.95      ", 1.2, 0.95),
    ]
    for name, temp, top_p in settings:
        text, ids = generate(model, tok, args.prompt, args.device,
                             args.max_new_tokens, temp, top_p, eos_id)
        print(f"[{name}] {text!r}")

    # 贪心确定性自检:T≈0 跑两次应一致
    a, _ = generate(model, tok, args.prompt, args.device, 20, 0.0, None, eos_id)
    b, _ = generate(model, tok, args.prompt, args.device, 20, 0.0, None, eos_id)
    print(f"\n[自检] 贪心两次一致? {a == b}")
    print(f"[自检] 生成 id 全 < vocab_size? "
          f"{all(i < len(tok.vocab) for i in generate(model, tok, args.prompt, args.device, 40, 0.8, None, eos_id)[1])}")


if __name__ == "__main__":
    main()