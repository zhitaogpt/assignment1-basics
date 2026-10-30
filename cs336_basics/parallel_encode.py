"""并行 BPE 编码 glue:用 find_chunk_boundaries 按 <|endoftext|> 对齐切块,
多核并行调用现成的 Tokenizer.encode,按块序拼接成 uint16 .npy。

不改 tokenizer 核心代码,只是把单核 encode 换成 mp.Pool。

用法:
    .venv/bin/python parallel_encode.py data/owt_train.txt data/owt_train_32k.npy \
        --tokenizer-dir out/owt_32k
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import tempfile
from pathlib import Path

import numpy as np

from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.tokenizer import Tokenizer

SPECIAL = "<|endoftext|>"
_TOK: Tokenizer | None = None


def _init(tok_dir: str) -> None:
    global _TOK
    _TOK = Tokenizer.from_files(f"{tok_dir}/vocab.json", f"{tok_dir}/merges.txt", [SPECIAL])


def _encode_chunk(task: tuple[str, int, int, str]) -> tuple[str, int]:
    path, start, end, out_path = task
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    ids = np.asarray(_TOK.encode(text), dtype=np.uint16)
    np.save(out_path, ids)
    return out_path, len(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--procs", type=int, default=min(112, os.cpu_count() or 8))
    args = ap.parse_args()

    n_chunks = args.procs * 4
    with open(args.input, "rb") as f:
        bounds = find_chunk_boundaries(f, n_chunks, SPECIAL.encode())
    tmp = Path(tempfile.mkdtemp(prefix="enc_chunks_", dir="out"))
    tasks = [(args.input, s, e, str(tmp / f"chunk_{i:05d}.npy"))
             for i, (s, e) in enumerate(zip(bounds, bounds[1:]))]
    print(f"[enc] {args.input}: {len(tasks)} chunks x {args.procs} procs", flush=True)

    total = 0
    with mp.Pool(args.procs, initializer=_init, initargs=(args.tokenizer_dir,)) as pool:
        for out, n in pool.imap(_encode_chunk, tasks):  # imap 保序
            total += n
            print(f"[enc] {Path(out).name}: {n:,} (cum {total:,})", flush=True)

    ids = np.concatenate([np.load(t[3]) for t in tasks])  # 按块序拼接
    np.save(args.output, ids)
    print(f"[enc] saved {args.output}: {len(ids):,} tokens, max_id={ids.max()}", flush=True)
    for t in tasks:
        os.remove(t[3])
    tmp.rmdir()


if __name__ == "__main__":
    main()
