"""Experiment runner for BPE training (assignment 1, section 2.5).

Glue/scaffolding only. It reuses the student's `train_bpe` and:

  1. measures wall-clock training time,
  2. measures peak RSS (the real memory metric, not dict/heap item counts),
  3. serializes vocab / merges to disk in GPT-2 format,
  4. reports the longest token in the vocabulary.

The GPT-2 byte <-> unicode mapping below is used only to *write* files. The
*inverse* mapping (and the whole `Tokenizer` class: `from_files`, `encode`,
`decode`, `encode_iterable`, special-token handling) is still yours to write in
section 2.6.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from cs336_basics.train_bpe import train_bpe


# --------------------------------------------------------------------------- #
# GPT-2 byte <-> visible-unicode mapping (forward direction only)
# --------------------------------------------------------------------------- #

# Bytes that are already printable in Latin-1 map to themselves; everything
# else gets consecutive code points starting at 256, in increasing byte order.
_PRINTABLE_BYTES = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))


def _build_byte_to_unicode() -> dict[int, str]:
    printable = set(_PRINTABLE_BYTES)
    mapping: dict[int, str] = {}
    n = 0
    for b in range(256):
        if b in printable:
            mapping[b] = chr(b)
        else:
            mapping[b] = chr(256 + n)
            n += 1
    return mapping


BYTE_TO_UNICODE = _build_byte_to_unicode()


def token_to_str(token: bytes) -> str:
    """Render a token's bytes as the GPT-2 visible string (e.g. b' the' -> 'Ġthe')."""
    return "".join(BYTE_TO_UNICODE[b] for b in token)


# --------------------------------------------------------------------------- #
# Serialization (GPT-2 format: vocab.json + merges.txt)
# --------------------------------------------------------------------------- #

def save_vocab_and_merges(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump({token_to_str(tok): tid for tid, tok in vocab.items()}, f, ensure_ascii=False)

    with open(out_dir / "merges.txt", "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for a, b in merges:
            f.write(f"{token_to_str(a)} {token_to_str(b)}\n")


# --------------------------------------------------------------------------- #
# Measurement helpers
# --------------------------------------------------------------------------- #

def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB.

    `ru_maxrss` is bytes on macOS and kilobytes on Linux, hence the branch.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer and serialize it.")
    parser.add_argument("input_path", type=Path, help="training corpus (plain text)")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    args = parser.parse_args()

    start = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )
    elapsed = time.perf_counter() - start

    save_vocab_and_merges(vocab, merges, args.out_dir)

    longest = max(vocab.values(), key=len)
    print(f"input         : {args.input_path}")
    print(f"vocab size    : {len(vocab)}  (merges: {len(merges)})")
    print(f"training time : {elapsed:.1f} s")
    print(f"peak RSS      : {peak_rss_mb():.0f} MB")
    print(f"longest token : {longest!r}  ({len(longest)} bytes)")
    print(f"wrote         : {args.out_dir}/vocab.json, {args.out_dir}/merges.txt")


if __name__ == "__main__":
    main()