"""Section 2.7 tokenizer experiments — driver script.

(a) and (b) are pure plumbing (sample docs, encode, divide) and are filled in.
(c) throughput and (d) uint16 serialization are left as TODO stubs: they carry the
deliverables' reasoning, so fill them in yourself.

Run from the repo root:
    uv run python -m cs336_basics.run_2_7
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer

DATA = Path("data")
OUT = Path("out")
SPECIAL = "<|endoftext|>"  # document separator in BOTH corpora


def load_tokenizer(dir_path: Path, expected_vocab_size: int) -> Tokenizer:
    tok = Tokenizer.from_files(
        str(dir_path / "vocab.json"), str(dir_path / "merges.txt"), [SPECIAL]
    )
    assert len(tok.vocab) == expected_vocab_size, (len(tok.vocab), expected_vocab_size)
    return tok


def sample_documents(text_path: Path, n: int = 10, prefix_chars: int = 2_000_000) -> list[str]:
    """Read a BOUNDED prefix (not the whole file), split on SPECIAL, keep n docs.

    The last split piece may be truncated mid-document; it is dropped.
    """
    with open(text_path, encoding="utf-8") as f:
        chunk = f.read(prefix_chars)
    docs = [d for d in chunk.split(SPECIAL) if d.strip()]
    if len(docs) > 1:
        docs = docs[:-1]  # drop the possibly-truncated tail
    return docs[:n]


def compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> float:
    """bytes/token: UTF-8 byte length of the raw text divided by number of token IDs."""
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    total_tokens = sum(len(tokenizer.encode(d)) for d in docs)
    return total_bytes / total_tokens


def measure_throughput(
    tokenizer: Tokenizer, text_path: Path, prefix_chars: int = 5_000_000
) -> tuple[float, int, int, float]:
    """Feed a bounded slice of the corpus through encode_iterable, time it.

    Returns (bytes_per_sec, n_bytes, n_tokens, elapsed_sec).
    """
    with open(text_path, encoding="utf-8") as f:
        text = f.read(prefix_chars)
    n_bytes = len(text.encode("utf-8"))
    lines = text.splitlines(keepends=True)  # mimics a file handle's line iteration
    start = time.perf_counter()
    n_tokens = sum(1 for _ in tokenizer.encode_iterable(lines))
    elapsed = time.perf_counter() - start
    return n_bytes / elapsed, n_bytes, n_tokens, elapsed


def encode_corpus_to_uint16(tokenizer: Tokenizer, text_path: Path, bin_path: Path,
                            flush_every: int = 1_000_000) -> int:
    """Stream-encode a corpus and append raw uint16 ids to bin_path (memory-bounded).

    We do NOT accumulate all ids in a Python list (a 500M-token list would cost GBs).
    Instead we flush fixed-size chunks straight to disk, then read back with
    np.fromfile(bin_path, dtype=np.uint16).
    """
    n = 0
    buf: list[int] = []
    with open(bin_path, "wb") as out, open(text_path, encoding="utf-8") as f:
        for _id in tokenizer.encode_iterable(f):
            buf.append(_id)
            if len(buf) >= flush_every:
                np.array(buf, dtype=np.uint16).tofile(out)
                n += len(buf)
                buf.clear()
        if buf:
            np.array(buf, dtype=np.uint16).tofile(out)
            n += len(buf)
    return n


def main() -> None:
    ts = load_tokenizer(OUT / "tinystories_10k", 10_000)

    if "smoke" in sys.argv:
        n = encode_corpus_to_uint16(ts, DATA / "TinyStoriesV2-GPT4-valid.txt", OUT / "ts_valid.bin")
        a = np.fromfile(OUT / "ts_valid.bin", dtype=np.uint16)
        print(f"(d) smoke: {n:,} tokens | reloaded {a.shape} {a.dtype} | first ids {a[:10].tolist()}")
        return

    owt = load_tokenizer(OUT / "owt_valid_32k", 32_000)

    ts_docs = sample_documents(DATA / "TinyStoriesV2-GPT4-valid.txt")
    owt_docs = sample_documents(DATA / "owt_valid.txt")
    print(f"sampled {len(ts_docs)} TinyStories docs, {len(owt_docs)} OWT docs")

    # ---- (a) each tokenizer on its own corpus ----
    ts_ratio = compression_ratio(ts, ts_docs)
    owt_ratio = compression_ratio(owt, owt_docs)
    print(f"(a) TinyStories 10K on TinyStories : {ts_ratio:.3f} bytes/token")
    print(f"(a) OWT 32K         on OWT         : {owt_ratio:.3f} bytes/token")

    # ---- (b) OWT sample through the TinyStories tokenizer ----
    cross_ratio = compression_ratio(ts, owt_docs)
    print(f"(b) TinyStories 10K on OWT         : {cross_ratio:.3f} bytes/token")

    # ---- (c) throughput (bytes/second) ----
    bps, nb, nt, dt = measure_throughput(ts, DATA / "TinyStoriesV2-GPT4-valid.txt")
    print(f"(c) throughput: {bps:,.0f} bytes/s  ({nb} bytes, {nt} tokens, {dt:.3f}s)")
    # TODO(c): how long for the 825 GB Pile? compute seconds = 825e9 / bps, then days.
    #          Also: is `user ~= real`? (single-threaded) check with /usr/bin/time if curious.

    # ---- (d) serialize datasets as uint16 (pass "d" on the command line to run) ----
    if "d" in sys.argv:
        for name, tok, text in [
            ("tinystories_valid", ts, DATA / "TinyStoriesV2-GPT4-valid.txt"),
            ("tinystories_train", ts, DATA / "TinyStoriesV2-GPT4-train.txt"),
            # ("owt_valid", owt, DATA / "owt_valid.txt"),   # owt_train still .gz
        ]:
            n = encode_corpus_to_uint16(tok, text, OUT / f"{name}.bin")
            print(f"(d) {name}: {n:,} tokens -> {OUT / (name + '.bin')}")
    # TODO(d): why is uint16 appropriate? (compare the uint16 max value to your vocab size)


if __name__ == "__main__":
    main()