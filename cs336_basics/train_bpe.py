from __future__ import annotations

import os
import heapq
import multiprocessing as mp

from collections.abc import Iterable
from collections import Counter
import regex as re

from .pretokenization_example import find_chunk_boundaries

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def _pretokenize_chunk(args: tuple[str, int, int, tuple[str, ...]]) -> Counter:
    """Pre-tokenize one [start, end) byte range of a corpus; return its Counter.

    SCAFFOLDING around the existing counting logic. Kept at module top level
    (not a closure) so it is picklable under the 'spawn' start method that
    multiprocessing uses on macOS: each worker re-imports this module.
    """
    path, start, end, special_tokens = args
    special_pat = re.compile("|".join(re.escape(tok) for tok in special_tokens))
    counter = Counter()
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    for txt in re.split(special_pat, chunk):
        counter.update(PAT.findall(txt))  # <-- your original counting logic
    return counter


class MaxBytesTupleItem:
    def __init__(self, freq: int, bytes_tuple: tuple):
        self.freq = freq
        self.bytes_tuple = bytes_tuple

    def __lt__(self, other):
        # 1. 先比外层第一个元素（假设也是降序排列，如果是升序则改为 self.freq < other.freq）
        if self.freq != other.freq:
            return self.freq > other.freq

        # 2. 如果第一个元素相同，对比第二个元素（元组反向对比，实现字典序降序）
        return self.bytes_tuple > other.bytes_tuple

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    special_pat = re.compile('|'.join(re.escape(token) for token in special_tokens))
    
    merge_cnt = vocab_size - len(special_tokens) - 256
    vocab = {i:bytes([i]) for i in range(256)}
    merges = []
    # print(text)
    # split_text = []

    # ---- pre-tokenization (parallel across chunks) ----
    # special_tokens[0] (e.g. <|endoftext|>) delimits documents; we align chunk
    # boundaries to it so no pre-token / merge ever crosses a boundary.
    num_workers = 8
    split_token = special_tokens[0].encode("utf-8")
    with open(input_path, "rb") as f:
        # more chunks than workers -> better load balancing
        boundaries = find_chunk_boundaries(f, num_workers * 8, split_token)

    ranges = [
        (str(input_path), start, end, tuple(special_tokens))
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with mp.Pool(num_workers) as pool:
        partials = pool.map(_pretokenize_chunk, ranges)

    word_counter = Counter()
    for partial in partials:
        word_counter.update(partial)

    freq_counter = Counter()

    for word_str, freq in word_counter.items():
        word_bytes = word_str.encode('utf-8')
        freq_counter[tuple(bytes([_]) for _ in word_bytes)] += freq
    
    max_token_heap = []

    freq_items = list(freq_counter.items())
    token_bag = dict()
    token_cnt = Counter()
    
    # 统计token
    for i, (word, freq) in enumerate(freq_items):
        for j in range(len(word) - 1):
            token1 = word[j]
            token2 = word[j+1]
            merge_token = (token1, token2)
            token_cnt[merge_token] += freq
            if merge_token not in token_bag:
                token_bag[merge_token] = {i}
            else:
                token_bag[merge_token].add(i)
    
    for merge_token, freq in token_cnt.items():
        try:
            # print(f"Pushing to heap: merge_token={merge_token}, freq={freq}")
            heapq.heappush(max_token_heap, MaxBytesTupleItem(freq, merge_token))
        except Exception as e:
            print(f"Error pushing to heap: {e}")
            print(f"merge_token: {merge_token}, freq: {freq}")

    # merge
    # DEBUG_CNT=56
    while merge_cnt > 0:
        # DEBUG_TAG= False
        # DEBUG_ROUND = vocab_size - len(special_tokens) - 256 - DEBUG_CNT
        # if merge_cnt >= DEBUG_ROUND - 1 and merge_cnt <= DEBUG_ROUND + 1:
        #     DEBUG_TAG= True
        #     top_10 = token_cnt.most_common(10)
        #     for item in top_10:
        #         print(f"Top 10 token: {item}")
        while  (max_token_heap[0].freq != token_cnt.get( max_token_heap[0].bytes_tuple,-1)):
            pop_item = heapq.heappop(max_token_heap)
            # if DEBUG_TAG:
                # print("pop from heap, because it's not in token_cnt or freq mismatch",pop_item.bytes_tuple, pop_item.freq)
        item = heapq.heappop(max_token_heap)
        vocab[len(vocab)] = b''.join(item.bytes_tuple)
        merge_cnt -= 1
        merges.append(item.bytes_tuple)
        # # if DEBUG_TAG:
        # print("Current merge count:", merge_cnt)
        # print("Current merge token:", item.bytes_tuple, item.freq)
        # 更新频次
        merge_token = item.bytes_tuple
        local_update_list = list(token_bag[merge_token])
        for i in local_update_list:
            word, freq = freq_items[i]
            new_word = []
            j = 0
            while j < len(word):
                if j < len(word) - 1 and (word[j], word[j+1]) == merge_token:
                    new_word.append(b''.join(merge_token))
                    j += 2
                else:
                    new_word.append(word[j])
                    j += 1
            new_word_tuple = tuple(new_word)
            freq_items[i] = (new_word_tuple, freq)

            for k in range(len(word) - 1):
                token1 = word[k]
                token2 = word[k+1]
                old_merge_token = (token1, token2)
                token_cnt[old_merge_token] -= freq
                if token_cnt[old_merge_token] == 0:
                    del token_cnt[old_merge_token]
                    del token_bag[old_merge_token]
                else:
                    assert token_cnt[old_merge_token] > 0
                    if i in token_bag[old_merge_token]:
                        token_bag[old_merge_token].remove(i)
                heapq.heappush(max_token_heap, MaxBytesTupleItem(token_cnt[old_merge_token], old_merge_token))
            # 更新 token_cnt
            for k in range(len(new_word_tuple) - 1):
                token1 = new_word_tuple[k]
                token2 = new_word_tuple[k+1]
                new_merge_token = (token1, token2)
                token_cnt[new_merge_token] += freq
                if new_merge_token not in token_bag:
                    token_bag[new_merge_token] = set([i])
                else:
                    token_bag[new_merge_token].add(i)
                # if new_merge_token not in old_token:
                heapq.heappush(max_token_heap, MaxBytesTupleItem(token_cnt[new_merge_token], new_merge_token))
            # for k in range(len(new_word_tuple) - 1):
            #     token1 = new_word_tuple[k]
            #     token2 = new_word_tuple[k+1]
            #     new_merge_token = (token1, token2)

            #     if new_merge_token not in old_token:
                    # if DEBUG_TAG:
                    #     print(f"Pushing new merge token to heap: {new_merge_token}, freq: {token_cnt[new_merge_token]}")
                    

    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')
    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe(
        input_path="tests/fixtures/corpus.en",
        vocab_size=340,
        special_tokens=["<|endoftext|>"]
    )
    print("Vocabulary:")
    for idx, token in vocab.items():
        print(f"{idx}: {token}")
    print("\nMerges:")
    for merge in merges:
        print(merge)
