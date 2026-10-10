"""
BPE 对拍调试脚本 —— 对比你的实现和 naive 参考实现在指定轮次的差异
用法: uv run python /tmp/compare_bpe_round.py
"""
import sys
sys.path.insert(0, "/Users/guanxuan.zzt/repos/AI-learning/cs336/assignment1-basics")

from collections import Counter
import regex as re

# ============ CONFIG ============
INPUT_PATH = "/Users/guanxuan.zzt/repos/AI-learning/cs336/assignment1-basics/tests/fixtures/corpus.en"
SPECIAL_TOKENS = ["<|endoftext|>"]
TARGET_ROUND = 57          # 要对比的轮次
TOP_K = 5                  # 打印 Top K 候选
# ================================

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def build_freq_table(input_path, special_tokens):
    """构建词频表（和你的 train_bpe 一致）"""
    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8")

    special_pat = re.compile("|".join(re.escape(t) for t in special_tokens))
    freq = Counter()
    for part in re.split(special_pat, text):
        for m in re.finditer(PAT, part):
            word_bytes = m.group().encode("utf-8")
            freq[tuple(bytes([b]) for b in word_bytes)] += 1
    return freq


def count_pairs(freq_table):
    """全量统计 pair 频率"""
    cnt = Counter()
    for word, f in freq_table.items():
        for i in range(len(word) - 1):
            cnt[(word[i], word[i+1])] += f
    return cnt


def apply_merge(freq_table, merge_pair):
    """应用一次 merge 到词频表"""
    new_freq = Counter()
    for word, f in freq_table.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == merge_pair:
                new_word.append(word[i] + word[i+1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_freq[tuple(new_word)] += f
    return new_freq


def run_naive_rounds(freq_table, num_rounds):
    """运行 naive 版本，返回每轮记录"""
    records = []
    for rnd in range(1, num_rounds + 1):
        pair_cnt = count_pairs(freq_table)
        if not pair_cnt:
            break

        # 找最高频率
        best_freq = max(pair_cnt.values())
        tied = [p for p, c in pair_cnt.items() if c == best_freq]
        # 平票取字典序大
        best = max(tied)

        records.append({
            "round": rnd,
            "selected": best,
            "freq": best_freq,
            "top_k": sorted(pair_cnt.items(), key=lambda x: (-x[1], x[0]), reverse=False)[:TOP_K],
            "all_tied": tied,
        })

        freq_table = apply_merge(freq_table, best)

    return records, freq_table


def main():
    print(f"Building freq table from {INPUT_PATH}...")
    freq_table = build_freq_table(INPUT_PATH, SPECIAL_TOKENS)
    print(f"Vocab size: {len(freq_table)} unique words")

    print(f"\nRunning naive BPE for {TARGET_ROUND} rounds...")
    records, final_table = run_naive_rounds(freq_table, TARGET_ROUND)

    # 打印目标轮次详情
    target = records[TARGET_ROUND - 1]  # 0-indexed
    print(f"\n{'='*60}")
    print(f"Round {TARGET_ROUND} - Naive Reference")
    print(f"{'='*60}")
    print(f"Selected: {target['selected']}, freq={target['freq']}")
    print(f"\nTop {TOP_K} candidates:")
    for pair, cnt in target['top_k']:
        marker = " <-- SELECTED" if pair == target['selected'] else ""
        print(f"  {pair}: {cnt}{marker}")

    if len(target['all_tied']) > 1:
        print(f"\nTied at freq={target['freq']} ({len(target['all_tied'])} pairs):")
        for p in sorted(target['all_tied']):
            print(f"  {p}")

    # 打印前后各 2 轮上下文
    print(f"\n{'='*60}")
    print("Context (rounds {}-{}):".format(max(1, TARGET_ROUND-2), TARGET_ROUND+2))
    print(f"{'='*60}")
    for rec in records[max(0, TARGET_ROUND-3):TARGET_ROUND+2]:
        rnd = rec['round']
        sel = rec['selected']
        freq = rec['freq']
        marker = " <-- TARGET" if rnd == TARGET_ROUND else ""
        print(f"Round {rnd:3d}: {sel} freq={freq}{marker}")

    print(f"\n{'='*60}")
    print("Now compare with YOUR implementation:")
    print("1. Add print statements in your train_bpe.py at round", TARGET_ROUND)
    print("2. Print token_cnt[pair] for the pairs above")
    print("3. Check if your frequencies match")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
