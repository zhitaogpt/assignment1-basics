"""
BPE 对拍调试脚本 —— 只含打印骨架，算法复用你 train_bpe.py 里的逻辑。
用法: 改 CONFIG 里的语料路径, 然后:
    cd ~/repos/AI-learning/cs336/assignment1-basics
    uv run python /tmp/debug_bpe.py
"""
import sys
sys.path.insert(0, "/Users/guanxuan.zzt/repos/AI-learning/cs336/assignment1-basics")

from collections import Counter
import regex as re

# ============ CONFIG ============
INPUT_PATH = "/Users/guanxuan.zzt/repos/AI-learning/cs336/assignment1-basics/tests/fixtures/corpus.en.test"
SPECIAL_TOKENS = ["<|endoftext|>"]
NUM_ROUNDS = 5          # 手动模拟前 N 轮 merge
TOP_K = 8               # 每轮打印频率前 K 的 pair
# ================================

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def show(title):
    print(f"\n{'='*20} {title} {'='*20}")


# ---------- 块 1: 词频表 ----------
show("原始文本")
with open(INPUT_PATH, "rb") as f:
    text = f.read().decode("utf-8")
print(repr(text))

special_pat = re.compile("|".join(re.escape(t) for t in SPECIAL_TOKENS))
freq = Counter()
for part in re.split(special_pat, text):
    for m in re.finditer(PAT, part):
        word_bytes = m.group().encode("utf-8")
        freq[tuple(bytes([b]) for b in word_bytes)] += 1

show("词频表 (词 -> 次数)")
for word, cnt in sorted(freq.items(), key=lambda kv: -kv[1]):
    print(f"{b''.join(word)!r:40s} {cnt}")

# ---------- 块 2: 初始 pair 计数 ----------
def count_pairs(freq_table):
    cnt = Counter()
    for word, f in freq_table.items():
        for i in range(len(word) - 1):
            cnt[(word[i], word[i+1])] += f
    return cnt

pair_cnt = count_pairs(freq)
show(f"初始 pair Top {TOP_K}")
for pair, c in pair_cnt.most_common(TOP_K):
    print(f"{pair}  freq={c}")

# ---------- 块 3: 手动模拟前 N 轮 merge (naive 全量重数) ----------
show(f"手动模拟前 {NUM_ROUNDS} 轮 merge")
for rnd in range(1, NUM_ROUNDS + 1):
    if not pair_cnt:
        print("(无 pair 可 merge)")
        break
    best_freq = max(pair_cnt.values())
    tied = [p for p, c in pair_cnt.items() if c == best_freq]
    # spec: 平票取字典序大的
    best = max(tied)
    print(f"\n--- 第 {rnd} 轮 ---")
    print(f"最高频率 = {best_freq}, 平票候选 = {len(tied)} 个:")
    for p in sorted(tied):
        print(f"    {p}")
    print(f">>> 选中: {best}")

    # 应用 merge 到词频表
    new_freq = Counter()
    for word, f in freq.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == best:
                new_word.append(word[i] + word[i+1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_freq[tuple(new_word)] += f
    freq = new_freq
    pair_cnt = count_pairs(freq)

show(f"模拟 {NUM_ROUNDS} 轮后的 merges 序列 (答案锚点: [(b' ', b't'), (b' ', b'a'), ...])")
