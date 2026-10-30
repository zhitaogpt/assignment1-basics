"""§7 实验日志汇总(纯 glue):解析 train.py 的 stdout 日志,输出对比表 / 曲线图 / 报告数字。

用法:
    python cs336_basics/plot_curves.py out/exp/*.log                       # 对比表
    python cs336_basics/plot_curves.py out/exp/*.log --plot out/exp/curves.png
    python cs336_basics/plot_curves.py out/exp/*.log --report              # 报告数字清单

识别的日志行(train.py 现在打印的格式):
    step 300, loss: 5.123, lr: 0.0009
    val_loss: 5.010, perplexity : 150.3
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

TRAIN_RE = re.compile(r"step\s+(\d+),\s*loss:\s*([0-9.eE+-]+)")
VAL_RE = re.compile(r"val_loss:\s*([0-9.eE+-]+),\s*perplexity\s*:\s*([0-9.eE+-]+)")


def parse_log(path: Path):
    train_steps, train_losses = [], []
    val_steps, val_losses, val_ppls = [], [], []
    last_step = -1
    for line in path.read_text(errors="ignore").splitlines():
        m = TRAIN_RE.search(line)
        if m:
            last_step = int(m.group(1))
            train_steps.append(last_step)
            train_losses.append(float(m.group(2)))
            continue
        m = VAL_RE.search(line)
        if m:
            val_steps.append(last_step)
            val_losses.append(float(m.group(1)))
            val_ppls.append(float(m.group(2)))
    return {
        "name": path.stem,
        "train_steps": train_steps,
        "train_losses": train_losses,
        "val_steps": val_steps,
        "val_losses": val_losses,
        "val_ppls": val_ppls,
    }


def _loss_stats(losses):
    """返回 (min, max, last, 是否发散, 首个发散 step 索引 or None)。"""
    if not losses:
        return None
    best = min(losses)
    last = losses[-1]
    # 发散判据:最后一个点 > 最小值 * 1.5,且曾经出现过明显上升
    diverged = last > best * 1.5
    diverge_at = None
    if diverged:
        for i, v in enumerate(losses):
            if v > best * 1.5:
                diverge_at = i
                break
    return best, max(losses), last, diverged, diverge_at


def summary_row(r) -> str:
    n = len(r["train_steps"])
    final_train = f"{r['train_losses'][-1]:.4f}" if n else "—"
    if r["val_losses"]:
        best = min(r["val_losses"])
        best_i = r["val_losses"].index(best)
        return (f"{r['name']:<22} {n:>6} {final_train:>12} "
                f"{best:>10.4f} @{r['val_steps'][best_i]:<6} "
                f"{r['val_losses'][-1]:>10.4f} {r['val_ppls'][-1]:>10.2f}")
    return f"{r['name']:<22} {n:>6} {final_train:>12} {'—':>10} {'—':>7} {'—':>10} {'—':>10}"


def report_block(r) -> str:
    """单个 run 的"报告数字"块——你写结论时直接抄这里。"""
    lines = [f"  [{r['name']}]"]
    ts = _loss_stats(r["train_losses"])
    if ts:
        best, mx, last, div, div_at = ts
        lines.append(f"    train loss: best={best:.4f}  last={last:.4f}  max={mx:.4f}")
        if div:
            step = r["train_steps"][div_at] if div_at is not None else "?"
            lines.append(f"    ⚠ train loss 发散:step≈{step} 起,best={best:.4f} → last={last:.4f}")
    if r["val_losses"]:
        best = min(r["val_losses"])
        best_i = r["val_losses"].index(best)
        lines.append(
            f"    val   loss: best={best:.4f} @step{r['val_steps'][best_i]}  "
            f"last={r['val_losses'][-1]:.4f}  ppl_last={r['val_ppls'][-1]:.2f}"
        )
        # 过拟合信号:last 比 best 差超过 5%
        if r["val_losses"][-1] > best * 1.05:
            lines.append(f"    ⚠ 过拟合迹象:val 从 {best:.4f} 回升到 {r['val_losses'][-1]:.4f}")
    return "\n".join(lines)


def make_plot(results, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] 未安装 matplotlib,跳过绘图(pip install matplotlib 后可生成曲线图)")
        return
    # 按 run 长度(最大 step)分组:full(40000 步)和 lite(5000 步)混在一张图上,
    # 前者会把后者压进左侧一条缝。拆成多张图:out_path 放最长的 full 组,
    # 其余每组一张 <stem>_<steps>steps.png;只有一种长度时就一张图。
    by_steps: dict[int, list] = {}
    for r in results:
        by_steps.setdefault(max(r["train_steps"], default=0), []).append(r)
    longest = max(by_steps)
    jobs = [(out_path, by_steps[longest])]
    for k in sorted(by_steps):
        if k != longest:
            jobs.append((out_path.with_name(f"{out_path.stem}_{k}steps{out_path.suffix}"),
                         by_steps[k]))
    for path, runs in jobs:
        _plot_group(plt, runs, path)


def _plot_group(plt, results, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # 发散 run(如 no_rmsnorm)的 loss 会到天文数字,线性坐标会把其他曲线压成直线。
    # 对策:x/y 都用 log 坐标,且 y 上限只按未发散 run 计算——
    # 发散曲线冲出上边界自然截断,既看清正常曲线形状,又能看到它"飞天"的时刻。
    diverged = {r["name"] for r in results
                if (s := _loss_stats(r["train_losses"])) and s[3]}
    # 分组编码:同组同线型+marker,组内同色系不同深浅:
    #   baseline=黑实线+圆点, lr=红系虚线+三角, batch=蓝系点划线+方块, ablation=紫系点线+菱形
    _GROUPS = {  # 组前缀 -> (linestyle, marker, 色系 colormap;None=纯黑)
        "baseline": ("-", "o", None),
        "lr":       ("--", "^", "Reds"),
        "batch":    ("-.", "s", "Blues"),
        "ablation": (":", "D", "Purples"),
        "main":     ("-", "X", "Greens"),
    }
    by_group: dict[str, list] = {}
    for r in results:
        g = r["name"].split("_")[0]
        by_group.setdefault(g if g in _GROUPS else "baseline", []).append(r)
    for gname, runs in by_group.items():
        ls, marker, cmap_name = _GROUPS[gname]
        n = len(runs)
        for j, r in enumerate(runs):
            tag = " (diverged)" if r["name"] in diverged else ""
            color = ("black" if cmap_name is None
                     else plt.get_cmap(cmap_name)(0.45 + 0.5 * (j / max(n - 1, 1))))
            if r["train_steps"]:
                # log-x 下 step 0 无法显示,滤掉非正点
                xs, ys = zip(*[(s, l) for s, l in zip(r["train_steps"], r["train_losses"]) if s > 0])
                axes[0].plot(xs, ys, ls, color=color,
                             label=r["name"] + tag, alpha=0.85, linewidth=1.4)
            if r["val_steps"]:
                xs, ys = zip(*[(s, l) for s, l in zip(r["val_steps"], r["val_losses"]) if s > 0])
                axes[1].plot(xs, ys, ls, color=color, marker=marker,
                             label=r["name"] + tag, alpha=0.85, linewidth=1.4, markersize=5)
    healthy_train = [l for r in results if r["name"] not in diverged for l in r["train_losses"]]
    healthy_val = [l for r in results if r["name"] not in diverged for l in r["val_losses"]]
    if healthy_train:
        axes[0].set_ylim(min(healthy_train) * 0.9, max(healthy_train) * 1.5)
    if healthy_val:
        axes[1].set_ylim(min(healthy_val) * 0.9, max(healthy_val) * 1.5)
    axes[0].set(xlabel="step (log)", ylabel="train loss", title="train loss (log-log)")
    axes[1].set(xlabel="step (log)", ylabel="val loss", title="val loss (log-log)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"[plot] 已保存 {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", type=Path)
    p.add_argument("--plot", type=Path, default=None)
    p.add_argument("--report", action="store_true", help="打印每个 run 的报告数字块")
    args = p.parse_args()

    results = [parse_log(f) for f in args.logs if f.is_file()]
    if not results:
        print("没有可解析的日志文件")
        return

    header = f"{'run':<22} {'steps':>6} {'final_train':>12} {'best_val':>10} {'val@last':>10} {'ppl@last':>10}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda r: r["name"]):
        print(summary_row(r))

    if args.report:
        print("\n========== 报告数字(每条结论直接抄这里) ==========")
        for r in sorted(results, key=lambda r: r["name"]):
            print(report_block(r))
        print("==================================================")

    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        make_plot(results, args.plot)


if __name__ == "__main__":
    main()
