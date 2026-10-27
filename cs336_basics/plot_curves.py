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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in results:
        if r["train_steps"]:
            axes[0].plot(r["train_steps"], r["train_losses"], label=r["name"], alpha=0.8)
        if r["val_steps"]:
            axes[1].plot(r["val_steps"], r["val_losses"], "o-", label=r["name"], alpha=0.8)
    axes[0].set(xlabel="step", ylabel="train loss", title="train loss")
    axes[1].set(xlabel="step", ylabel="val loss", title="val loss")
    for ax in axes:
        ax.grid(alpha=0.3)
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
