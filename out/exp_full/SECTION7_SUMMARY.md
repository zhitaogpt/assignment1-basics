# §7 实验 Summary（官方配置：TinyStories 全量训练 + LR/Batch 扫描 + 消融）

> 数据来源：`../logs_full/step5_report.log`（`bash run_7_full.sh report` 对本目录 `*.log` 的解析）
> 编排脚本：`run_7_full.sh`；逐步执行日志在 `../logs_full/step{1..5}_*.log`
> 本文档位于 exp_full/ 下，文中文件名路径均相对本目录
> 日期：2026-10-29 ｜ 硬件：2× NVIDIA A800-SXM4-80GB

## 实验设置

- **架构（§7.2.1 官方固定，未改）**：vocab=10000, ctx=256, d_model=512, d_ff=1344, layers=4, heads=16, RoPE θ=10000, pre-norm RMSNorm, SwiGLU
- **数据**：TinyStories 全量，自训 10k BPE 编码 → train 541,229,347 token / valid 5,465,883 token（`data/ts_{train,valid}_10k_full.npy`）
- **两档预算**：`lite`=5000 步（41M token，调参用）；`full`=40000 步（327.68M token，官方目标，交付用）
- **基础超参**：batch=32, warmup=100, cosine 对齐 num_steps, grad clip 1.0；lr 扫描后选定 **lr_max=1e-3, lr_min=1e-4**

## 总表（best_val = 全程最优验证 loss）

| run | profile | steps | final_train | best_val | ppl@last |
|---|---|---|---|---|---|
| **baseline** | **full** | 40000 | 1.4080 | **1.3335 @38000** | **3.82** |
| lr_3e-4 | lite | 5000 | 1.7633 | 1.8224 @4500 | 6.19 |
| **lr_1e-3** | lite | 5000 | 1.7349 | **1.6716 @4000** | 5.32 |
| lr_3e-3 | lite | 5000 | 2.0999 | 2.2343 @4500 | 9.34 |
| lr_1e-2 ⚠发散 | lite | 5000 | 3.2864 | 3.2691 @4500 | 26.29 |
| lr_3e-2 ⚠发散 | lite | 5000 | 3.6233 | 3.6378 @4500 | 38.01 |
| lr_1e-1 ⚠发散 | lite | 5000 | 3.9840 | 3.8646 @4500 | 47.68 |
| lr_3e-1 ⚠爆炸后回落 | lite | 5000 | 3.8890 | 3.9900 @4500 | 54.05 |
| batch_1 | lite | 5000 | 2.3836 | 2.9129 @4500 | 18.41 |
| batch_8 | lite | 5000 | 1.9322 | 1.9882 @4500 | 7.30 |
| batch_32 | lite | 5000 | 1.6057 | 1.6397 @4500 | 5.15 |
| batch_64 | lite | 5000 | 1.5473 | 1.5726 @4000 | 4.82 |
| **batch_128** | lite | 5000 | 1.4749 | **1.4682 @4500** | 4.34 |
| ablation_pre_norm（基线架构） | lite | 5000 | 1.5755 | 1.6475 @4000 | 5.44 |
| ablation_post_norm | lite | 5000 | 1.6409 | 1.6446 @4500 | 5.18 |
| ablation_silu | lite | 5000 | 1.6360 | 1.6575 @4500 | 5.25 |
| ablation_nope | lite | 5000 | 1.8264 | 1.7586 @4500 | 5.80 |
| ablation_no_rmsnorm | lite | 5000 | 1.9331 | 1.8461 @4500 | 6.34 |
| ablation_no_rmsnorm_lowlr | lite | 5000 | 2.6646 | 2.7105 @4000 | 15.23 |

图：`curves.png`（full，log-log）；`curves_4900steps.png`（全部 lite run，log-log）。

## 结论

1. **交付指标达成**：full baseline（328M token）**best_val=1.3335（@38000 步），final 1.3405，ppl 3.82**，低于 ≤1.45 的目标线。val 曲线到 40k 步仍在缓慢下行，未出现过拟合回升。
2. **学习率（lite 扫描 7 档，learning_rate a/b）**：**1e-3 最优**（1.6716）；3e-4 欠拟合（1.82）；3e-3 偏差（2.23）。**发散/爆炸形态（b 问，edge of stability）**：
   - 1e-2 / 3e-2：loss 冲上高位后分别卡在 ~3.3 / ~3.6 下不来（未 NaN，实质性失败）；
   - **1e-1**：直接 plateau 在 ~3.9，全程没有有效学习；
   - **3e-1：真正"爆炸"——train loss 在 ~step 400-600 飙升到 237（val 冲到 ~130），随后被梯度裁剪（max_grad_norm=1.0）+ AdamW 自适应拉回，高位震荡后回落到 ~3.9 plateau**。没有 NaN，裁剪兜住了底但救不了收敛。
   - 规律：发散阈值在 3e-3（尚能学，只是差）与 1e-2（失败）之间；最优 lr 1e-3 距离发散边界约一个数量级——符合"edge of stability"经验：越过边界后 loss 不是平滑变差而是量级式跳变（1.67 → 3.27 → 3.64 → 3.86 → 3.99），且 lr 越大越早进入高 plateau。图见 `lr_curves.png`（3e-1 的尖峰清晰可见）。
3. **Batch size（batch_size_experiment，固定 5000 步口径）**：1→128 单调变好（2.91→1.47）。注意口径：固定步数下大 batch 同时吃到"更多 token"的红利，此对比是"等步数"而非"等 token"。
4. **消融（§7.3 四问，lite，lr=1e-3）**：
   - **post_norm（1.6446）≈ pre_norm（1.6475）**：4 层浅网下两者基本打平，pre-norm 的稳定性优势在浅网不明显；
   - **silu（1.6575）略差于 SwiGLU（1.6475）**：门控分支有稳定但不大的收益；
   - **nope（1.7586）明显差**：RoPE 位置信息贡献显著（+0.11 val loss）；
   - **no_rmsnorm（1.8461）最差**：在官方架构+1e-3 下未发散但全程落后；**降到 1/10 lr（1e-4）反而更差（2.7105）——低 lr 救不了无 norm，只是学得更慢**。
5. **采样质量（generate）**：greedy / T=0.8 / T=0.8+top_p=0.9 输出连贯的 TinyStories 风格故事并正确以 `<|endoftext|>` 收尾；T=1.5 明显退化（符合温度预期）；贪心两次自检一致、生成 id 全部合法。样例见 `generate.txt`。

## OWT main_experiment（已完成）

同架构、同 328M token 预算（40000 步）、lr=1e-3，在 OpenWebText 全量上训练（train 27.3 亿 token / valid 6640 万；32k BPE 训了 3h，编码用 128 核并行 `parallel_encode.py` 13 分钟完成）：

| run | steps | final_train | best_val | ppl@last |
|---|---|---|---|---|
| baseline（TinyStories） | 40000 | 1.4080 | **1.3335** | **3.82** |
| main_owt（OpenWebText） | 40000 | 4.0658 | **3.9006 @38500** | **51.33** |

**对比解读（对应 main_experiment 的两个 deliverable）：**

- 同架构同预算下 OWT 的 val loss 比 TinyStories 高 ~2.57（3.90 vs 1.33）。这不是模型问题，而是数据分布差异：TinyStories 是合成的小词汇儿童故事（10k 词表、句式高度模板化，条件熵天然低）；OWT 是真实网页抓取（主题杂、噪声大、32k 词表、长文档结构复杂），每 token 固有不确定性高得多。**跨数据集的 per-token loss 不可直接比较"模型好坏"，只能比较"数据有多可预测"**。
- 词表也变了（10k→32k），随机基线从 ln(10000)=9.21 升到 ln(32000)=10.37；相对各自随机基线，TinyStories 降到 14.5%，OWT 降到 37.7%——OWT 的学习难度依然显著更高。
- 末段 val 有轻微回升（3.9006@38500 → 3.9382@40000），328M token 对 OWT 这种分布来说接近当前配置的容量/预算边界。
- 采样：OWT checkpoint 的生成在 `generate_owt.txt`（prompt "The history of the"）。语法像英文但连贯性、事实性明显差于 TinyStories 样例——同预算下数据复杂度决定生成质量上限。
- 训练插曲：OWT 训练中段曾因 npy memmap 在 OSS 网盘上冷缓存缺页被 IO 饿死 ~2h（GPU 0%、进程 D 状态），顺序 `cat` 预热页缓存后不重启恢复满速。教训：OSS + memmap 训练前先预热或拷本地盘。

## 复现

```bash
bash run_7_full.sh data                                   # step1 全量编码(一次性,~52min)
PROFILE=lite DEVICE=cuda:0 bash run_7_full.sh lr          # step2 lr 扫描
export LR=1e-3                                            # step2 选出的最优
PROFILE=lite DEVICE=cuda:0 bash run_7_full.sh ablation    # step3 终端1
PROFILE=lite DEVICE=cuda:1 bash run_7_full.sh batch       # step3 终端2
PROFILE=full DEVICE=cuda:0 bash run_7_full.sh baseline    # step4 交付 full(~70min)
bash run_7_full.sh generate                               # step4 采样
bash run_7_full.sh report                                 # step5 汇总
```

注意事项：

- 训练 stdout 经管道 tee 时是块缓冲（~8KB），full run 的日志会分段出现，属正常现象。
- 未固定随机种子，重跑数值小幅波动，结论方向稳定。
- 画图约定：log-log 双对数坐标；full/lite 自动分图；发散 run 图例标 "(diverged)" 且不参与 y 轴上限计算。
