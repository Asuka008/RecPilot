# 12：首个 MicroLens 全量 HSTU 训练候选

日期：2026-10-21。**33 个服务器 CPU 测试、三轮短训练及资源审核已通过；
首轮全量 30 轮训练与冻结模型的测试评估均已回传 PASS。
结果和下一步验证诊断见[笔记 13](learning-13-microlens-baseline-results.md)。**

## 为什么可以进入全量训练

用户回传短训练报告：实际设备为 A100 MIG 2g.20gb，约 19.50 GiB 可见显存，
PyTorch 2.10.0+cu128、CUDA runtime 12.8；峰值 allocated/reserved 为
0.154/0.193 GiB，总耗时约 17.091 秒。ID/HSTU 参数均有更新，固定训练 probe
损失下降，epoch 3 的检查点输出精确一致，处理后文件哈希未变。
完整数值和边界见[笔记 11](learning-11-microlens-hstu.md)。

这验证了训练路径，并没有证明泛化收益。接下来扩大用户覆盖，保留其他设置，
先获得可信的全量验证曲线。显存有余量也不必同时扩大 batch；batch 会改变更新
次数和优化过程。正式实验仍需记录实际峰值，不能把短训练显存当作全量上限。

## 首轮固定设置

配置：`configs/hstu-microlens-train-server.json`，本次没有改动其数值。

| 项目 | 设置 |
|---|---|
| 数据 | 既有 MicroLens 处理文件；50,000 个训练用户、50,000 个验证用户 |
| 目录 | 19,220 个正 ID；0 为 padding |
| 初始权重 | 新初始化，seed 42；不恢复 1,024 用户 smoke 权重 |
| 模型 | 纯 ID、位置偏置，L=32、D=64、2 block、2 head，QK/V 每 head 为 32 |
| 正则 | dropout 0.1、weight decay 0.0001 |
| 训练 | FP32、TF32 关闭；30 轮，batch 32，worker 0 |
| 优化 | AdamW，lr=0.001，betas=(0.9,0.98)，梯度裁剪范数 1 |
| 目标 | 有效移位位置的全目录 CE；归一化共享 embedding，temperature=0.2 |
| 验证 | 每轮，全目录，batch 64；Recall/NDCG@10/20 |
| 选择 | 最大验证 NDCG@10；相同分数保留较早轮次 |

30 轮是预先设置的首轮计算预算，不是已经找到的最优轮数。不依据测试分数调整
轮次、学习率、序列长度或模型。现有数据约定的完整历史过滤、稳定同分排序、
零频目标保留和切片保持不变；模型输入形状与关键路径见笔记 11。

## 服务器命令

在现有服务器仓库和 reference 环境中执行：

```bash
cd "$HOME/AgenticRec"
git pull --ff-only
source /opt/conda/etc/profile.d/conda.sh
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
git status --short --branch
git log -1 --oneline
nvidia-smi
python -m src.run_hstu_microlens train --config configs/hstu-microlens-train-server.json
```

代码没有变化，已通过的 33 个测试无需因为本次说明更新而重复运行。
启动前确认服务器没有源代码改动、设备可用；环境名称不代表实际分配。
运行期间保持终端会话和代码/配置版本稳定，不同时启动另一个训练。

每轮应打印 `Epoch n/30`、训练损失、固定训练 batch probe 和验证 NDCG@10。
这里训练损失是按有效预测位置平均，验证指标是按用户平均；probe 只是一个固定
训练 batch，不能代替全量验证。无需损失或验证指标逐轮单调改善。

## 产物、通过标准与恢复

独立目录：`runs/hstu-microlens/<UTC时间>/`。每个完成的轮次写入：

- `results.json`：每轮训练/验证及切片、选择结果、代码/数据/配置/环境身份。
- `best.pt`：验证 NDCG@10 选择的模型与优化器状态及 RNG。
- `last.pt`：最近完整轮次的状态，用于同版本继续训练。
- `config.json`：本次配置快照。

正常结束为 epoch 30、`MicroLens HSTU train: PASS`，结果中应有 30 条 history、
验证 all 组 50,000 用户、train_users=50,000、evaluation_users=50,000，
且检查点输出一致、参数更新和数据哈希门禁通过。PASS 表示执行与检查成功，
不以胜过 ItemCF 为运行成功条件。全量模式记录 probe 是否下降，但不以此单个
batch 的下降作为质量门槛。

请返回训练末尾输出、结果路径，之后再读取 best_epoch、best_score、完整验证
指标、显存和切片。先与 ItemCF 验证 NDCG@10=0.024495、Recall@10=0.04804
比较；还需看其他 cutoff 和分组，再决定是否固定配置或进行有限验证集调参。
不预设 HSTU 一定优于 ItemCF，不用 ML-1M 分数替代此实验。

若中断，保留原目录的 `last.pt` 与对应 `best.pt`，在相同代码、配置和运行
环境继续：

```bash
MICROLENS_PREVIOUS_RUN="runs/hstu-microlens/<中断训练的实际UTC目录>"
python -m src.run_hstu_microlens train --config configs/hstu-microlens-train-server.json --resume "$MICROLENS_PREVIOUS_RUN/last.pt"
```

占位符须替换为实际路径。继续执行会创建新目录，并继承此前完整轮次的 history
和最佳模型。只支持完整轮次边界恢复，半轮进度会丢弃。训练期间不要拉取新提交，
因为入口会核对 Git、源码哈希、配置、数据与运行环境。合成 CPU 恢复一致性测试
已通过；真实数据的独立中断/恢复仍无回传记录。

**训练命令不会评估测试集。** 首轮设置现已按下方记录冻结；首轮测试已完成；跨种子
稳定性和性能结论仍需分别验证。

## 首轮全量训练通过记录与测试配置冻结

用户回传 30 个完整 epoch 和 `MicroLens HSTU train: PASS`，结果路径：
`runs/hstu-microlens/20261022T023420952080Z/results.json`。

| 轮次 | 训练损失 | 固定训练 probe 损失 | 验证 NDCG@10 |
|---|---:|---:|---:|
| 1 | 9.077149 | 8.698148 | 0.019292 |
| 10 | 8.234203 | 8.267822 | 0.036570 |
| 28（日志中最佳） | 8.134221 | 8.222760 | 0.037724 |
| 30 | 8.120724 | 8.200473 | 0.037618 |

比固定 ItemCF 的验证 NDCG@10=0.024494992910568 约增加 0.013229，
相对约 +54.0%，这里使用日志六位小数，应以报告中的未舍入结果为最终记录。
训练损失整体下降，后期验证指标有波动；不要求每轮单调，也不因第 30 轮略低
就改选择规则。按预先定义的验证规则使用 best.pt，而不是最后一轮 last.pt。
单次结果不能隔离全部架构因素，也不能证明统计显著、跨种子稳定或线上收益。

当前冻结首轮配置、seed=42、30 轮预算与 NDCG@10 验证选择规则，开展一次
固定的基线测试评估。不继续增加轮次或看测试结果选参数。后续诊断、简单神经
对照和改进应作为独立实验，用验证数据决定。

先读取精确选择、全量验证切片、检查及资源摘要：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("runs/hstu-microlens/20261022T023420952080Z/results.json")
r = json.loads(p.read_text())
assert r["status"] == "PASS" and len(r["history"]) == 30
best = next(row for row in r["history"] if row["epoch"] == r["selection"]["best_epoch"])
summary = {k: r.get(k) for k in ("status", "data", "selection", "checks", "checksums_unchanged", "elapsed_seconds")}
summary["best_validation"] = best["validation"]
summary["git"] = r["provenance"]["git"]
summary["data_sha256"] = r["provenance"]["data_sha256"]
summary["runtime"] = {k: r["runtime"].get(k) for k in ("torch", "cuda_runtime", "device", "gpu", "gpu_total_memory_bytes")}
summary["memory_gib"] = {k: round(v / 1024**3, 3) for k, v in r.get("memory_bytes", {}).items()}
print(json.dumps(summary, indent=2))
PY
```

然后执行测试评估；不重跑训练、不更改配置、不需要更新环境：

```bash
python -m src.run_hstu_microlens evaluate --config configs/hstu-microlens-train-server.json --checkpoint runs/hstu-microlens/20261022T023420952080Z/best.pt --split test
```

预期输出 split=test、all 组 50,000 用户的 Recall/NDCG@10/20，以及频次与
历史长度分组，最后 `MicroLens HSTU evaluate: PASS` 和新 results.json 路径。
评估会核对配置、数据身份和源码身份；训练未被本次说明更新修改。
回传上述摘要、测试指标与结果路径后，再与固定 ItemCF 测试分数比较和分析切片。
全量训练的实际硬件/耗时/显存和精确选择字段尚未回传，不能套用 smoke 数字。


## 冻结测试已通过

用户按本页 best.pt 命令完成测试评估并返回 `MicroLens HSTU evaluate: PASS`。
测试 run 为 `runs/hstu-microlens/20261022T030714112850Z/results.json`。
四项总体指标、切片与下一步验证诊断见[笔记 13](learning-13-microlens-baseline-results.md)。
不用为了收集资源/验证摘要重新训练或重复测试；读取已有报告即可。
