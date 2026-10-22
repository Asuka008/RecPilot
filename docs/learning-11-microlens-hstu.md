# 11：把纯 ID HSTU 接到 MicroLens

更新日期：2026-10-21。**33 个聚焦服务器 CPU 测试与三轮真实数据短训练已通过。
实际设备、显存和报告字段待核对，全量训练仍待执行。**

## 当前阶段与目标

Popularity/ItemCF 全量验证、测试和官方 ML-1M 101 轮训练已有用户回传的服务器结果。
本次接入的是自己的 MicroLens 训练入口，不会启动或修改官方 ML-1M 实验。
第一步完成标准是：真实数据短训练、有限损失/梯度、参数更新、检查点恢复和
共享评估全部通过。它还不能证明 HSTU 优于 ItemCF。

入口是 `src/run_hstu_microlens.py`；训练与批量评估在 `src/microlens_training.py`；
配置检查和用户抽样在 `src/microlens_config.py`。复用现有 Dataset、IDHSTU 和
`src/checkpoints.py`。`src/evaluation.py` 的 RankingAccumulator 统一基线和
神经模型的排名与分组分母，原有目标频次结果字段保留，新增历史长度分组。

## 数据流与形状

设批量 B、最大长度 L、维度 D、目录物品数 N：

1. `load_data` 检查三个处理后文件、连续可逆映射、用户/物品/交互计数并计算哈希。
2. `SASRecTrainDataset` 只从 `train` 生成 `input_ids`、`target_ids` 和
   `padding_mask`，均为 `[B,L]`；ID 为 int64，mask 为 bool。这里保留 SASRec
   命名只是为了复用数据约定，不表示训练 SASRec。
3. 例子：`train=[1,2,3]`，L=4，则输入 `[0,0,1,2]`、标签 `[0,0,2,3]`。
   正向预测只用当前物品及过去物品，padding 位置不监督。
4. IDHSTU 内部把非零 ID 按原顺序紧凑到左边，供右侧 padding 的 HSTU block
   使用，再恢复调用者布局。输出 `[B,L,D]`，因此标签仍与输出位置一一对应。
5. mask 选出 M 个有效预测位置，得到 `[M,D]`；共享 embedding 给全部
   ID `1..N` 打分，得到 `[M,N]`，标签从 `1..N` 转成 CE 的 `0..N-1`。
6. 验证输入是训练历史，测试输入是训练历史加验证物品；取左侧 padding 输入
   最后一个位置的状态 `[B,D]`，全目录打分 `[B,N]` 后用共享规则排名。

模型可能只看最近 L 个历史物品，但排名过滤使用**完整可见历史**。
目标若出现在其可见历史中就报错，不按目标临时改过滤规则。候选 padding=0
永远排除，同分按映射 ID 升序；零训练频次目标保留在指标分母中。

## 为什么先用全目录交叉熵

当前目录 N=19,220，先用小批量 FP32，直接计算：

`loss = mean(-log softmax(cosine(query, all_item_embeddings) / temperature)[target])`

输入和候选共用 ID embedding，查询和候选都做 L2 归一化。默认 temperature=0.2，
损失按有效预测位置平均，长历史用户可能贡献更多预测位置；不是每用户等权的训练损失。
评估则仍按用户等权。训练分母不按用户历史过滤，也不根据验证/测试目标删负类。
未交互候选只是分类任务中的其他类，不是已知曝光负例。

这避免了首轮接入就引入负采样规则，但不是零成本：训练 logits 最多约
`B*L*N` 个 FP32 元素，batch=32、L=32 时仅这一张量上限约 75 MiB，
反向传播和优化器还需额外显存，实际以服务器峰值为准。零频物品虽然没有正向训练
事件，仍会在分类分母中收到梯度，不能称为训练完全没接触过的类。

这是 MicroLens 的明确适配：**位置偏置、无时间戳特征、全目录 CE**，与官方
ML-1M 时间偏置和 sampled softmax 的整套配方不同。官方复现和 block 对齐
不等于这个新训练入口已经通过验证。

## 配置与检查点选择

| 配置 | 用途 |
|---|---|
| `configs/hstu-microlens-smoke-server.json` | 3 轮、batch 32、1,024 训练用户和 1,024 验证用户，CUDA FP32 |
| `configs/hstu-microlens-train-server.json` | 全量、30 轮的候选起点；短训练通过后再审核 |
| `configs/hstu-microlens-debug.json` | 可选 CPU 排错，16 用户、2 轮、8 维小模型 |

服务器 smoke 模型 L=32、D=64、2 block、2 head，dropout=0.1。
选择 L=32 是短历史数据的保守起点，较长用户会被截断；以后可在验证集检查
截断影响。用户抽样只看记录位置与 seed，不读取目标内容，也不消耗训练随机状态。
训练和验证抽样使用不同 seed，可有用户重叠，这符合按用户留出交互的既定协议。
smoke 频次统计只使用其抽取的训练用户，不能拿 smoke 切片和全量基线直接比较。

训练每轮只评估验证集，以 NDCG@10 选择 `best.pt`，完全同分保留较早轮次；
`last.pt` 保存当前轮次以便继续。full 模式不允许用户数量上限，smoke 模式
不允许测试集评估。测试是单独命令，待模型设置与验证选择冻结后才运行。

频次切片为 0、1–5、6–20、21+，历史长度切片为 1–5、6–10、11–20、21+，
都附用户数；空组指标为 null。历史长度按完整可见历史计数，不按截断后长度。

## 服务器：现在先运行这些

先按 Git 流程同步本次代码和配置。不要在服务器手改源文件，也不需要重装
已经通过官方实验的环境。以下命令从仓库根目录运行：

```bash
cd "$HOME/AgenticRec"
git pull --ff-only
source /opt/conda/etc/profile.d/conda.sh
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
nvidia-smi
python -m unittest tests.test_dataset tests.test_baselines tests.test_hstu_training tests.test_microlens_config tests.test_microlens_training -v
python -m src.run_hstu_microlens train --config configs/hstu-microlens-smoke-server.json
```

测试只读临时合成数据，在服务器 CPU 执行，包括：全目录损失与移位标签、
padding 标签忽略、完整历史过滤、同分排序、零频分母、与基线评估一致，以及
带 dropout 的两轮完整训练和一轮后恢复训练之间的参数、优化器、RNG 精确一致。
不会启动真实数据全量训练。模型既有因果/padding 对齐检查沿用已有服务器记录。

短训练每轮打印训练损失、固定训练 batch 的 probe loss 和验证 NDCG@10。
正常最后为 `MicroLens HSTU train: PASS`，并保存 `results.json` 路径。
PASS 要求梯度/权重有限、ID 和 HSTU 参数确实更新、固定 probe 损失下降、
最后检查点加载后输出完全一致，三个数据文件执行前后哈希不变。
不要求每个 batch 或每轮验证指标单调改善，不预设胜过 ItemCF 的门槛。

请回传测试结尾、三轮输出、最终状态、结果路径和 results.json 的 `checks`、
`memory_bytes`。若遇 OOM，保留失败报告，按实际占用再调整配置；不靠名义
H200 显存推断当前容量。最近确认的设备实际上是 A100 MIG 19.50 GiB。

## 可选：真实数据按轮次恢复检查

CPU 测试通过后，也可在同一服务器运行下列单独检查。先停止在第一轮，保留
配置中的总轮数 3；预期结果 `PARTIAL`，退出码 0 表示按要求完成部分轮次：

```bash
python -m src.run_hstu_microlens train --config configs/hstu-microlens-smoke-server.json --stop-after-epoch 1
```

把输出实际目录填入变量，再继续剩余两轮；这里的占位符必须替换：

```bash
MICROLENS_PARTIAL_RUN="runs/hstu-microlens-smoke/<第一轮输出的UTC目录>"
python -m src.run_hstu_microlens train --config configs/hstu-microlens-smoke-server.json --resume "$MICROLENS_PARTIAL_RUN/last.pt"
```

继续执行写入新的结果目录，原目录保持不动。恢复要求相同配置、数据/抽样用户、
代码哈希/Git 状态、设备/依赖环境，恢复 Python、NumPy、CPU/CUDA RNG。
每轮 DataLoader 顺序由 seed+epoch 独立生成，支持完整轮次边界恢复，**不支持
轮次中间精确恢复**；进程被终止后从最后完整轮次继续，不沿用半轮进度。
原目录 `last.pt` 和其对应 `best.pt` 必须保留在一起。若只有 best，没有 last，
可显式评估 best，但它不是此入口的续训点。

## 短训练通过后再执行

审核学习率、batch、序列长度、耗时/显存和恢复情况后，才启动全量候选配置：

```bash
python -m src.run_hstu_microlens train --config configs/hstu-microlens-train-server.json
```

训练不自动运行测试。冻结设置后，用验证选择的 checkpoint 运行：

```bash
MICROLENS_FULL_RUN="runs/hstu-microlens/<正式训练输出的UTC目录>"
python -m src.run_hstu_microlens evaluate --config configs/hstu-microlens-train-server.json --checkpoint "$MICROLENS_FULL_RUN/best.pt" --split test
```

比较 ItemCF 时使用相同全量数据、候选目录和评估规则。训练目标不同于简单基线，
但评估协议相同；ML-1M 分数不能与 MicroLens 分数直接比较。

## 产物与本地检查

`runs/<配置目录>/<UTC时间>/` 保存 `config.json`、`results.json`、`best.pt`、
`last.pt`。独立评估只保存其配置与结果。报告含每轮指标与切片、probe、代码
文件哈希/Git 状态、上游 revision、三个数据哈希、抽样用户身份、seed、环境、
实际设备/精度、耗时和 allocated/reserved 显存峰值。检查点用 state_dict，
含优化器、配置、epoch、step、指标、实验身份和 RNG，不序列化整个模型对象。
本次没有写入原始/处理后数据，也没有提交日志或模型权重。

本地不安装或运行真实训练。本次使用的轻量检查：

```bash
python -m unittest tests.test_baselines tests.test_microlens_config -v
PYTHONPYCACHEPREFIX=/private/tmp/agenticrec-pycache python3 -m compileall -q src tests
git diff --check
```

配置测试在系统 Python 通过；16 个合并测试使用应用自带的 NumPy 环境通过。
本机没有 PyTorch，因此未在 Mac 执行神经网络测试。后续用户回传服务器测试和短训练 PASS，
见下方记录；本地没有读取或独立复核服务器原始产物。

## 服务器通过记录（本地日期 2026-10-21）

用户回传聚焦测试 `Ran 33 tests in 11.080s`、`OK`，随后使用
`configs/hstu-microlens-smoke-server.json` 完成三轮训练并返回
`MicroLens HSTU train: PASS`。结果路径：
`runs/hstu-microlens-smoke/20261022T022602941154Z/results.json`。
路径时间为 UTC，不将其直接当作用户本地日期。

| 轮次 | 训练损失 | 固定训练 batch probe 损失 | 验证 NDCG@10 |
|---|---:|---:|---:|
| 1 | 9.946178 | 8.586332 | 0.001397 |
| 2 | 8.179191 | 7.200485 | 0.004264 |
| 3 | 7.135582 | 6.391037 | 0.005421 |

这说明训练入口在这次短训练中能够学习，不能拿 1,024 用户的结果与全量
ItemCF 直接比较。33 个 CPU 测试包含合成数据带 dropout 的按轮次恢复一致性；
真实数据的单独中断/恢复命令尚无用户回传结果。按当前入口的 PASS 定义，
参数更新、有限梯度、检查点加载输出一致、probe 下降和数据哈希不变门禁通过；
实际字段仍需读取报告确认。不要将之前 A100 MIG 的设备记录自动套用到此次运行。

下一步先读取这次报告，暂不重跑短训练或启动全量训练：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("runs/hstu-microlens-smoke/20261022T022602941154Z/results.json")
r = json.loads(p.read_text())
summary = {k: r.get(k) for k in ("status", "data", "checks", "checksums_unchanged", "elapsed_seconds")}
summary["git"] = r["provenance"]["git"]
summary["runtime"] = {k: r["runtime"].get(k) for k in ("torch", "cuda_runtime", "device", "gpu", "gpu_total_memory_bytes", "dtype", "tf32")}
summary["memory_gib"] = {k: round(v / 1024**3, 3) for k, v in r.get("memory_bytes", {}).items()}
print(json.dumps(summary, indent=2))
PY
```

核对实际 GPU/分配显存、峰值 allocated/reserved、Git/数据用户数和 checks 后，
再审核全量 batch 32、30 轮等设置。全量训练从新初始化开始，不从 smoke 的
1,024 用户检查点继续；训练只按验证 NDCG@10 选模型，测试仍需冻结后显式执行。
