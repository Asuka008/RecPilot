# 17：平均池化神经对照——只平均当前可见前缀

日期：2026-10-22。**实现、配置与服务器验证入口已完成；本地 5 个纯 Python
配置测试、语法与入口检查通过。用户已返回三轮服务器 smoke PASS；12 个聚焦
CPU 测试的结尾尚未返回，全量对照待运行。当前没有完整效果或性能结论。**

## 这个对照回答什么

原纯 ID HSTU 与固定频次校正已有一组单种子结果，见[笔记 16](learning-16-corrected-test-results.md)。
现在检查：当前短历史数据上，复杂 HSTU 编码器相比“历史 ID 向量平均”带来
多少收益。它不是更换项目主模型，也不是预设能解决低频问题。

先比较**未校正的 PrefixMean 与未校正的 HSTU**。编码器和打分校正的作用
分开记录，不能拿原始平均池化与校正后的 HSTU 直接归因到编码器。
原 HSTU 验证参考为 Recall@10/20=0.07112/0.10176，NDCG@10/20=
0.03772422803068947/0.04544371132907886。

## 数据与计算路径

新增目录 `controls/` 保存该对照。共享 `src/` 和此前两个打分诊断/评估脚本
没有改动，因此旧检查点的严格源码身份仍可复核。

| 文件 | 职责 |
|---|---|
| `controls/mean_pooling.py` | PrefixMean 编码器与原候选打分头复用 |
| `controls/config.py` | 与明确的 HSTU 参考配置逐项匹配 |
| `controls/run_mean_pooling.py` | 训练、验证选择、按轮次恢复与显式评估 |

输入继续来自 SASRecTrainDataset：`input_ids`、`target_ids` 和 bool
`padding_mask` 均为 `[B,L]`。例如训练历史 `[1,2,3,4]` 生成输入
`[0,1,2,3]`、标签 `[0,2,3,4]`。各有效位置分别预测 2、3、4。

每个位置 t 的输出为：

`query_t = normalize(mean(该位置及其之前的有效输入 ID 向量))`

因此三个预测只使用 `[1]`、`[1,2]`、`[1,2,3]`。不能先平均完整输入
`[1,2,3]` 再把结果复制给三个位置，否则预测 2 时已经看到了未来的 2 和 3。

模型复用 compact_ids 将有效 ID 保序紧凑到左边，查 embedding 得到
`[B,L,D]`，应用与参考相同概率的输入 dropout。padding 向量被置零。
固定下三角矩阵 `[L,L]` 与输入相乘得到各前缀和；紧凑后有效位置的计数为
1、2、…，相除得到均值，再 L2 归一化、恢复调用者布局并将 padding 输出置零。
重复观看按事件重复计入，不去重。空输入的输出全零；评估仍拒绝空历史。

使用下三角矩阵而不是浮点 CUDA cumsum，是为了兼容现有严格确定性模式：
[PyTorch 2.10 文档](https://docs.pytorch.org/docs/2.10/generated/torch.use_deterministic_algorithms.html)
将该版本的 CUDA 浮点 cumsum 列为确定性模式下报错的操作。本实现是 L=32 的
稠密参考路径，计算量为 O(B·L²·D)，不将它宣传为已实现线性时间优化。

输出仍为 `[B,L,D]`，共享 full_catalog_loss 用有效 mask 取 `[M,D]`，对完整
目录打分为 `[M,N]`。候选归一化、共享输入/输出 embedding、温度、CE 和
有效预测位置加权全部复用原实现。训练标签不含验证/测试交互。

验证继续使用训练历史；测试历史为 train 加 val。模型读最近 L 个物品，排名
过滤完整可见历史；目录、padding 排除、稳定同分顺序和频次分组不变。

## 如何控制比较条件

每份控制配置显式指定 reference_config；除编码器标识和输出目录外，原实验
字段必须与该 HSTU 配置完全一致，否则拒绝启动。包括数据路径、模式、种子、
批量、长度、维度、dropout、学习率、温度、weight decay、cutoff、用户抽样与轮数。
控制配置不允许额外 alpha，首轮没有候选分数校正。

为了匹配初始 ID 表与起始随机数状态，模型在 CPU 短暂构造同配置、**未训练**
的 IDHSTU，仅保留它的 item_embedding 与 input_dropout。临时对象的 attention
与位置参数丢弃，不参与前向、优化或 checkpoint。不会复制原已训练检查点。
这样候选打分还能直接复用原 IDHSTU.score_candidates，实现保持一致。

模型只有 `item_embedding.weight` 一个可训练张量，无学习的位置向量或 attention
参数。运行记录实际参数量、对应 HSTU 参数量和初始 ID 表 SHA-256。
ID 表占主要参数，不能假定移除复杂编码后参数量或总耗时就会大幅下降。

它仍不是严格隔离所有结构因素的实验：位置表示和 block dropout 等结构同时
被移除，两个模型优化难度也可能不同。相同种子和初始 ID 表不保证训练期间
所有 dropout 随机调用始终逐一对齐；预算与主要超参相同不代表各自调至最优。

## 配置与阶段门禁

| 配置 | 对应参考与用途 |
|---|---|
| `configs/mean-pooling-debug.json` | 对应 HSTU debug，CPU 16 用户、2 轮、8 维；可选服务器排错 |
| `configs/mean-pooling-smoke-server.json` | 对应 HSTU smoke，CUDA FP32，1,024 训练/验证用户，3 轮 |
| `configs/mean-pooling-train-server.json` | 对应 HSTU full，50,000 用户、seed 42、30 轮、训练/验证 batch 32/64 |

model 字段保留 HSTU 参考构造参数以复现初始化；它不表示平均池化含有 2 个
HSTU block。真正训练的结构由 control=causal-prefix-mean-v1 标识。

现在只执行服务器 CPU 合约测试和三轮 CUDA 短训练，先不自动启动全量训练：

```bash
cd "$HOME/AgenticRec"
git pull --ff-only
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
python -m unittest tests.test_mean_config tests.test_mean_pooling tests.test_mean_runner -v
nvidia-smi
python -m controls.run_mean_pooling train --config configs/mean-pooling-smoke-server.json
```

预期 **12 个测试、结尾 OK**，覆盖：

- 配置配对和预算一致，结果目录分离，拒绝不匹配设置与打分校正。
- 初始 ID 表/RNG 与参考相同，候选打分一致，只有 ID 表可训练。
- 手算前缀、左右 padding 等价、空历史、重复事件和完整前缀内顺序不变性。
- 未来输入不能改变早期状态，也不能收到该早期状态产生的输入梯度。
- padding 标签不参与损失、有限梯度及参数更新。
- 带 dropout 的两轮完整训练与第一轮后恢复，参数/优化器/RNG/指标精确一致，
  best.pt 独立验证与保存记录一致，smoke 不允许测试集。

因果梯度检查只对早期隐藏状态求导。全目录 CE 会给许多候选 ID 表梯度，
不要求损失对某个未来物品类别的梯度为零；候选类别更新与读取未来输入是两回事。

短训练先打印控制/参考参数量，再打印三轮 loss、probe loss 和验证 NDCG@10。
预期最后 `Mean pooling train: PASS`。要求参数更新、固定训练 probe 损失下降、
检查点加载输出完全一致、数据哈希未变；不要求短训练优于 HSTU。
请返回测试结尾、三轮输出和结果路径，随后核对实际设备、参数量、初始化身份、
checks、耗时和显存。实际设备仍需重新检查，环境名不代表 H200 分配。

## 产物与恢复

结果保存到配置指定的 `runs/mean-pooling*/<UTC时间>/`：config.json、
results.json、best.pt 和 last.pt。每轮只评估验证，以验证 NDCG@10 选择 best，
完全同分保留较早轮次。输出 raw scores，不应用之前选出的 alpha。

Checkpoint 复用 state_dict、配置、epoch/step、优化器、指标、实验身份与 RNG
约定。新身份同时记录未改动的共享 src 哈希与明确的 controls 文件哈希，以及
HSTU 参考配置快照/哈希；不删除或绕过旧身份检查。

按完整轮次边界恢复（占位符换成真实目录，保持 Git/配置/环境版本不变）：

```bash
MEAN_PREVIOUS_RUN="runs/mean-pooling-smoke/<实际UTC目录>"
python -m controls.run_mean_pooling train --config configs/mean-pooling-smoke-server.json --resume "$MEAN_PREVIOUS_RUN/last.pt"
```

保留原 sibling best.pt，继续执行创建新目录。可用 `--stop-after-epoch 1`
显式停在第一轮，结果 PARTIAL；不支持半轮精确恢复。

## 短训练通过后

资源和恢复检查通过后，再审核并运行完整候选：

```bash
python -m controls.run_mean_pooling train --config configs/mean-pooling-train-server.json
```

先比较两种未校正模型的完整验证总体/切片、参数量和实际设备下的耗时/显存。
不要因为测试结果再选配置。训练不会自动评估测试；设置冻结后才显式运行
`evaluate --config CONFIG --checkpoint RUN/best.pt --split test`，smoke 配置拒绝 test。
如果平均池化相近，只能说明当前配方下复杂编码增益有限；如果落后，再检查
收益所在群体。都不能直接证明因果或解决低频。

本地没有 PyTorch，未执行模型或 DataLoader。已完成的本地检查是 5 个配置
测试、源码编译、CLI help、diff 格式，以及共享 src/旧诊断脚本未改动检查。
服务器 smoke 已通过；聚焦 CPU 测试输出、完整资源报告、全量结果与实际性能仍待核对。


## 服务器 smoke 记录（2026-10-22）

用户返回 `Mean pooling train: PASS`，报告路径为
`runs/mean-pooling-smoke/20261022T155527051279Z/results.json`。

| 轮次 | 训练 loss | 固定 probe loss | 验证 NDCG@10 |
| --- | --- | --- | --- |
| 1 | 10.029359 | 9.183881 | 0.000836 |
| 2 | 8.812370 | 8.000587 | 0.004429 |
| 3 | 7.695507 | 6.918766 | 0.006182 |

Mean 参数 1,230,144，参考 HSTU 参数 1,273,406，仅减少约 3.40%。
两者的参数主要都在 ID 表，编码计算更简单不代表模型文件大幅缩小或已经测得加速。
PASS 按入口要求包含参数更新、固定 probe 下降、恢复输出精确一致及数据哈希未变；
完整 checks、运行时与显存字段尚未返回，当前 GPU 型号不能从环境名推断。

这是流程正确性与可训练性门槛，不能用这三轮的 NDCG 与正式 HSTU 比较。
下一步确认 12 个 CPU 测试 OK，检查当前设备，再用现有完整配置训练 30 轮。
以验证 NDCG@10 选择 best，比较未校正 HSTU 的 0.03772422803068947，
同时查看目标频次/历史长度切片；先不运行测试集，也不根据测试指标调参。
