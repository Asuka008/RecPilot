# 20. 标题初始化与固定融合 HSTU

本阶段要回答：标题中已有的内容关系，能否改善纯 ID HSTU 的稀疏反馈表现？
已有验证发现是 HSTU 总体更强，但低频目标落后于 Mean/ItemCF。它支持做标题
对照，不证明缺少标题就是根因。实现完成，用户已确认 17 个服务器测试全部通过，两个 CUDA 短训练均 PASS。
两个全量 30 轮训练也已由用户回传 PASS；完整验证分组与资源现已回传，报告配对 PASS。当前下一步固定初始化第 30 轮
候选，做一次冻结测试评估以完成这一轮对照；尚不能认定稳定收益。

## 已有产物无需重跑

用户返回完整缓存读取 PASS，路径为
`runs/title-features-minilm/20261023T013804609858Z`。
`features.npy` 形状 `[19221,384]`、FP32，包含第 0 行全零 padding。
SHA256 为 `af5c0482e9b37294e1e4817810d5e2b0b407c22759207078fd402f7ea0525b25`。
19,220 个物品均有标题，10 条被截断；编码约 37.33 秒，峰值 allocated/reserved
分别为 515812864/828375040 bytes。编码器冻结、重复输出精确一致、batch padding
一致、单位向量、保存后精确恢复均已由服务器输出确认。

## 三种表示，保持同一个训练问题

令 E 为可训练 ID 表，S 为固定的 64 维标题表，物品有效表示记为 V。

| 对照 | 初始化 | 历史输入和候选使用的 V | 训练期间 |
|---|---|---|---|
| 既有纯 ID HSTU | 原始随机 E | E | 更新 E 与 HSTU |
| title-init | E 复制 S | E | E 可继续训练，标题只提供起点 |
| title-fusion | 保留原始随机 E | E + 0.5 S | 更新 E 与 HSTU，S 保持冻结 |

0.5 是预先指定的候选配方，不是已经选出的最优权重，也不是此前流行度修正的
alpha。第一轮两个实验都使用原始 cosine 分数，不叠加频次修正。没有频次门控、
没有文本编码器微调、没有额外训练轮数。两个实验均从头训练；不能把已训练好的
纯 ID checkpoint 当作它们的初始权重。

`content_hstu/model.py` 先完整构造原始 `IDHSTU`，再替换物品查表模块。因此同一
seed 的位置表、HSTU 参数和随机数消耗匹配原始模型。初始化模式仅改变物品表；
融合模式连初始 ID 表也保持一致。可训练参数数目与原始 HSTU 相同，额外存储一个
固定 S。固定融合权重不等于固定贡献比例：E 会随训练改变。

输入 `[B,L]` 经 V 查表得到 `[B,L,64]`，沿用原来的位置表示和因果 HSTU。
输出归一化查询 `[B,L,64]`；共享候选 IDs `[K]` 经同一 V 查表并归一化，计算
cosine 分数 `[B,K]`。因此标题同时作用于历史表示与候选表示。padding ID 0
保持零，候选不允许 0；完整历史过滤和同分排序继续使用共享评估代码。

## 为什么需要 PCA，如何避免目标泄漏

冻结 MiniLM 表为 384 维，而已经验证的 HSTU 宽度是 64。当前选择固定 PCA，
用于维度适配，避免新增可训练投影层改变参数预算。这是当前实验配方，不保证
PCA 是最佳语义压缩方法。PCA 最大化标题特征方差，不直接优化推荐质量。

`content_hstu/projection.py`：

1. 仅从所选训练用户的 `train` 历史收集不同物品，每个物品权重相同；不读取
   `val`/`test` 目标来选择拟合行，不按物品流行度重复计权。
2. 在这些行上计算 FP64 均值和协方差，取前 D 个主成分，固定每列符号。
3. 用该均值/主成分变换所有目录物品，归一化后乘 `0.02 * sqrt(D)`，近似匹配
   原始 ID 初始化的行尺度；第 0 行始终零。零方差或退化投影明确报错。
4. 把均值 `[384]`、主成分 `[384,D]`、拟合物品 ID 列表和 S `[N+1,D]` 保存到
   `projection.npz`，记录逐数组哈希和方差保留比例。

短训练仅用其 1024 个训练用户中的物品拟合；全量配方使用全部训练用户中的
物品重新拟合。因此 smoke 和 full 的 PCA 可以不同，二者 checkpoint 不混用。
拟合成员由完整训练历史决定，可能包括因长度截断未作为某次优化输入的物品；
仍然不使用 held-out 目标。零训练频次物品可以通过自己的标题获得 S，前提是本
协议假设目录和这些标题已知；这不是严格按发布时间回放的新物品上线模拟。

跨 NumPy/BLAS 环境，特征分解不保证逐位一致，尤其是重复特征值附近。代码在
续跑/评估时重新推导并核对 content_id；不一致会拒绝执行，而非静默换投影。
当前实验应保留原运行环境。此处不承诺跨平台 checkpoint 迁移。

## 关键入口和可复现性

- `content_hstu/config.py`：强制原配置字段与指定纯 ID 配方完全匹配，仅输出路径
  和声明的标题设置不同；拒绝隐式 alpha、未知配置和输入目录重叠。
- `content_hstu/artifacts.py`：复用已验证 `load_cache`，核对原始输入、文件哈希、
  完整目录、缓存与当前训练数据身份；不能误用 smoke 缓存或其他映射。
- `content_hstu/run.py`：复用现有 Dataset、full-catalog CE、AdamW、评估和 checkpoint
  工具，记录 Git/source/data/config/runtime/content 身份。每轮仅用 val NDCG@10
  选 checkpoint，同分保留更早轮次；同时保存 last.pt 便于精确续跑。

新实现独立于 `src/`、`controls/` 和 `title_features/`，避免改变既有模型的源码
身份校验。每个 run 保存配置、projection.npz、best.pt、last.pt、results.json。
checkpoint 包括模型/优化器/RNG state_dict 和冻结语义表；模型恢复后检查隐藏输出
和候选分数精确相等，完成后验证固定表、输入与源文件未改变。

## 已完成的服务器验证（保留复现命令）

使用已验证的 `agenticrec-hstu-reference` 环境，不重新安装依赖。Mac 已通过
9 个纯 Python/小型 NumPy 测试及语法检查；下面共 17 个测试，新增 8 个依赖
PyTorch 的模型/训练检查也已由用户确认全部通过。测试使用很小的合成数据与 CPU。
下列测试及 smoke 命令无需为本次推进再次执行。

```bash
cd "$HOME/AgenticRec"
git pull --ff-only
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"

python -m unittest tests.test_content_config tests.test_content_projection tests.test_content_artifacts tests.test_content_model tests.test_content_runner -v
```

期待 `Ran 17 tests` / `OK`。再依次运行两个三轮短训练：

```bash
python -m content_hstu.run train --config configs/title-init-smoke-server.json
python -m content_hstu.run train --config configs/title-fusion-smoke-server.json
```

每个实验应显示参数数目 `1273406`，完成后 `Title HSTU train: PASS`。
PASS 要求有限梯度、物品/HSTU 参数更新、固定训练 probe loss 下降、恢复输出和
分数一致、冻结表和输入不变。NDCG 数值不预设，也不要求 smoke 超过基线。
服务器回传记录（2026-10-23）：

| 模式 | epoch 1→3 loss | epoch 1→3 probe | 三轮 val NDCG@10 | 状态 |
|---|---|---|---|---|
| title-init | 9.650886→6.883350 | 8.118129→6.121463 | 0.010190 / 0.007294 / 0.007940 | PASS |
| title-fusion | 9.891118→7.155380 | 8.621272→6.410565 | 0.001962 / 0.004034 / 0.006858 | PASS |

两个模式均报告 1,273,406 个参数，与 HSTU reference 相同。初始化短训练运行了
两次，显示的逐轮数值一致；仅凭这些打印精度不能声称全部产物逐位一致。
报告路径：

- `runs/title-init-smoke-server/20261023T022111405200Z/results.json`
- `runs/title-init-smoke-server/20261023T022133518327Z/results.json`
- `runs/title-fusion-smoke-server/20261023T022236721652Z/results.json`

短训练验证了流程可运行，不用它决定正式实验胜者。当前尚未收到这两个模式的
完整资源字段，不推断速度或显存；正式训练结果见下方最新记录。

## 短训练通过后的正式实验

短训练关卡现已通过。每个模式各 30 轮、相同 seed/数据/预算，均从头训练，
不传 smoke checkpoint。依次执行，一个结束后再启动另一个：

```bash
cd "$HOME/AgenticRec"
git pull --ff-only
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
python -m content_hstu.run train --config configs/title-init-train-server.json
python -m content_hstu.run train --config configs/title-fusion-train-server.json
```

预期各显示 30 轮日志并以 `Title HSTU train: PASS` 结束。返回两段训练日志和
`Saved:` 路径；下一步读取各自 best epoch 的完整验证分组和资源报告。
主比较是未经频次修正的纯 ID HSTU（val NDCG@10=0.0377242280），同时检查
频次 0、1–5、6–20、21+ 的 Recall/NDCG 以及用户历史长度分组。
若总体下降但低频提升，记录权衡，不凭一个分组宣称总体更优。
此前 alpha=0.5 的分数修正结果单列，不将标题效应和分数修正效应混合。

续跑用原配置加 `--resume <原运行目录>/last.pt`，保留同目录 best.pt，要求相同
Git/runtime/content 身份。可用 `--stop-after-epoch 1` 测试 epoch 边界暂停，配置
仍保留完整训练轮数。单独评估入口为：

```bash
python -m content_hstu.run evaluate --config configs/title-init-train-server.json --checkpoint <运行目录>/best.pt --split val
```

融合实验使用自己的 config/checkpoint。全量配置支持显式 `--split test`，但只在
验证集选择并冻结配方后执行；smoke 禁止 test。当前不需要任何测试集新结果。
单种子差异不等于显著提升，标题效果、低频改善和多种子稳健性均等待实际实验。
若有收益，再评估额外存储/训练/服务成本；MiniOneRec 继续按约定留待后续评估。


## 正式训练已完成：先读取分组，不重跑（2026-10-23）

用户回传两个 30 轮完整日志和 `Title HSTU train: PASS`，参数数目均 1,273,406。
从打印精度看，两个模式的最高验证 NDCG@10 都出现在第 30 轮：

| 模式 | 最佳轮次 | val NDCG@10 | 相对纯 ID |
|---|---:|---:|---:|
| 原始纯 ID HSTU | 28 | 0.03772422803068947 | 基准 |
| 标题初始化 | 30 | 约 0.037931 | 约 +0.55% |
| 固定标题融合 | 30 | 约 0.037252 | 约 −1.25% |

标题初始化 loss 从 8.897017 降到 8.105081；固定融合从 9.014198 降到 8.115843。
这是单种子的日志摘要，完整精度以报告为准。当前总体差距小，不能认定标题有
稳定收益，也不能从总体指标断言长尾没有改善。两者最佳轮次位于预算末尾本身
不足以支持延长训练；先按原定 30 轮预算完成比较。

新增 `scripts/compare_title_validation.py` 只使用标准库读取三个已有 results.json。
它核对 full/PASS/完整历史、配置身份、共享数据和用户/源码身份、相同预算、
两个标题实验的相同 content_id、最早最佳验证轮次和分组人数；输出两张 NDCG@10/
Recall@20 对照表以及全部最佳验证指标、运行环境、耗时、显存、投影摘要。
这里的 pairing PASS 只说明报告字段匹配，不代表重新校验 checkpoint 或当前数据
文件，也不是测试集准入许可。不会导入 PyTorch、加载模型、重做 PCA 或执行测试。

在服务器现有环境执行：

```bash
git pull --ff-only
python scripts/compare_title_validation.py \
  --id-report runs/hstu-microlens/20261022T023420952080Z/results.json \
  --init-report runs/title-init-train-server/20261023T162529774092Z/results.json \
  --fusion-report runs/title-fusion-train-server/20261023T165455784364Z/results.json
```

预期 `Validation report pairing: PASS`，随后输出分组表和精确摘要。把输出返回即可。
若拒绝配对，先检查具体差异，不绕过身份检查。本地 5 个合成报告测试和 CLI 检查
已通过；真实报告仅在服务器，实际配对结果与资源字段仍待回传。已有 17 个模型
测试和所有训练都无需重复，暂不执行 test、不增加训练轮数或融合权重搜索。

比较时先看总体，再看频次 0/1–5/6–20/21+，区分整体增益与组间权衡。后续是否
做多种子验证或冻结测试配方依据这些验证结果决定；不以已看过的测试结果调参。


## 完整验证结论与冻结测试候选（2026-10-23）

用户已返回 `Validation report pairing: PASS` 和三个完整摘要。报告字段确认数据/
用户/共享源码/配方匹配，两个标题实验的 content_id 与 PCA 数组身份也一致。

| 验证结果 | 纯 ID | 标题初始化 | 固定融合 |
|---|---:|---:|---:|
| 最佳 NDCG@10 | 0.03772422803068947 | 0.03793132350166986 | 0.03725230719301553 |
| Recall@20 | 0.10176 | 0.10240 | 0.10182 |
| 0 频次 Top-20 命中 / 844 | 0 | 0 | 0 |
| 1–5 频次 Top-20 命中 / 7264 | 0 | 0 | 0 |
| 6–20 频次 Top-20 命中 / 18905 | 191 | 221 | 206 |
| 21+ 频次 Top-20 命中 / 22987 | 4897 | 4899 | 4885 |
| 全部 Top-20 命中 / 50000 | 5088 | 5120 | 5091 |
| 整次运行秒数 | 1196.934 | 1688.929 | 1253.458 |
| 峰值 allocated GiB | 0.16010 | 0.16926 | 0.16926 |
| 峰值 reserved GiB | 0.22656 | 0.234375 | 0.234375 |

初始化的 NDCG@10 相对 +0.54897%，Recall@20 净增 32 次命中，其中中频 +30、
高频 +2。频次 0 与 1–5 合计 8108 个验证目标依然零命中，核心稀疏反馈问题
没有解决。融合的 NDCG@10 相对 −1.25098%，Recall@20 仅净增 3 次命中（中频
+15、高频 −12）。总体 NDCG 和 Recall 可能方向不同，不能只挑更好看的指标。

历史长度分组也不是全面提升：初始化在 1–5 历史长度组的 NDCG@10 略低
（0.03328613 vs 0.03338192），6–10 的 Recall@20 略低；11–20 和 21+ 的
两项指标更好。21+ 组仅 608 人。这里的历史长度分组与物品频次是不同轴。

三次环境均为 A100-SXM4-80GB MIG 2g.20gb、20937965568 visible bytes，
PyTorch 2.10.0+cu128、CUDA runtime 12.8、NumPy 1.26.4、FP32、TF32 关闭。
初始化/融合 Git 均 921b276f7b502e9f942de2f5c18614da99a1987c，纯 ID 为
190ce06025868bbfecf36847d6ad28591ec1e267，均报告 clean。共享模型源码匹配，
Git 提交不同由新实验入口及文档等变化造成，不能只凭提交号不同判为不可比。
运行时间是整次训练及附带工作，未分解 PCA、训练、评估和 I/O，也不是服务延迟；
不把初始化多用的约 8.2 分钟直接解释为架构开销。

PCA 在 18490 个不同训练历史物品上拟合，384→64，保留 0.5789973137664619
的特征方差，表行范数 0.16。两个模式逐数组哈希一致，表 SHA256 为
`1ad2b747b8da982e26019450ebb5273b501875c71016930ed3be18188cc7e317`。
57.9% 是数值方差保留比例，不是“保留 57.9% 语义”，也不能据此认定 PCA
造成负结果。本实验只约束当前 MiniLM/PCA/初始化或固定融合配方，不否定所有
文本方法的可能价值。

### 本轮冻结决定

按事先约定的验证 NDCG@10，选择 title-init、seed 42、第 30 轮 best.pt，
维持原始 cosine 评分、温度 0.2、现有 PCA 与配置；不加 alpha 修正，不延长
训练，不调融合权重。固定融合留作验证阶段的负结果，不继续本轮 test。
选择这个候选是遵循既定验证规则，不是宣布它足以替换现有主线，也不构成
显著性证据。多种子稳健性仍未知。测试结果仅用于报告这份已冻结配方的表现，
不能用它继续选轮、调权重或决定保留哪份 test 结果。

### 执行已冻结候选的测试

新增 `scripts/evaluate_selected_title.py`，默认读取
`configs/hstu-title-selected-test-server.json` 的三个报告路径、选择模式、轮次和
训练配置。先复用报告比较，再验证当前训练/参考配置、模型源码、处理后数据和
报告记录的 best.pt 文件哈希。它不修改旧模型模块，也不提供评分或轮次覆盖项。
随后调用既有 `content_hstu.run evaluate`：后者重新核对缓存、PCA/content_id、
checkpoint 身份，并读取当前 GPU/runtime 后运行测试集。

```bash
git pull --ff-only
python scripts/evaluate_selected_title.py
```

预期先显示 `Frozen title report/checkpoint preflight: PASS`、title-init/epoch 30，
再输出测试指标和 `Title HSTU evaluate: PASS`。把输出与 Saved 路径返回即可。
不用再训练、再跑 17 个测试或重新编码。仅预检可加 `--check-only`，它不执行模型
且不验证缓存/PCA 的重新推导；完整评估器会完成后面的检查。

本地 4 个新增准入检查、既有 5 个报告比较检查及 3 个配置检查（共 12）已通过，
同时完成 CLI/语法检查。此真实 checkpoint 准入及冻结测试仍待服务器执行。
主对照是已冻结纯 ID test NDCG@10=0.0247288911；已有修正后分数 0.0271070230
应单列，避免把评分修正和标题效应混为一谈。
