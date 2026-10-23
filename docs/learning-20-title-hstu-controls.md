# 20. 标题初始化与固定融合 HSTU

本阶段要回答：标题中已有的内容关系，能否改善纯 ID HSTU 的稀疏反馈表现？
已有验证发现是 HSTU 总体更强，但低频目标落后于 Mean/ItemCF。它支持做标题
对照，不证明缺少标题就是根因。实现完成，服务器模型测试和短训练待验证。

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

## 现在在服务器执行

使用已验证的 `agenticrec-hstu-reference` 环境，不重新安装依赖。Mac 已通过
9 个纯 Python/小型 NumPy 测试及语法检查；下面共 17 个测试，新增 8 个依赖
PyTorch 的模型/训练检查尚待服务器验证。测试使用很小的合成数据与 CPU。

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
把测试结尾、两个训练末尾和各自 `Saved:` 路径返回即可。

## 短训练通过后的正式实验

完整配置已经提供，等短训练通过再运行。每个模式各 30 轮、相同 seed/数据/
预算，通过验证集比较既有纯 ID、标题初始化、固定融合，查看总体和频次分组。
正式命令是把上述配置换成 `title-init-train-server.json` 和
`title-fusion-train-server.json`；不要把 smoke checkpoint 用于正式训练。

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
