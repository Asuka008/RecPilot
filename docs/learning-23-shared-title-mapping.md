# 23｜共享标题映射：先检验迁移机制

## 为什么选这条路径

笔记 22 的完整验证诊断支持标题有信号：384 维标题均值低频命中 225，
置换对照 11；64 维为 202，对照 3。三个现有 HSTU 低频仍零命中，
目标中位排名约一万。继续调固定融合权重缺少依据。

本轮保留 HSTU 和冻结文本编码器，移除独立的可训练 ID 表，改用共享内容映射。
先做两个成对配方：map-frozen 固定映射，只训练 HSTU；map-learned 同时训练映射。
两者从相同的训练物品 PCA 起步，历史输入、候选输出共用映射。
这样不会再让每个物品的自由 ID 参数吸收固定内容项。
这是检验内容迁移的机制实验，不是已确定的生产替代方案；移除 ID 表可能损失总体精度。

暂不同时加入 ID 残差、门控、ID dropout、ItemCF、热门度修正、采样损失或新编码器。
固定对照仍训练 HSTU，不能与未训练的标题均值评分器混为一谈。

## 代码与形状

独立包 `mapped_hstu/` 保持历史 src/content_hstu/title_features 源码不变。
`config.py` 复用已有配置契约，要求相同基线预算。`artifacts.py` 核验原缓存并
复用训练历史中不同物品拟合的 PCA 均值和分量；不重新编码文本。
`run.py` 从已有隔离训练器派生，保留恢复、版本/数据身份及资源记录。
它仅允许 train 和 validation evaluate，没有测试集入口。

`model.py` 的共享查表按以下公式计算：

```
v_i = 0.02 * sqrt(64) * normalize(W @ (title_i - mean))
```

缓存 `[19221,384]` 固定；mean `[384]` 固定；W `[64,384]` 从 PCA 分量转置初始化。
padding 最终强制为零。没有 bias，也没有每个物品独立的可训练参数。
历史查表输出 `[B,32,64]`；HSTU 输出对应位置的归一化查询；候选表示为
`[19220,64]`，两端采用同一个 W。训练继续使用原全目录交叉熵与温度 0.2。
零正交互物品仍参与分母；共享参数允许从其他物品获得更新，但不保证冷启动成功。

该计算使用 FP32 中心化与投影，和历史 float64 PCA 后保存的表可能存在数值差异。
新成对配方初始计算一致；不声称逐位复现旧 title-init 的初值。
构造函数恢复初始化随机状态，避免额外线性层初始化改变两组的 dropout 随机流。

正式配置下，原 HSTU 可训练参数 1,273,406，其中 ID 表 1,230,144。
固定映射组可训练参数为 43,262，学习映射组为 67,838，差 24,576 个共享参数。
冻结缓存占内存但不属于可训练参数；运行时会打印实际参数数目。
这个差别正是所检验的干预，不宣称与纯 ID 是等参数架构比较。

## 验证及判定顺序

1. 服务器 7 个聚焦检查：配置、成对初始输出、无自由 ID 参数、共享输入/输出、
   padding/因果性、梯度与冻结缓存、端到端训练/恢复/验证及禁止测试。
2. 两个配方各做 3 轮、1,024 用户 smoke。检查参数更新、probe 下降、检查点输出
   恢复完全一致、输入哈希未变；固定组还必须证明 W 没改变。smoke 指标不作效果比较。
3. smoke 通过后才运行两个已固定的 30 轮全量配方；种子 42，沿用相同数据和预算。
   各自按验证总体 NDCG@10 选最早最佳轮，不按低频测试切片挑轮，不追加轮数。
4. 首要机制对照是 learned 对 frozen 的验证 NDCG@10，报告低频 Recall@20、
   零频 Recall@20、人数与命中数。若总体或低频退化，明确报告取舍，不宣称全面成功。
   只有超过原始 ID 总体 NDCG@10=.0377242280 且改善其低频零命中，才具备
   进一步验证实际改进的理由；单种子仍不足以宣称稳健提升。
5. 若未证明学习映射有益，先记录负结果，不能自动叠加 dropout 或调大量权重。
   若有益，再预登记多种子和后续独立消融。本轮不触碰测试集。

## 现在在服务器执行

本地仅完成三个配置测试、语法和 CLI 检查；四个 PyTorch 检查与 smoke 等待服务器验证。
没有在 Mac 运行模型，没有新的推荐效果结果。

```bash
cd ~/AgenticRec
git pull --ff-only
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
python -m unittest tests.test_mapped_config tests.test_mapped_model -v && \
python -m mapped_hstu.run train --config configs/map-frozen-smoke-server.json && \
python -m mapped_hstu.run train --config configs/map-learned-smoke-server.json
```

预期 `Ran 7 tests ... OK`，两个 smoke 最后分别显示 `Mapped HSTU train: PASS`。
请返回测试结尾、两次训练日志和保存路径。运行报告自动记录当前实际设备/版本、
Git、数据、缓存/PCA、源码、参数、耗时与显存；环境名不能代表实际 GPU。

以下全量命令已准备好，但应在检查 smoke 后再执行：

```bash
python -m mapped_hstu.run train --config configs/map-frozen-train-server.json
python -m mapped_hstu.run train --config configs/map-learned-train-server.json
```

需要中断续训时使用同环境/源码下的 `--resume <原运行目录>/last.pt`；保留同目录 best.pt。
所有运行产物保持 Git 忽略。不重复之前完成的诊断、标题编码或测试评估。


## Frozen projection registration fix (2026-10-24)

The user returned six passing tests and one failing runner integration test.
The fixed projection was a requires_grad=False Parameter, while the shared
training routine deliberately requires a finite gradient for every named
parameter. It is now a registered buffer; the learned projection remains a
Parameter. Both use F.linear and save the same weight key in state_dict.
The original training routine and gradient guard remain unchanged. Tests now
explicitly check buffer/parameter registration and checkpoint inclusion.
Three local configuration tests and syntax checks pass; corrected neural tests
and real-data smokes await server verification. The shell && chain stopped at
tests, so neither real-data smoke ran. Pull and repeat the seven focused tests
followed by the two smoke commands above. No prior training rerun is required.


## Paired mapping smoke server PASS (2026-10-24)

The user returned all seven focused tests PASS (13.702s) and both three-epoch
smokes PASS at commit53a191b. Runtime: Python3.11.16, torch2.10.0+cu128,
CUDA12.8, A100-SXM4-80GB MIG2g.20gb,20937965568 visible bytes.
Frozen:43262 trainable parameters; loss9.738615->9.434431,
probe9.461222->9.072540, final validation NDCG@10=.010448.
Report:runs/map-frozen-smoke-server/20261024T165622195010Z/results.json.
Learned:67838 trainable parameters; loss9.679343->9.212684,
probe9.271473->8.798348, final validation NDCG@10=.009406
(best printed .009813 at epoch1).
Report:runs/map-learned-smoke-server/20261024T165639012746Z/results.json.
These are1024-user smokes, not evidence of a full-data improvement. Lower learned
training loss does not establish better validation ranking. Actual final resource
fields were not supplied. Both smoke gates passed; next execute the predeclared
paired30-epoch full-data validation-only recipes without changing hyperparameters.
Do not repeat smoke, cache preparation or old test evaluations. Compare each
model's earliest validation-selected best epoch, not simply its last epoch.


## Full paired mapping training PASS (2026-10-24)

User returned both30-epoch full-data training runs PASS on A100 MIG2g.20gb,
Python3.11.16/torch2.10.0+cu128. Frozen43262 parameters: best printed validation
NDCG@10=.012059 at epoch28; final loss9.286922. Report:
runs/map-frozen-train-server/20261024T165826137072Z/results.json.
Learned67838 parameters: best printed validation NDCG@10=.014608 at epoch28;
final loss8.956967. Report:
runs/map-learned-train-server/20261024T171919631080Z/results.json.
From rounded logs learned is about21.14% higher than frozen, but about61.28%
lower than raw ID (.0377242280). It is only about3.92% above direct384 title
mean (.0140572575), and that comparison changes encoder/dimension/training.
No robust architecture or end-to-end improvement is established. Exact best
selection, frequency/history slices, paired identities and resource fields have
not yet been returned. Next read these saved reports only; do not add epochs,
change tuning, or evaluate test. Tail and zero-frequency improvement is unknown
until selected validation slices are inspected. Retain ID HSTU as main model.
