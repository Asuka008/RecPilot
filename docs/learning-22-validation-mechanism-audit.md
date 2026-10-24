# 22｜验证集机制诊断

本次落实笔记 21 的研究结论：先检查标题信号、目标排名和协同互补，再选择训练改动。
实现已完成；12 个标准库/微型 NumPy 检查和语法检查本地通过。
2 个服务器 CPU 检查及真实数据 CUDA 推理仍待验证。没有新的效果结论。

## 输入和执行路径

入口是 `scripts/audit_recommendation_mechanisms.py`，固定配置为
`configs/recommendation-mechanism-audit-server.json`。
`diagnostics/tail_admission.py` 核对三个训练报告、原先验证选中的 best.pt、
配置、源文件清单、数据、缓存及 ItemCF 验证报告的身份。
原有 src、content_hstu、title_features 和训练配置没有修改。

依次诊断八个评分器：ID HSTU、title-init、title-fusion、ItemCF，以及
原始 384 维/PCA 64 维标题均值，各自附一个置换对照。
神经模型输入为 `[B,32]` 左补零历史，输出查询 `[B,64]`；标题均值查询为
`[B,384]` 或 `[B,64]`。精确评分批次为 `[B,19220]`，不会保存全量评分矩阵。
标题均值采用最近 32 个事件，历史与候选向量归一化。ItemCF 保留原基线的完整
历史输入；所以互补诊断不是严格控制输入长度的架构实验。

所有方法排除 padding 和完整可见历史，保留零频目标，同分时小 ID 优先。
共享加载器读取原始 train/val/test 文件并检查格式；诊断记录随后仅保留
user_id/train/val，不使用测试目标计算指标或选择参数。

置换种子固定为 20261023，只在训练频次组内置换标题行，两个维度共用一个置换，
历史和候选两端应用同一对应关系，padding 不动。它保留组内频次信息和向量分布，
只有一次置换，不是显著性检验。零均值查询采用全零评分及稳定 ID 回退，并记录次数。

## 诊断输出与解释

`diagnostics/tail_statistics.py` 计算目标前方所有更高分或同分小 ID 候选，得到精确
排名；按 all/0/1–5/6–20/21+ 训练频次分组统计排名分位数和 R@20/100/500/1000。
分差定义为目标分数减第 20 名分数。分差为零也可能因同分规则未进入 Top-20；
不同评分器的分差尺度不能直接比较。候选少于 20 时记录实际边界 K。

HSTU 与 ItemCF 按同一用户统计共同、独有及均未命中数。命中并集只是理想上限，
两路列表可能包含 40 个物品，不能当成实际 Top-20 融合成绩。
标题均值弱不证明监督学习后的内容表示无效，强也不保证接入 HSTU 后有效。

必须先复现四个已有评分器的完整验证切片及指标，绝对误差容忍值 `1e-10`。
总体 NDCG@10 应为 ID .03772422803068947、init .03793132350166986、
fusion .03725230719301553、ItemCF .024494992910568。
任一锚点不符即保存 FAIL，不自动放宽容忍值或开始训练。

运行目录保存 `results.json`、`per_user_diagnostics.npz` 和 `title_permutation.npy`。
逐用户文件包含对齐的 user_id、目标训练频次和八个评分器的排名/分差/边界 K，
没有测试目标或全目录分数。报告记录输入哈希、Git、实际 GPU/软件栈、资源及耗时，
结束时再次核对输入。PyTorch 显存统计不包含所有外部分配，诊断耗时不是服务延迟。

## 服务器命令

使用现有环境，无需安装依赖或重复已完成的训练、标题编码、测试集评估。

```bash
cd ~/AgenticRec
git pull --ff-only
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
python -m unittest tests.test_mechanism_admission tests.test_mechanism_statistics tests.test_mechanism_server -v && \
python scripts/audit_recommendation_mechanisms.py
```

预期 `Ran 17 tests ... OK`。服务器专属检查包括标题均值手算，以及四用户八物品
合成状态的完整推理和错误锚点拒绝；它们不训练真实模型，也不证明推荐效果。
正式诊断先显示 `Frozen diagnostic file admission: PASS` 和实际设备，随后各变体
打印进度，四个已有评分器分别打印 `frozen validation anchor PASS`。
成功结束显示 `Recommendation mechanism audit: PASS`。

可选的纯文件预检：

```bash
python scripts/audit_recommendation_mechanisms.py --check-only
```

返回末尾八个变体摘要、Same-K complementarity、保存路径及 PASS/FAIL。
出现错误先返回报错，不修改既有模型或校验规则。根据报告选择一个有证据的改动；
没有模型选择或自动调参步骤，也没有预设标题一定能解决长尾。


## Legacy ItemCF report fix (2026-10-24)

The user reported all original 14 tests and three HSTU validation anchors PASS
on A100 MIG. The audit failed on a missing history-length axis in the historical
ItemCF report, not an observed metric mismatch. Only legacy ItemCF may omit this
axis: its frequency groups remain strictly checked, and newly computed history
metrics are labeled computed_without_historical_anchor. Modern reports still
check both axes. Fifteen lightweight local tests pass; the updated server checks
and real audit remain pending (17 focused tests total). Repeat validation
inference only; no retraining, title encoding or test evaluation is needed.
The failed run did not save per-user ranks, so inference must restart.


## 2026-10-24 完整服务器诊断 PASS

依据用户返回的控制台输出，17 个测试通过（7.413 秒），三个 HSTU 的两个
验证轴及旧 ItemCF 的频次轴均复现。完整诊断 PASS，保存于
`runs/recommendation-mechanism-audit/20261024T162518831454Z/results.json`。
实际为 A100 MIG 2g.20gb、PyTorch 2.10.0+cu128、FP32，50,000 个验证用户。
本地未独立读取服务器完整报告，最终耗时/峰值显存字段尚未返回。

| 方法 | 总体 NDCG@10 | 低频 1–5 Top-20 命中/7264 | 零频 Top-20 命中/844 |
|---|---:|---:|---:|
| id | 0.03772423 | 0 | 0 |
| title-init | 0.03793132 | 0 | 0 |
| title-fusion | 0.03725231 | 0 | 0 |
| itemcf | 0.02449499 | 348 | 0 |
| title-mean-384 | 0.01405726 | 225 | 26 |
| title-mean-384-permuted | 0.00032139 | 11 | 0 |
| title-mean-64 | 0.01161999 | 202 | 23 |
| title-mean-64-permuted | 0.00031425 | 3 | 2 |

主要证据与解释：

- 原始 HSTU 低频目标排名中位数 10428.5，Top-100/1000 仅 3/108 次命中；
  init/fusion 中位数为 10642.5/10251。失败不是普遍卡在第 20 名边缘，
  但这仍不能独立证明损失函数或某一层是根因。
- 384 维标题低频命中 225，对照置换 11；64 维为 202，对照 3。
  支持内容对应关系包含可用信号，不构成单次置换的显著性检验。
- PCA 后低频 Top-20 命中从 225 降至 202，但 Top-1000 从 1497 升至 1712，
  排名中位数也改善。不能简单认定 PCA 全面损坏语义，更不能解释 HSTU 的零命中。
- 原始标题直接推荐零频命中 26，64 维命中 23；两个置换对照为 0/2。
  这是已知目录、逐用户划分下的零训练交互结果，不是严格全局时间的新发布视频证据。
- HSTU/ItemCF 共同命中 1943，独有 3145/1674；ItemCF 独有命中中
  低频 348、中频 1011、高频 315。并集 6762（R@20 上限 .13524）
  不是实际 Top-20 融合效果，零频仍无协同命中。

下一项候选优先检验共享的内容到偏好映射，保持 HSTU 和冻结标题编码器。
先明确历史/候选两端表示、参数量对照与验证指标，再做小型实现；ID 信息丢弃
作为独立消融，不和映射同时混改。低频目标训练频次与全部用户/候选身份必须保持一致。
当前内容映射尚未实现，也没有证据保证它能同时改善总体与低频指标。
ItemCF 融合保留为已有协同信号的备选，不与新内容模型同时开展。
不重复此次诊断或测试集评估，不将标题均值总体较差解释为可直接替代 HSTU。
