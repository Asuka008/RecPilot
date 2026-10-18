# 教学笔记 04：统一评估与简单基线

2026-10-18：代码已实现，10 个小型合成测试在本地通过。完整 MicroLens
运行与指标仍等待服务器验证；本地没有运行完整评估或模型训练。

## 从哪里进入代码

- `src/run_baselines.py`：读取配置，校验数据，拟合基线，评估并保存结果。
- `src/baselines.py`：Popularity 与 ItemCF，只读取记录中的 `train`。
- `src/evaluation.py`：候选过滤、稳定 Top-K、单目标指标与分组汇总。
- `tests/test_baselines.py`：手算答案、泄漏边界及两用户端到端运行。

输入沿用现有 `sequences.jsonl` 的 `{user_id, train, val, test}`。
读取时检查 ID 范围、用户唯一性、最低历史长度、映射可逆性和 metadata 数量。
这些检查不能从不含时间戳的文件重新证明时间顺序；时间顺序由前一预处理阶段建立。

## 如何打分

Popularity 统计训练历史中的交互次数，重复观看会重复计数。输出一个形状为
`[num_items + 1]` 的 float64 分数数组，数组下标就是映射视频 ID，0 留给 padding。

ItemCF 先把每个用户的训练历史去重，使用二元用户-视频关系。设 `U(i)` 为训练中
看过视频 i 的用户集合：

```text
similarity(i,j) = |U(i) ∩ U(j)| / sqrt(|U(i)| × |U(j)|)
```

例如两名用户分别看过 `[1,2]`、`[1,3]`，则相似度 `s(1,2)=1/sqrt(2)`。
这里利用共同观看关系，不读取标题，也不使用行为顺序；它因此适合作为 HSTU 的
低成本对照。Popularity 使用事件次数，ItemCF 使用用户集合，两者的计数定义不同。

实现使用倒排表，每次只为一个源视频累计共现；按相似度保留最多 100 个邻居，
同分选 ID 较小的邻居。不建立完整的物品平方矩阵。相似度原本对称，但各源视频
独立截取邻居后，保留的邻居关系不一定对称。拟合成本主要随各用户历史长度平方之和
增长，邻居存储最多约为 `num_items × neighbors`，仍需服务器验证实际时间与内存。

推荐时，对用户可见历史中的不同视频，累加其邻居相似度；历史重复不增加权重。
无邻居或没有训练记录的视频贡献为零。未获得分数的候选同为零分，统一按 ID 排序
作为确定性回退，没有额外使用热度补分。截断邻居是打分方法的定义，候选评估仍然
覆盖完整目录，不是采样候选评估。

## 评估协议

首版协议命名为 `per-user-leave-one-out-full-catalog-v1`：

| 阶段 | 用户输入历史 | 真实目标 | 基线拟合数据 |
|---|---|---|---|
| 验证 | train | val | 所有用户的 train |
| 测试 | train + val | test | 所有用户的 train |

候选采用完整映射目录，显式假设该目录预先已知，排除 padding 和该用户的完整
可见历史，不对历史截断。相同分数按映射 ID 升序排列。Top-K 使用 partition 找到
分数边界，再显式处理边界同分项，避免不稳定排序。候选不足 K 时全部返回。

若目标已在过滤历史中，程序报错，要求另建允许复看的协议；不会删除该用户或
为了命中目标而临时解除过滤。当前数据审计显示没有这种重复目标。
该协议不保证不同用户之间满足统一全局时间截止点。

每名用户只有一个相关目标，目标排名为 r 时：

```text
Recall@K = 1 if r <= K else 0
NDCG@K   = 1/log2(r+1) if r <= K else 0
```

目标排第二时 `NDCG=1/log2(3)≈0.63093`。所有指标按用户平均。
目标训练频次使用训练交互次数，分为 0、1–5、6–20、21+，同时报告用户数。
零频目标保留在总体分母和独立切片中；空切片指标为 `null`，不是零分。
当前数据预期验证零频目标 844 条，测试零频目标 2,226 条。

## 本地验证与服务器命令

本地仅运行小型合成检查，共 10 个测试：手算命中与折损、padding/历史过滤、
边界同分、与穷举排序一致、拒绝非法分数、训练统计不依赖 held-out 字段、
ItemCF 余弦与去重、零分回退、分组分母、验证/测试历史和微型命令行运行。
其中两个基线的端到端检查只使用两名虚构用户。

```bash
python -m unittest tests.test_baselines -v
```

代码提交并推送后，在服务器运行：

```bash
cd "$HOME/AgenticRec"
git pull --ff-only
source /opt/conda/etc/profile.d/conda.sh
conda activate "$HOME/conda-envs/agenticrec-h200"
python -m unittest tests.test_dataset tests.test_baselines -v
python -m src.run_baselines --config configs/baselines-server.json --split val
```

预期 18 个测试通过，随后依次显示两个基线的总体指标及 `Saved:` 结果路径。
本阶段是 CPU 统计基线，不需要 CUDA 或 GPU；完整数据运行仍放在服务器。
不要在 Mac 上运行上述完整评估命令。

先审查验证结果与耗时，若调 ItemCF 邻居数量，只依据验证结果选择。
设置固定后再运行测试，不按测试分数调参：

```bash
python -m src.run_baselines --config configs/baselines-server.json --split test
```

## 输出与复现

每次运行在 `runs/baselines/<UTC时间>/results.json` 保存配置快照、Git 提交与
工作区是否有改动、三个输入文件的 SHA-256、Python 与安装包版本、随机种子、
评估协议、CPU/分数精度、两个基线各自的拟合与评估耗时、总体及频次分组指标。
算法没有随机采样，seed 仅作为实验记录。时间字段随运行变化，指标应确定复现。
每完成一个基线就写入文件，后续基线失败时仍保留已完成结果。

检查输出中的 `git.dirty`，正式比较应使用干净且已提交的代码。结果目录被 Git
忽略，需要单独保存。metadata 的原始文件 checksum 只标识预处理来源；本次还对
三个实际输入文件分别计算 hash，防止同一份 metadata 搭配不同序列。

当前没有 HSTU 模型、GPU/混合精度验证或线上收益结论。服务器测试通过和完整
指标返回后，才可把本阶段记为服务器已验证。
