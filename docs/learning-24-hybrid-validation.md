# 24｜有限评分组合验证

此前 ItemCF 有1674次 HSTU 未命中的验证目标，其中低频348次。本轮只检验评分互补。
入口 scripts/evaluate_hybrid_validation.py 使用现有准入和原始 ID best.pt，
按原配方用训练历史构建 ItemCF。一次 HSTU 推理评估 alpha=0/.1/.25/.5。
输入为 [B,32]，全目录评分为 [B,19220]。

固定公式：(1-alpha)*(HSTU cosine+1)/2 + alpha*CF/max(CF)。
CF 最大值来自排除完整历史后的候选。没有 CF 信号时回退原始 HSTU；
alpha=0 保留原分数。继续共享过滤和同分规则，不按未知真实目标频次选择模型。
归一化只是明确的工程假设，不能保证不同来源分数已经概率校准。

先核对原始 HSTU 和 ItemCF 的已有验证指标；旧 ItemCF 只核对已有频次轴。
固定候选条件为总体 NDCG@10 不低于原始 HSTU，且低频有命中，然后选最高总体
NDCG，精确平分取较小 alpha。这只是筛选：一条命中不证明有效改进，需要审查
各组数量及代价。还须与频次修正 HSTU 的验证 NDCG@10=.0403449990 比较。
无符合者即记录负结果，不自动扩展网格。并集上限不当作实际组合指标。

本地三个微型 NumPy 测试及语法检查通过；真实服务器执行待验证。没有新模型
训练或测试集评估。验证集经过多次实验使用，不能声称独立确认或统计显著性。
报告记录 Git、来源哈希、运行设备、版本、配置、显存和耗时，并检查输入未变。

```bash
cd ~/AgenticRec
git pull --ff-only
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
python -m unittest tests.test_hybrid_validation -v && \
python scripts/evaluate_hybrid_validation.py
```

预期三个测试 OK，结尾 Hybrid validation: PASS。PASS 表示执行及锚点正确，
不代表存在有益组合。返回四组 alpha 摘要、selection 和保存路径。
产物位于 runs/hybrid-validation。为复用准入，仍校验已有标题报告/缓存身份，
实际只执行 ID HSTU 推理。原模型、训练配置和缓存不变。
