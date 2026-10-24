# 25：冻结 HSTU + ItemCF 配方，检查测试集表现

## 目的与边界

验证集预先声明的四个权重中，alpha=0.25 被选中：NDCG@10 为
0.044361011644391685，Recall@20 为 0.11482，低频目标命中由 0 增加到 72。
现在检查这个固定方案能否在测试集保持收益，不重新训练，不扩大权重搜索。
这仍是单种子结果；本项目测试集已被多个方案使用，不能称为完全未接触的独立确认。

## 代码与数据流

- `scripts/hybrid_test_admission.py` 重新核算验证选择规则，并检查验证报告绑定的
  数据、源码、配置、检查点哈希以及原始 HSTU 验证指标。失败时不启动模型评估。
- `configs/hybrid-selected-test-server.json` 固定验证报告路径和 alpha=0.25。
- `scripts/evaluate_selected_hybrid.py` 加载原始 ID HSTU 的 best.pt，使用训练历史
  构建 ItemCF。测试用户输入为训练历史加验证物品，测试目标不进入输入。
  HSTU 输入为 `[B,L]`，候选得分为 `[B,19220]`；逐用户补 padding 位后组合
  两路 `[19221]` 得分。过滤完整可见历史，采用既有确定性排序规则。
- 评分直接复用已验证的 `evaluate_hybrid_validation.combine`：
  `0.75 * (HSTU_cosine + 1) / 2 + 0.25 * ItemCF / max(eligible_ItemCF)`。
  CF 没有可用信号时回退原始 HSTU。不叠加标题或此前的频次修正。

ItemCF 的训练统计仍只来自训练历史；查询时可使用该用户已经看过的验证物品，
这与测试 Dataset 的可见历史一致。频次分组也只依赖训练历史。
准入复用现有审计，因此仍需要服务器上原有标题报告和缓存文件，但不执行标题模型。

## 服务器验证

```bash
git pull --ff-only
python -m unittest tests.test_hybrid_test_admission tests.test_hybrid_validation -v && \
python scripts/evaluate_selected_hybrid.py
```

预期 7 个测试 OK，随后 `Frozen hybrid test admission: PASS`，最后
`Frozen hybrid test: PASS`。只做文件准入可添加 `--check-only`。
新报告保存到 `runs/hybrid-selected-test/<timestamp>/results.json`，包括两种分组、
冻结配方、来源哈希、Git、运行栈、耗时和显存。PASS 表示执行与身份校验通过，
不意味着指标必然提升。本地仅验证轻量测试与语法，真实模型推理等待服务器验证。

回传指标后同时对照原始 HSTU（测试 NDCG@10=0.0247288911、Recall@20=0.07238）
和固定频次修正 HSTU（0.0271070230、0.07572），并报告低频命中与零频组。
如果不提升，就如实记录，不根据这次测试修改权重。
