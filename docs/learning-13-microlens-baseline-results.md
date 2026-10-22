# 13：首个纯 ID HSTU 结果与验证诊断

日期：2026-10-21。**首个全量 HSTU 已完成训练和冻结测试；下一步是读取验证
切片并确定一个可检验的改进问题。当前没有新增模型或文本融合。**

## 已验证的结果

依据用户回传的服务器输出，30 轮训练 PASS，验证选出的 best.pt 测试也 PASS。
训练目录为 `runs/hstu-microlens/20261022T023420952080Z/`，测试目录为
`runs/hstu-microlens/20261022T030714112850Z/`。配置、选择规则和训练过程见
[笔记 12](learning-12-microlens-full-training.md)。日志中验证最佳为第 28 轮
NDCG@10=0.037724；未舍入选择字段、全部验证指标和正式训练资源摘要尚未回传。

在同一逐用户留一、全目录、完整可见历史过滤和稳定同分排序协议下：

| 测试指标 | 固定 ItemCF | 纯 ID HSTU | 相对变化 |
|---|---:|---:|---:|
| Recall@10 | 0.03534 | 0.04882 | +38.14% |
| NDCG@10 | 0.0174729380 | 0.0247288911 | +41.53% |
| Recall@20 | 0.05760 | 0.07238 | +25.66% |
| NDCG@20 | 0.0230875271 | 0.0306860794 | +32.91% |

相对变化定义为 `(HSTU / ItemCF - 1) * 100%`。Recall@10 绝对增加
0.01348，即 1.348 个百分点；Recall@20 绝对增加 1.478 个百分点。
每个 split 有 50,000 用户，Top-10 命中从 1,767 到 2,441，Top-20 从
2,880 到 3,619。NDCG 同时考虑命中名次，不能直接乘用户数解释成命中数。

这是一组 seed=42 的离线比较，不是跨种子均值、统计显著性、线上提升或严格
隔离架构因素的实验。ItemCF 没有训练神经模型的目标和调参过程，不能把全部
差异归因为 HSTU 的顺序建模。SASRec 没有被比较。

## 总体提升掩盖了什么

按目标物品在完整训练历史中的事件频次分组，测试结果：

| 频次 | 用户数 | Recall@10 | Recall@20 | Top-20 命中数 |
|---|---:|---:|---:|---:|
| 0 | 2,226 | 0 | 0 | 0 |
| 1–5 | 11,557 | 0 | 0 | 0 |
| 6–20 | 19,270 | 0.00363259 | 0.00793980 | 153 |
| 21+ | 16,947 | 0.13990677 | 0.20451997 | 3,466 |

频次 0 与 1–5 合计 13,783 用户，占 27.566%，Top-20 都未命中；
频次 21+ 占约三分之一用户，贡献约 95.8% 的 Top-20 命中。
这是长尾绝对表现弱的证据；是否比 ItemCF 的对应切片更差，仍需对齐切片比较。
它不证明文字必然能解决问题，也不等于模型对所有用户都只输出相同热门物品。
候选覆盖、榜单重合程度和分数分布尚未测量。

按完整可见历史长度的测试结果：

| 长度 | 用户数 | NDCG@10 | Recall@20 |
|---|---:|---:|---:|
| 1–5 | 31,090 | 0.02181388 | 0.06420071 |
| 6–10 | 14,960 | 0.02799804 | 0.07987968 |
| 11–20 | 3,264 | 0.03361627 | 0.10355392 |
| 21+ | 686 | 0.04326085 | 0.13119534 |

不能直接把更长历史组的更高分解释为“增加历史就会提升”；不同组的目标频次、
用户活跃度等也可能不同。既有协议只保证每用户自身的时间顺序，不是全局时间回放。
后续选择改进依据验证数据，保持这次测试结果和首个模型冻结。

## 本地审计：历史出现不等于正向监督

本次在 Mac 做了不含 PyTorch 的纯数据审计，没有运行模型。命令为：

```bash
python3 -m unittest tests.test_supervision_audit -v
python3 scripts/audit_hstu_supervision.py --config configs/hstu-microlens-train-server.json
```

两个微型测试通过；完整审计已本地运行，服务器复跑尚无回传。
脚本使用标准库，只读完整训练历史和用于分组的验证目标，不读取测试标签。
数据与配置不变时输出应一致。三个数据哈希随输出保存，其本地值与此前审计一致：

```text
77d9ad58f38d8c0072067686a63b82753ffc89609fadf6ae436ca93ca2e25827  sequences.jsonl
969f875d0d0285c9f4c2ddaffdfff27d591b06bdca4b46a7ecc995ac3f62cb94  item_mapping.json
988a3fdf64b7e26bc72cc31cf430d3f14da6bbc4bad9f1c4b84d90b7c17adbea  metadata.json
```

`audit_supervision(records, num_items, L)` 输入是处理后的用户记录列表和目录/长度，
输出为一个计数字典。它复现 `build_training_example` 的核心切片：
`retained = train[-(L+1):]`，正标签为 `retained[1:]`，不包括 padding。

L=32 时共有 259,708 个训练历史事件，但有效正标签位置为 206,613。
每个用户至少少一个首物品标签，合计 50,000；截断又减少 3,095 个标签。
历史中出现的物品有 18,490 个，真正当过正标签的有 18,008 个：482 个历史
出现过的物品从未成为保留的正标签。整个目录有 1,212 个物品没有这种正标签。
它们仍可能作为输入或全目录 CE 的其他候选类接收梯度，不能说完全没训练过。

| 验证目标历史频次 | 验证用户数 | 目标没有保留正标签的用户数 |
|---|---:|---:|
| 0 | 844 | 844 |
| 1–5 | 7,264 | 74 |
| 6–20 | 18,905 | 5 |
| 21+ | 22,987 | 0 |

因此定义差异真实存在，但验证低频组大多数目标有过正向监督，不能仅靠
“从未当过正标签”解释整个长尾问题。有正标签也不代表反馈充分或一定能学好。
目前不改数据划分、频次分桶或已报告的评估分母。

## 下一步：读取验证切片，与 ItemCF 对齐

无需重新训练或测试。在服务器读取已有两个验证报告，先核对三份数据哈希，
再打印相同分组的用户数、Recall@20 和 NDCG@10：

```bash
python - <<'PY'
import json
from pathlib import Path
h = json.loads(Path("runs/hstu-microlens/20261022T023420952080Z/results.json").read_text())
b = json.loads(Path("runs/baselines/20261018T201816099815Z/results.json").read_text())
assert h["status"] == "PASS" and b["split"] == "val"
assert h["provenance"]["data_sha256"] == b["data_sha256"], "Data hashes differ"
best = next(row for row in h["history"] if row["epoch"] == h["selection"]["best_epoch"])
hg = best["validation"]["groups_by_target_train_frequency"]
bg = b["models"]["itemcf"]["groups_by_target_train_frequency"]
print("selection:", h["selection"])
print("group users ItemCF_R20 HSTU_R20 ItemCF_N10 HSTU_N10")
for name in ("all", "0", "1-5", "6-20", "21+"):
    x, y = bg[name], hg[name]
    assert x["users"] == y["users"]
    print(name, x["users"], x["metrics"]["Recall@20"], y["metrics"]["Recall@20"], x["metrics"]["NDCG@10"], y["metrics"]["NDCG@10"])
print("checks:", h.get("checks"), "data unchanged:", h.get("checksums_unchanged"))
print("GPU:", h["runtime"].get("gpu"), "elapsed seconds:", h.get("elapsed_seconds"))
print("memory GiB:", {k: round(v/1024**3, 3) for k, v in h.get("memory_bytes", {}).items()})
PY
```

这些输出用于判断：高频收益与长尾损失是否在验证集重现、优化目标与表示是否
值得先检查，以及是否增加相同 embedding/打分/目标预算下的 masked mean pooling
神经对照。该对照仍未实现；文本初始化、重加权或更复杂融合也没有选定。
先检查证据，再选一个假设，用独立配置记录后续实验，不因测试切片直接改参数。
