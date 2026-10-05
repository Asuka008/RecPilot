# 教学文档 01：SASRec 的 Dataset 与 DataLoader

## 这部分解决什么问题

预处理后，每个用户的数据是一条按时间排序的 item 序列：

```text
train = [12, 31, 9, 71]
```

SASRec 的训练目标是：看到当前的历史后，预测下一个 item。因此同一条序列要错开一位，变成输入和目标：

```text
input  = [12, 31, 9]
target = [31, 9, 71]
```

三个有效位置分别表示：

- 看到 `12` 时预测 `31`。
- 看到 `12, 31` 时预测 `9`。
- 看到 `12, 31, 9` 时预测 `71`。

## 核心代码

`build_training_example()` 先保留最近的 `max_seq_len + 1` 个 item，再错位生成最多 `max_seq_len` 个预测位置：

```python
retained_sequence = sequence[-(max_seq_len + 1) :]
input_ids = retained_sequence[:-1]
target_ids = retained_sequence[1:]
```

只保留最近的交互，是因为用户最近的兴趣通常对下一次行为更有价值，同时 Transformer 需要固定的最大序列长度。

## 左侧补零和 padding mask

一个 batch 中的序列长度必须相同。项目保留索引 `0` 作为 padding，并在左侧补零：

```text
max_seq_len = 5
input_ids   = [0, 0, 12, 31, 9]
target_ids  = [0, 0, 31, 9, 71]
```

左侧补零让最新的行为始终靠右对齐。`padding_mask` 告诉模型哪些位置是真实 item：

```text
padding_mask = [False, False, True, True, True]
```

padding 位置不应参与 attention 或 loss 计算。后面实现 SASRec 时，这个 mask 会和 causal mask 同时使用：

- padding mask 屏蔽补零位置。
- causal mask 防止当前位置看到未来 item。

## 训练、验证和测试的区别

假设预处理记录为：

```text
train = [12, 31, 9]
val   = 71
test  = 82
```

三个阶段使用的数据不同：

| 阶段 | 模型可见历史 | 需要预测 |
|---|---|---|
| 训练 | `train` 内部的错位序列 | `train` 中的下一个 item |
| 验证 | `train` | `val` |
| 测试 | `train + val` | `test` |

验证时不能把 `val` 放进历史，测试时不能把 `test` 放进历史。否则模型已经看到了要预测的答案，这就是时间泄漏。

## Dataset 输出约定

### 训练样本

`SASRecTrainDataset` 返回：

| 字段 | dtype | 单个样本形状 | 用途 |
|---|---|---|---|
| `user_id` | `torch.long` | `[]` | 追踪用户，便于调试 |
| `input_ids` | `torch.long` | `[max_seq_len]` | 模型输入 |
| `target_ids` | `torch.long` | `[max_seq_len]` | 每个位置的下一 item |
| `padding_mask` | `torch.bool` | `[max_seq_len]` | 标记真实位置 |

DataLoader 组成 batch 后，后三者的形状前面会增加 `batch_size`，例如 `[256, 50]`。

### 验证或测试样本

`SASRecEvalDataset` 只有一个待排序的真实目标，所以返回标量 `target_id`，而不是长度为 `max_seq_len` 的 `target_ids`。

## 为什么 DataLoader 使用固定随机种子

训练时可以打乱用户样本的出现顺序，但不能打乱每个用户序列内部的时间顺序。`torch.Generator().manual_seed(42)` 让 batch 顺序在相同环境下可复现，同时序列内部仍然保持时间顺序。

## 如何验证

在服务器的项目环境中运行：

```bash
python -m unittest tests.test_dataset -v
```

测试会确认：

1. 短序列能正确左侧补零。
2. 过长序列只保留最近交互。
3. 无法构成下一 item 目标的序列会被拒绝。
4. 验证和测试历史没有未来数据泄漏。
5. DataLoader 可以把单个样本堆叠成正确的 batch 形状。

这一阶段通过后，下一步就是用相同数据合同实现 popularity baseline 和离线排序指标，然后再进入 SASRec 模型本身。
