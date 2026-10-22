# 18：冻结均值对照，再评估测试集

日期：2026-10-22。入口已实现，6 个纯 Python 合约测试本地通过；真实服务器
预检、既有 12 个神经/配置测试的结尾及均值模型测试集结果仍待返回。

## 为什么现在可以测试

均值模型训练预算固定为 30 轮，以验证 NDCG@10 选择第 19 轮，精确值
0.02965074791224214。四个历史长度组 HSTU 均胜出；低频目标组 Mean 反而更好，
见[笔记 17](learning-17-prefix-mean-control.md)。现在补齐冻结对照的测试记录，
不根据测试成绩改变学习率、轮次、打分校正或融合权重。

## 输入、检查与输出

新增 `scripts/evaluate_mean_control.py`，只使用标准库和既有纯 Python 配置校验。
输入是 Mean/HSTU 两个训练 `results.json` 和 Mean 配置，都是结构化字典，
训练历史为长度 30 的列表；不读取标签来重新选择配方。

预检要求：

- 两次训练均 PASS，配置与当前冻结配置一致，历史包含全部轮次。
- 处理后数据哈希、训练/评估用户身份、共享源码身份配对。
- 当前共享源码与控制源码、数据文件仍符合训练哈希；额外 src 文件也会拒绝。
- Mean 每轮选择指标来自验证集，最高分同分保留最早轮次。
- 使用训练报告旁的 `best.pt`，实际哈希与训练报告所记值一致。

通过后用当前 Python 调用既有 `controls.run_mean_pooling evaluate --split test`，
保留它原有的数据、配置、checkpoint、源码校验及全目录排序协议。
不修改 `src/` 或 `controls/`，因而不改变已有检查点绑定的模型源码身份。
入口没有 alpha、轮次或 checkpoint 覆盖参数。`--check-only` 仅预检，不运行模型。

评估仍由原入口创建独立 UTC 目录、输出 JSON 分组指标并记录输入 checkpoint
及其 provenance、当前环境、数据和源码身份。不会覆盖训练结果或 checkpoint。
预检失败立即停止，不自动修补数据、配置或旧报告。预检是配对与完整性检查，
不能代替数值、因果性或恢复行为的神经测试。

## 服务器命令

```bash
cd "$HOME/AgenticRec"
git pull --ff-only
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
nvidia-smi
python -m unittest tests.test_mean_frozen_test tests.test_mean_config tests.test_mean_pooling tests.test_mean_runner -v
```

预期 18 个测试 OK；然后执行一次冻结评估：

```bash
python scripts/evaluate_mean_control.py \
  --mean-report runs/mean-pooling/20261022T160521830353Z/results.json \
  --hstu-report runs/hstu-microlens/20261022T023420952080Z/results.json
```

预期先打印 `Frozen mean preflight: PASS` 和第 19 轮选择记录，随后是测试集
分组指标、`Saved: .../results.json` 和 `Mean pooling evaluate: PASS`。
请返回测试结尾、评估输出和结果路径。均值模型的测试指标目前未知，不能预填。

如果只想核对文件，给最后命令添加 `--check-only`。
本地 6 个小测试覆盖合法配对、最早同分、拒绝测试选择、未完成历史、
配置/用户身份差异、源码/数据/检查点修改和新共享源码文件。
本地没有运行 PyTorch 或实际推荐评估；只有服务器完成后才能称测试集已验证。

评估结束先报告同协议的 Mean、原始 HSTU、固定校正 HSTU 结果。
之后若推进标题实验，配方选择仍只用验证集，不能把本次测试反馈当成选参信号。
