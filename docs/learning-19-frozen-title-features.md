# 19：先把标题变成可复核的冻结特征

日期：2026-10-22。**特征准备链路已实现，本地 11 个轻量测试和完整标题对齐
审计通过；3 个 PyTorch 池化测试、服务器依赖检查、真实编码和全量缓存待验证。
目前没有标题融合训练或推荐效果结论。**

## 本轮目的和验收

此前验证集已说明：HSTU 总体优于均值，但在低频目标组落后；均值冻结测试也已完成。
标题是补充信息的候选，不能把缺少标题认定为根因。此次先完成一条可独立检查的
“原始标题 → 映射 ID → 冻结向量 → 文件缓存”链路，不同时修改推荐编码器和训练。

验收为：服务器小规模编码通过，然后完整缓存形状正确、padding 为零、有限值、
非 padding 行单位范数、保存恢复一致、原始和处理后文件哈希不变。它不等于推荐提升。

## 已核对的输入

本地 `data/raw/MicroLens-50k_titles.csv` 有 19,220 个唯一物品 ID，全部映射物品
都有非空标题，无多余物品；重复文本的额外行有 87 条，仍保留为不同物品。
折叠空白后字符数最小/中位数/最大为 4/109/6293。798 条包含非 ASCII，88 条
包含中日韩统一表意字符；这不是语言识别，不能据此宣称全部为英文。
英文样本支持先尝试轻量英文模型，但翻译噪声、截断和少量其他语言需要记录。

原始标题文件 SHA-256：
`244aad5380cbbe0fb43458cfcda5ebe820f534384602f80a64dbbcd07dd30e49`。
按映射排序并折叠空白后的标题表摘要：
`e4dee17e5a4fbdf31d79d1462ccc60411c81051e125c48ba28c7de86bbfdcdf7`。

`title_features/data.py::align_titles` 按 `idx2item` 查询原始 ID，
不把 CSV 行号或原始物品编号当作 embedding 行号。保留大小写、标点和 hashtag，
只折叠 Unicode 空白；缺失、空标题、重复物品 ID、不可逆映射立即失败。

## 编码器和形状

候选为 [all-MiniLM-L6-v2 官方模型](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)，
输出 384 维，官方用法是按 attention mask 对 token 隐藏状态取均值后 L2 归一化。
模型修订固定为 `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，查询依据为
[模型仓库元数据](https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2)。
不使用浮动 main，也不下载 ONNX、视频或音频资源。

配置 `configs/title-features-minilm-server.json` 固定 Transformers 4.57.1、
FP32、eager attention、最多 256 tokens（含特殊 token）、batch 32、seed 42；
服务器实际 GPU 必须重新检查。256 是该文本模型的截断约定，并非用户历史长度。
每条标题超过长度上限时记录截断数，不能把标题字符数当成 token 数。

`title_features/encoder.py` 的数据路径：

1. 标题列表 `[B]` → tokenizer 的 ID/mask `[B,T]`。
2. 冻结模型 `.eval()`、`requires_grad_(False)` 和 inference mode → `[B,T,384]`。
3. 排除 padding token 的均值池化、保留特殊 token，再归一化 → `[B,384]`。
4. 按映射行号写入 `[19221,384]` FP32，0 行严格为零；完整数组约 28.16 MiB。

这里平均的是一个标题里的 token，与上一阶段平均用户历史物品的模型不是一回事。
该阶段没有优化器、负采样或推荐损失。后续 HSTU 的 64 维表示不能直接和 384 维相加，
需要显式维度适配；本轮不预设已经实现初始化或融合。

实现检查同一批重复推理完全一致，单标题与带 padding 的批量编码在 1e-5 内一致，
所有编码器参数无梯度；保存 NPY 后逐元素恢复一致。记录模型/词表文件哈希，
以及 Git、配置、源码、输入、包版本、设备、耗时、峰值显存。只下载固定版本的
配置、safetensors 和 tokenizer 文件，并禁止远程自定义代码。

## 缓存使用和信息边界

每个动作写入独立 `runs/title-features-minilm/<UTC>/`，包括 `results.json`，
smoke/build 还保存 `features.npy`；不会覆盖原数据或基线产物，也不会提交 Git。
smoke 只编码前 64 个映射物品，不能用作完整训练缓存。

`title_features/cache.py::load_cache(directory, expected_inputs)` 拒绝失败、smoke、
部分目录、非冻结模型、不同标题/映射/数据摘要、损坏文件或错误形状/范数。
它返回只读内存映射数组和报告；后续模型需把此缓存身份写入自己的 provenance。
新代码位于 `title_features/` 和新脚本，旧 `src/`、`controls/` 和配置不变。

编码不使用交互标签，不根据验证/测试目标挑标题；把整个已知目录的静态标题
编码为特征，与既有“目录已知”协议一致。但标题在历史时刻是否已经可用并无证明，
不能称为严格全局时间回放，也无法排除预训练语料与标题的重叠。
文本相似不等于用户喜欢，先前测试结果不能作为后续标题融合的选参信号。

## 现在先在服务器执行

```bash
cd "$HOME/AgenticRec"
git pull --ff-only
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
python -m unittest tests.test_title_data tests.test_title_cache tests.test_title_preflight tests.test_title_encoder -v
python scripts/prepare_title_features.py preflight
```

预期 14 个测试 OK；preflight 不安装依赖、不下载模型、也不运行文本模型。
它报告 Python/Conda/pip、PyTorch/CUDA runtime、实际 GPU/驱动、包版本、pip check、
nvcc/sbatch 是否存在，以及标题和三个处理后文件是否齐全。
缺依赖/文件时结尾为 `NEEDS_SETUP`，并保存完整检查结果；这不等于编码失败。
请返回测试结尾与 preflight 输出，再根据实际依赖确定安装动作。

这符合 [AGENTS.md](../AGENTS.md) 要求：
“Before creating or changing the server environment, ask for or inspect the server's Python,
Conda, PyTorch, CUDA runtime, package-manager, and job-scheduler details.”
不能在尚未看过当前安装情况时先改动已验证的环境。

标题 CSV 没有提交 Git；如果 preflight 显示缺失，要从 Mac 原始文件单独传到
服务器 `data/raw/MicroLens-50k_titles.csv`，然后验证上面的 SHA-256。
`git pull` 只同步代码，不会带上标题文件。

## 检查后可执行的完整入口

依赖声明在 `environments/title-features-requirements.txt`，只有固定 Transformers，
不含 torch extra、不要求更换现有 PyTorch；实际安装方案以返回的 preflight 为准。
版本依赖依据是 [Transformers 4.57.1 官方 setup](https://github.com/huggingface/transformers/blob/v4.57.1/setup.py)。
该文件不是完整环境锁，编码报告会保存实际包版本，服务器验证后再冻结最终环境快照。

文件与依赖齐全后按顺序执行；每一步 PASS 才进入下一步：

```bash
python scripts/prepare_title_features.py audit
python scripts/prepare_title_features.py smoke
python scripts/prepare_title_features.py build
```

预期 audit 与本地的标题/映射摘要相同；smoke 最后为 `Title features smoke: PASS`，
64 个物品对应 `[65,384]`；build 最后为 `Title features build: PASS`，
19,220 个物品对应 `[19221,384]`。网络不可达时保留失败报告，诊断下载路径；
不自动换编码器或解除版本固定。模型下载和真实 CUDA 行为均待服务器验证。

完整缓存通过后再交付下游 ID-only、内容初始化、简单融合对照；必须同时定义
历史输入和候选输出如何使用内容，并预先固定训练预算、验证选择和无校正比较规则。
这不是当前已经完成的模型功能。
