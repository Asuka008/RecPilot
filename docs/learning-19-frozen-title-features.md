# 19：先把标题变成可复核的冻结特征

日期：2026-10-22。**特征准备链路已实现，本地 11 个轻量测试和完整标题对齐
审计通过；服务器 14 个聚焦测试已通过，依赖安装、预检、标题哈希、服务器对齐审计和 64 条标题 CUDA 编码和全量缓存 build 均已通过；正式缓存读取核对待返回。
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

依赖声明在 `environments/title-features-requirements.txt`，服务器预检后已固定四个文本包，
不含 torch extra、不要求更换现有 PyTorch；实际安装方案以返回的 preflight 为准。
版本依赖依据是 [Transformers 4.57.1 官方 setup](https://github.com/huggingface/transformers/blob/v4.57.1/setup.py)。
该文件不是完整环境锁，编码报告会保存实际包版本，服务器验证后再冻结最终环境快照。
另用 `title-features-constraints.txt` 保持已验证的 torch/numpy 版本。

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


优先级确认：用户在讨论 MiniOneRec 后明确选择先继续本笔记的标题特征流程。
MiniOneRec 留待主线标题实验和实测服务形成结果后再评估，不改变当前验收门槛。
服务器已返回 14 个测试 OK 和 preflight；当前栈与缺失项见下，可按核对后的约束安装。


## 2026-10-23 服务器预检与安装步骤

用户返回 14 个测试全部 OK，耗时 0.360 秒。预检报告：
`runs/title-features-minilm/20261023T011421721161Z/results.json`，状态 NEEDS_SETUP。

- Python 3.11.16，pip 26.2.1，Conda 26.9.2；pip check 无冲突。
- PyTorch 2.10.0+cu128，NumPy 1.26.4，CUDA runtime 12.8，可用 CUDA。
- 实际分配 A100-SXM4-80GB MIG 2g.20gb，20937965568 bytes；驱动 570.211.01。
- nvidia-smi 的整卡显存查询受限，分配内存以 PyTorch 设备属性为准。
- nvcc 和 sbatch 不存在；当前冻结推理流程不需要编译 CUDA 扩展或调度脚本。
- Transformers/tokenizers/huggingface-hub/safetensors 未安装；三个处理后文件齐全，标题 CSV 缺失。

已核对的文本包候选为 Transformers 4.57.1、huggingface-hub 0.36.0、
[tokenizers 0.22.1](https://pypi.org/project/tokenizers/0.22.1/)、
[safetensors 0.6.2](https://pypi.org/project/safetensors/0.6.2/)。
[Transformers 的官方依赖声明](https://github.com/huggingface/transformers/blob/v4.57.1/setup.py)
允许 hub >=0.34,<1.0，tokenizers >=0.22,<=0.23 和 safetensors >=0.4.3；
[Hub 0.36.0](https://pypi.org/project/huggingface-hub/0.36.0/) 为固定的兼容候选。
实际跨包导入、模型下载与运行仍需服务器验证，不能将版本区间匹配当作已执行通过。

先在服务器运行（安装无需重跑原来的 14 个已通过测试）：

```bash
cd "$HOME/AgenticRec"
git pull --ff-only
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
mkdir -p data/raw
python -m pip install -r environments/title-features-requirements.txt -c environments/title-features-constraints.txt
python -m pip check
```

通过 JupyterLab 文件浏览器将 Mac 的原始 `data/raw/MicroLens-50k_titles.csv`
上传至服务器 `~/AgenticRec/data/raw/`。文件约 2.3 MB，不加入 Git，不用 git pull 传输。
服务器此前缺失该文件，上传的是已有原始数据，不是生成或改写标题。

上传后验证完整字节哈希：

```bash
python - <<'PYTITLE'
import hashlib
from pathlib import Path
p = Path("data/raw/MicroLens-50k_titles.csv")
actual = hashlib.sha256(p.read_bytes()).hexdigest()
assert actual == "244aad5380cbbe0fb43458cfcda5ebe820f534384602f80a64dbbcd07dd30e49", actual
print("Title CSV checksum: PASS")
PYTITLE
```

再执行；每个动作 PASS 后进入下一步：

```bash
python scripts/prepare_title_features.py preflight
python scripts/prepare_title_features.py audit
python scripts/prepare_title_features.py smoke
```

预期依次为 preflight/audit/smoke PASS。smoke 首次下载固定模型快照，然后生成
64 个物品的 `[65,384]` 特征；0 行是 padding。请返回这次 smoke 的输出及结果路径，
再核对数值、缓存恢复和资源后全量编码。若安装、导入或网络步骤失败，返回错误，
保留当前报告诊断，不通过换版本或换 PyTorch 绕过失败。


## 2026-10-23 安装与 64 条标题编码通过

用户返回安装成功，四个文本包与固定版本一致；pip check 无冲突，BertModel
导入成功。PyTorch 2.10.0+cu128、NumPy 1.26.4、CUDA runtime 12.8 保持原值，
实际设备仍为 A100 MIG 2g.20gb。标题原始字节哈希验证 PASS。

- preflight PASS：`runs/title-features-minilm/20261023T013402313329Z/results.json`。
- audit PASS：`runs/title-features-minilm/20261023T013439647115Z/results.json`。
- smoke PASS：`runs/title-features-minilm/20261023T013447156086Z/results.json`。

服务器审计的 19,220 物品、零缺失/空标题、87 条重复文本额外行以及对齐摘要
均与 Mac 一致。六个固定版本的模型/分词器文件成功下载，权重文件约 90.9 MB。
两个 batch 共编码 64 条标题。按入口约定，PASS 已执行冻结参数检查、同批重复
推理精确一致、单条/批量 padding 一致性、有限单位向量、零 padding 行、
NPY 保存恢复精确一致及输入/模型快照/源码未变检查。
日志未展示实际 smoke 耗时、峰值显存和文件哈希字段，不填写推测数值。

本次新增包日志记录为：certifi 2026.10.19、charset_normalizer 3.5.2、
hf-xet 1.7.0、idna 3.20、pyyaml 6.0.3、regex 2026.9.29、requests 2.34.2、
tqdm 4.70.1、urllib3 2.8.0，以及四个已固定的文本包。完整环境仍以运行报告
中的 packages 为准；这些观测不是另一次安装要求。

现在不重复安装、上传、单元测试或 smoke。下一步直接生成完整缓存：

```bash
python scripts/prepare_title_features.py build
```

模型将复用已下载的固定快照，仍按 batch 32 推理，输出 CPU 数组从 65 行扩大为
19,221 行（约 28.16 MiB），不是加大 GPU batch 或训练模型。预期最后进度为
`Encoded titles 19220/19220`，最后状态 `Title features build: PASS`。
请返回最后几行和新报告路径。保存的完整文件须为 `[19221,384]` FP32，且通过
完整数组的零 padding、单位范数、有限性和恢复检查；smoke 文件仍不能用于训练。

全量缓存通过后，先核对缓存加载及正式产物身份，再进入标题初始化/简单融合的
HSTU 对照实现。未完成此步前，不声称标题已接入推荐模型或已有推荐增益。


## 2026-10-23 全目录缓存 build PASS

用户返回 `Encoded titles 19220/19220` 和 `Title features build: PASS`。
产物报告为 `runs/title-features-minilm/20261023T013804609858Z/results.json`，
对应缓存文件为同目录 `features.npy`。按构建入口的完整检查，产物应为
`[19221,384]` FP32，零 padding 行、有限单位向量和保存恢复均通过，
输入/模型快照/源码哈希未变。没有重新训练推荐模型。

下一步用下游将调用的正式 `load_cache` 读取该产物；它重新核对当前原始标题、
处理后数据、配置、模型修订和 NPY 文件身份，拒绝 smoke 或被改动的缓存。
同时读取实际截断数量和资源字段，无需重复编码或重新安装。

```bash
python - <<'PYCACHE'
import json
from pathlib import Path
from title_features.cache import load_cache
from title_features.data import input_hashes

p = Path("runs/title-features-minilm/20261023T013804609858Z")
r = json.loads((p / "results.json").read_text())
x, r = load_cache(p, input_hashes(Path.cwd(), r["config"]))
summary = {"shape": list(x.shape), "dtype": str(x.dtype),
           "read_only": not x.flags.writeable,
           "features_sha256": r["features_sha256"], "checks": r["checks"],
           "encoder": r["encoder"], "elapsed_seconds": r["elapsed_seconds"],
           "memory_bytes": r.get("memory_bytes"), "gpu": r["runtime"].get("gpu")}
print(json.dumps(summary, indent=2))
print("Full title cache load: PASS")
PYCACHE
```

请返回这段小检查的输出，它不运行文本模型，只读现有文件。截断条数、耗时、
显存与缓存哈希尚未展示，记录中不填推测值。随后开始下游标题初始化/简单融合，
共享现有 HSTU 目标和评估规则，明确 384→64 适配及历史/候选表示。
本次 build 成功不是标题推荐效果已经验证，也不能将本次缓存直接称为 MiniOneRec SID。


## 完整缓存消费验证回传（2026-10-23）

用户已返回 `Full title cache load: PASS`，前面的待验证步骤现已完成：

- 形状 `(19221,384)`、FP32；完整缓存 SHA256：
  `af5c0482e9b37294e1e4817810d5e2b0b407c22759207078fd402f7ea0525b25`。
- 编码器 revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，可训练参数 0，
  eager attention、FP32、TF32 关闭；重复编码精确一致，batch padding 一致。
- 10 条标题截断；padding 全零、单位行、保存后精确恢复检查均通过。
- 全量编码 37.32873547798954 秒，峰值 allocated 515812864 bytes、
  reserved 828375040 bytes。六个模型/tokenizer 文件的哈希也已输出并记录在服务器报告。

无需重跑上述构建或读取。下游标题初始化/固定融合实现与服务器短训练命令见
[笔记 20](learning-20-title-hstu-controls.md)，推荐效果仍待实验。
