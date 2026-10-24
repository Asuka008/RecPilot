# 26：固定组合的精确推荐服务与性能测量

## 这一阶段解决什么

前一阶段已经观察到离线收益。现在将 alpha=.25 的 HSTU + ItemCF 封装为可重复
调用的推荐接口，检查封装前后的结果一致，并测量实际设备上的计算成本。
不重新训练，不用测试集重新选权重，不引入 ANN。19220 个候选全部评分。
这是本机演示服务，不是生产部署或在线收益证明。

## 代码路径与输入输出

- `serving/runtime.py` 启动时复用固定配方准入，检查已有验证报告、配置、源码、
  数据、检查点身份；模型只加载一次，ItemCF 只根据训练历史构建一次。
  当前仍依赖已有标题实验文件完成准入，但请求不运行标题模型。
- `serving/core.py` 接收 `{"history": [1, 2, 3], "k": 20}`。ID 必须是当前
  item_mapping.json 的映射 ID，按从旧到新排序；不是原始视频 ID。
  核心返回 ID、排序分数、兜底原因、候选数量和模型身份。
- `TorchScorer` 保留最近 L=32 个事件，左侧补零，生成 `[1,32]` int64 输入和
  bool 掩码，HSTU 最后位置输出 `[1,64]`，与全部候选评分得到 `[1,19220]`。
  ItemCF 和已看过滤仍使用完整传入历史；重复观看保留时序，CF 使用去重历史。
- 组合直接复用已验证的评分函数与排序规则。固定 alpha=.25，不接受权重参数。
  排序分数不是点击概率。完全相同的分数按映射 ID 升序打破平局。
- 空历史使用训练频次热门榜；CF 无可用信号时回退原始 HSTU。padding、未知 ID、
  布尔值、超长输入和非法 K 均报错；候选耗尽时返回少于 K 个，允许空列表。
  这两种兜底的质量没有额外实验支持，不称其冷启动解决方案。

服务配置与训练配置分开：server 使用 cuda:0；debug 使用 CPU。
不改写历史训练配置。接口限制 4096 个历史事件、K<=100、请求体<=64KiB。
`serving/http.py` 使用标准库串行 HTTPServer，绑定 127.0.0.1；不添加依赖。
请求不查找用户表，因此不会自动把某个用户的验证或测试目标放进历史。

## 先在服务器做一致性与性能检查

```bash
git pull --ff-only
python -m unittest tests.test_serving tests.test_serving_torch -v && \
python -m serving.run benchmark --config configs/hybrid-serving-server.json
```

预期 7 个测试 OK、Artifact admission: PASS、Serving benchmark: PASS。
本地 6 个轻量测试通过，覆盖请求、兜底、过滤、组合和 HTTP 适配；本地未运行
PyTorch。第 7 个微型 CPU 模型测试和真实推理/性能等待服务器验证。

benchmark 使用按训练历史长度排列、均匀取位置的 100 个样本，包括长历史。
不计算任何测试集质量指标。先用原 Dataset 的验证输入适配和原离线批量大小
评分，与服务单请求逐个对照：分数绝对误差<=1e-6，Top-20 必须完全一致。
如失败需检查误差或边界平局，不静默放宽标准。

之后在同一组历史上测原始 HSTU 和组合，每路预热 20 次、测量 100 次，
batch=1、并发=1、FP32、TF32 关闭，每次计时前后 CUDA 同步。
输出 p50/p95/p99、均值、单路串行处理速率，以及原始时间样本。
计时覆盖请求校验、HSTU、GPU->CPU 传输、CF 和排序，**不包含 HTTP、排队或启动**。
两路共享已加载的模型/CF，原始 HSTU 路径跳过 CF 评分，便于比较额外评分开销；
它不是两套独立部署的内存对比。原始先测、组合后测，100 次只是初步测量，
不能声称稳定尾延迟或并发吞吐上限。

报告保存到 `runs/hybrid-serving/<timestamp>/results.json`，含配置、数据/模型/源码
身份、Git、Python/Torch/CUDA/GPU、候选数、线程数、历史长度范围、启动耗时、
进程峰值 RSS 和每路 PyTorch 峰值显存。RSS 是进程自启动以来的峰值；显存不包含
所有外部分配，共享 GPU 负载也会影响延迟。推理服务不会下载模型或改写检查点。

## 通过后启动接口

```bash
python -m serving.run serve --config configs/hybrid-serving-server.json
```

另开同一服务器终端：

```bash
curl --fail http://127.0.0.1:8765/health
curl --fail http://127.0.0.1:8765/recommend \
  -H 'Content-Type: application/json' \
  -d '{"history":[1,2,3],"k":10}'
```

示例 ID 是合成请求，不代表真实用户。响应不应包含 0、1、2、3，不应重复。
`history: []` 应返回 `fallback: empty_history_popularity`；非法 ID 返回 HTTP 400。
启动输出 Ready 表示模型已经加载。Ctrl-C 结束服务。
当前只提供本机单进程演示；HTTP 端到端延迟、并发、鉴权和持久化索引尚未验证。
先回传 benchmark 的运行设备、parity、两路 measurements 和保存路径，再决定是否
需要优化，不能从离线训练时长推断服务延迟。

## 2026-10-24：服务器一致性与核心延迟 PASS

用户回传 7 个测试通过（0.175s）、Artifact admission: PASS 和 Serving benchmark: PASS。
报告：`runs/hybrid-serving/20261024T204850491181Z/results.json`。
本地未直接读取完整报告；工作负载长度分布和进程峰值 RSS 尚未回传。

100 个训练历史样本的最大分数误差为 2.123415470123291e-7，小于 1e-6；
Top-20 完全一致。此结果支持本次样本上的封装正确性，不代表所有可能请求都已验证。

| 单请求核心延迟 | 原始 HSTU | HSTU + ItemCF |
|---|---:|---:|
| 平均 | 3.354 ms | 3.705 ms |
| p50 | 3.348 ms | 3.612 ms |
| p95 | 3.438 ms | 3.932 ms |
| p99 | 3.546 ms | 4.505 ms |

平均增加约 0.351ms（10.45%）。结合已完成的离线收益，当前无需立刻增加 ANN
或更换模型，先完成真实 HTTP 调用检查。串行均值倒数约为 298.14/269.93 次每秒，
不是并发压测吞吐量；100 次测量的 p99 也不是稳定生产尾延迟。

运行环境：A100-SXM4-80GB MIG 2g.20gb，可见显存 20,937,965,568 字节，
Python3.11.16、Torch2.10.0+cu128、NumPy1.26.4、CUDA runtime12.8，FP32/TF32关闭，
24 个 Torch CPU 线程；19220 个候选、100 个 CF 邻居、1,111,056 条 CF 边。
启动 11.377s。两路峰值 allocated/reserved 为 49,789,952/62,914,560 字节
（约47.48/60MiB）；CF 位于 CPU，两路共享已加载资源，不能据此声称总服务只占这些内存。
HTTP 实际启动与调用仍待确认，使用上一节命令即可，不必重跑本次 benchmark。

## 服务器真实 HTTP 调用通过

用户随后回传 `/health` 返回 ready、catalog_items=19220；
`/recommend` 对 history=[1,2,3]、k=10 返回 10 个不同的映射 ID，均非 0/1/2/3，
分数按降序排列。eligible_items=19217、fallback=none、mode=hybrid。
两次响应 model_id 相同：
`17aaae44451f3d9fe5de7e4ee9a6e2e3a3ff981bf55374c4d836227f76d9abdb`。
这确认了真实 HTTP 请求到模型推理再到排序返回的调用链。请求为合成历史，
不证明推荐内容符合真实用户兴趣，也未测量 HTTP 端到端延迟或并发能力。
接口基本调用检查已完成，不必重复；空历史与非法输入此前有轻量适配测试，
其真实网络调用尚未单独回传。
