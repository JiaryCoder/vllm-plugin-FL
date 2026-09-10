# 第二阶段：复合算子与标准 Attention reference

后续已增加[选择性 reference](reference-selection.md)：通过 INCLUDE/EXCLUDE 主动选择算子；
显式排除优先于下文的默认 reference 候选链，可单独使用 vLLM Triton Attention。

本阶段在第一阶段的统一路由上扩展支持范围。仍按以下顺序选择：
**经过审查的 vLLM torch 代码 → 插件 torch 代码 → 严格报错 / 非严格使用允许的优化实现**。
`forward_native` 的名字、继承关系和旧的 `reference.torch` 标签都不作为纯 torch 的证明。

验证环境：容器 `zjr-reference-0911`，vLLM 0.24.0、PyTorch 2.11.0+cu130、
FlagGems 5.4.0rc1.post1+g634cabe31，CPU 与 NVIDIA H20。
其他厂商需要在其运行环境重新验证。

## 覆盖范围

| 组 | 本阶段支持 | 复用与补充 |
| --- | --- | --- |
| 第一阶段 | 原有 5 个 dispatch 算子、普通 RoPE/MRoPE/ApplyRotaryEmb、RMSNorm/Add RMSNorm IR | 全部保留 |
| 扩展 CustomOp | GemmaRMSNorm、RMSNormGated、SwigluOAIAndMul、SwigluStepAndMul、MRotaryEmbeddingInterleaved、Ernie4_5_VLRotaryEmbedding | 复用 vLLM torch 方法；Gemma 直接绑定 IR native；Interleaved 使用其实际 `forward` |
| MoE 路由 | topk_softmax、grouped_topk、GroupedTopk CustomOp | 高层 grouped_topk 优先复用 vLLM CPU torch 函数；低层预计算 score 接口使用插件 torch |
| MoE 计算 | moe_align_block_size、moe_sum、专家 GEMM、完整未量化专家计算 | 普通完整专家计算优先复用 vLLM CPU torch MoE；expert_map、额外激活、INT8 等使用插件补充 |
| W8A8 | offset-binary INT32 权重解包、动态逐 token INT8 量化、scaled linear、动态 W8A8 MoE | 复用已有插件 torch 量化代码，接入线性 kernel 选择、权重装载和两个 W8A8 experts bridge |
| GDN/FLA | causal_conv1d_fn/update、chunk/recurrent gated delta rule、l2norm、post-conv prep、gdn gating、fused sigmoid gating update、packed decode | 卷积复用 vLLM CPU torch 实现；递推与融合准备逻辑由插件 torch 补充 |
| 标准 Attention | decoder self-attention，prefill、chunked prefill、decode，MHA/GQA/MQA，因果/非因果 mask，sliding window、ALiBi、soft cap、paged KV 与 KV sharing | 新的 ATen backend；矩阵乘与 softmax 显式执行，不使用默认可能选择 FlashAttention 的 SDPA |

当前 FL dispatch 的 **11 个入口**均存在有明确条件的 torch reference 路径：
`silu_and_mul`、`gelu_and_mul`、`rms_norm`、`rotary_embedding`、
`dynamic_per_token_quant_int8`、`topk_softmax`、`grouped_topk`、
`moe_align_block_size`、`moe_sum`、`invoke_fused_moe_triton_kernel`、
`attention_backend`。最后两个旧 reference wrapper 不会再作为纯 torch 候选使用。

具体支持还包括 FP32/FP16/BF16、RMSNormGated 分组与门控顺序、MRoPE 位置轴、
MoE 非本地 expert 的零贡献、原地输出、路由权重施加位置、连续批处理状态槽、
GDN 可变长度与部分 speculative state-table 场景。
GDN 的 vLLM 0.24 状态布局明确为 `[N, HV, V, K]`。

## 启用与回退

必须启动新进程，在模型构建之前设置：

```bash
export VLLM_FL_REFERENCE_MODE=1
export VLLM_FL_STRICT=1
export VLLM_FL_REFERENCE_REPORT_DIR=/tmp/fl-reference-report
# 使用原来的 vLLM 启动命令。
```

worker 会关闭模型 compile/CUDA graphs，跳过 `flag_gems.enable()`，
并将 FP32 matmul 精度设为 highest，禁止 cuDNN TF32。
若 FlagGems 的 ATen 替换已经启用，reference 会拒绝执行，请重建 worker。
不支持在已装载的模型中动态切换模式或 KV/权重布局。

`VLLM_FL_PREFER=reference` 单独使用仍是旧的优先级模式。
`VLLM_FL_CONFIG` 指向的有效 YAML 会覆盖 policy 环境变量；
需要非严格模式时可写：

```yaml
prefer: vendor
strict: false
op_backends:
  attention_backend: [vendor:cuda, flagos]
```

reference 候选始终先尝试。非严格回退使用现有 policy 的顺序、厂商过滤和可用性。
CustomOp/自由函数没有 FL dispatch 时，回退到保存的原平台入口。
仅允许 `reference` 的 policy 不会凭空增加优化候选。

候选发现与能力检查失败可以回退；**实现开始执行后发生的异常不会重试**，
避免一次原地更新后再调用另一个实现。
Attention 的不支持配置在后端选择、分配 KV cache 之前检查。
已选定的 reference backend 不进行运行中的缓存布局迁移；
不支持的输出量化参数等会明确报错。

## 接入方式与容易遗漏的细节

- `functions.py` 把自由函数及已存在的模块 import 别名接到同一引擎；
  按函数对象身份替换，不按短名字全局猜测。
- CustomOp hook 拦截最终 `forward`，防止 GDN 构造函数后续重设
  `_forward_method` 覆盖路由。
- MoE 在权重转换前选择普通 `[E,N,K]` 布局。这里使用 Triton backend 的
  **装载布局约定**，计算会被 reference 接管。
  此类控制事件标记为 `reference.setup`，不计作执行 Triton kernel。
- vLLM CPU torch MoE 中的 SiLU 表项会构造 CustomOp，可能遇到 OOT 构造参数不兼容。
  适配器用隔离的函数 globals 将该表项绑定到既有静态 torch 方法，不改全局数学实现。
- CUDA ATen 不实现普通 `int32 @ int32`。W8A8 使用有界临时内存的整数乘加与
  INT64 reduction，再转 INT32；不以 float32 GEMM 冒充精确整数累加。
- GDN packed decode 的 beta 保留 FP32，与插件已有的
  `patch_vllm_packed_gdn_beta` 修复一致；GPU kernel 对照测试使用该既有修复。
  reference 本身不会发射这个 Triton kernel。

## 明确的边界

以下变体仍会被相应 reference 入口拒绝，或在非严格模式回落：

- MLA、稀疏/DSA attention、量化 KV、attention sinks、PrefixLM、
  cross/encoder attention、DCP/PCP、KV connector、融合输出量化。
- FP8/INT4/weight-only 或 block quantization、非对称/静态输入 W8A8、
  LoRA 与由 quant config 驱动的额外 clamped MoE。
- MoE DP/EP/SP/PCP dispatch collectives、异步共享专家调度。
  本地 expert_map 有测试；本阶段整模型验收采用单卡。
- 卷积 APC 的按块快照、speculative convolution snapshots。
  GDN recurrent 的多 token 原地状态写入要求二维 slot table；
  无索引 recurrent 仅审查了 B=1 的非原地接口。
- 老版 FL FLA CustomOp 的 `[K,V]` 状态布局及重复注册名字、
  未列入清单的类/变体。不会把它们当作 vLLM 0.24 的 `[V,K]` 接口执行。
- 第一阶段明确未审查的其他 RoPE/YaRN 等变体。

严格性与 ATen 检查作用于**已经接入路由且审查过的入口**。
调度器、采样器、通信、KV 元数据构建及任意直接 kernel 调用不因此自动成为 reference。
不能把本阶段的结果解释为整个进程或任意模型的所有调用都是纯 torch。
reference 很慢，目的是为精度诊断提供可检查的计算路径。

## 验证与复现

交付的 `validation.json`、JUnit、测试日志、模型输出与 routes 保存了本次结果。
验证包括原有 dispatch/IR 行为、严格/非严格分支、真实 import 别名、
数值与状态更新、CPU/GPU、原生 CUDA kernel 对照及真实 vLLM 引擎。

5 个本地随机小模型完成严格 reference 生成：
Llama、Qwen2 MoE、Qwen3-Next（GDN + MoE + Attention）、
W8A8 Llama、W8A8 Qwen2 MoE。
另验证了 Qwen3-Next 的 spawn/EngineCore 进程路径。
权重由下列工具生成，不需要下载模型：

```bash
python3 tools/reference_stage2_make_models.py --output /tmp/ref2-models
python3 tools/reference_stage2_model_smoke.py \
  --model /tmp/ref2-models/tiny-qwen3-next \
  --output /tmp/ref2-gdn.json --reference --multiprocess
python3 -m pytest tests/unit_tests/dispatch tests/unit_tests/quantization -q
```

模型 smoke 固定随机种子，以 token IDs 输入，关闭 prefix caching、启用 eager，
预留 64 MiB KV cache。去掉 `--reference` 会关闭 FL plugin，运行原生 vLLM 对照。

这些随机权重的生成是接入验证，**不是 benchmark 对齐验收**。
原生 vLLM 对照中，Llama/W8A8 Llama 的两条生成均一致；
Qwen2 MoE 与 Qwen3-Next 各有一条生成不同。
相同输入历史上的 top-logprob 最大绝对差约为：
Llama 0.00198，MoE 0.00211，GDN 0.03021，W8A8 Llama 0.00381。
MoE 的一次分歧发生在 reference 最高两个 BF16 logits 打平处；
GDN 差异仍需后续逐层/逐算子定位，不能统一归因于舍入或 beta。
分歧之后输入历史已不同，不再把后续 logprob 差当成同输入精度误差。

下一阶段可以基于实际模型和 benchmark，在相同输入历史下捕获中间量，
逐层/逐算子切换 reference 来定位差异。
