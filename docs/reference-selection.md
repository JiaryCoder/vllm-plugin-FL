# 选择性 reference

首次使用可先阅读[Reference 模式使用说明](reference-quickstart.md)，包含开启、按算子切换和原版对照命令。
标准 CustomOp/IR 可通过[动态 native 发现](reference-dynamic-native.md)接入；算子组也包含动态发现的类。

在 vLLM 0.24.0 的第一、二阶段 reference 路由上增加按算子选择功能。
可保持其他算子严格 reference，单独让 Attention 使用 vLLM Triton；也可只对指定算子启用 reference。

**默认 Attention 为 `TRITON_ATTN`，其余已接管复合算子仍按 reference 路由。**
需要纯 Torch Attention 时显式设置 `VLLM_FL_REFERENCE_ATTENTION_BACKEND=TORCH`。

## 配置

仍需设置 `VLLM_FL_REFERENCE_MODE=1`，并在模型构建前配置：

| 环境变量 | YAML 字段 | 含义 |
| --- | --- | --- |
| `VLLM_FL_REFERENCE_INCLUDE` | `reference_include` | 只让这些算子走 reference；未设置或 `all` 表示全部入口 |
| `VLLM_FL_REFERENCE_EXCLUDE` | `reference_exclude` | 将这些算子排除出 reference；默认空 |
| `VLLM_FL_REFERENCE_ATTENTION_BACKEND` | — | 默认 `TRITON_ATTN`；`TORCH` 为纯 Torch；`AUTO` 为原平台选择；也可填已注册的 vLLM backend 名称 |

环境变量使用逗号分隔，YAML 支持列表或逗号分隔字符串。显式空 INCLUDE 表示不选任何入口。
EXCLUDE 优先于 INCLUDE；有效 `VLLM_FL_CONFIG` YAML 完全覆盖相应环境变量。
名称拼写错误或非法类型立即报错。配置对象 `SelectionPolicy` 和 policy context 同样保留这两个字段。
设置 INCLUDE/EXCLUDE 不会自行开启 reference 模式。
Attention 后端设置独立于 YAML 的 include/exclude。`TORCH` 不会覆盖显式的 Attention 排除项；
要恢复纯 Torch，请同时清除 Attention 的排除项，并确保 INCLUDE 包含 Attention。
显式 CLI backend、Attention per-op order 优先于环境变量的优化后端设置；相互冲突的 CLI/per-op 组合会报错。

被选中的入口仍执行：审查过的 vLLM torch → 插件 torch → 严格报错 / 非严格回退。
被排除的入口主动使用保存的原平台实现，不受 reference 的缺失检查影响；执行失败直接抛出，不重试。
这不会关闭其他入口的 `VLLM_FL_STRICT=1` 检查。

对于 FL dispatch 入口，主动排除时默认仅允许 vendor 候选，仍遵守 vendor 黑白名单和 `op_backends` 限制；
没有允许的 vendor 会报错，不能悄悄改用 FlagGems 或 reference。
Attention 是可显式配置的例外：同时排除 `attention` 并设置其 per-op 后端时，允许指定的非 reference
候选，包括 `flagos`。这不放宽其他复合算子的 vendor 限制，也不启用 FlagGems ATen 替换。
CustomOp/IR/函数入口回到其原平台方法或 provider，未必每个原方法都是融合 kernel，实际路径以记录为准。
所有 reference 模式都会跳过 FlagGems OOT 算子/路由注册，也不启用 FlagGems ATen 替换。
如果当前进程此前已启用 FlagGems ATen 替换，执行 reference 或主动排除入口都会拒绝，请重建进程。

## Attention 使用 vLLM Triton

在容器 `zjr-reference-0911` 中，等模型下载完成后执行：

```bash
env -u VLLM_FL_CONFIG -u VLLM_FL_PER_OP -u VLLM_FL_REFERENCE_INCLUDE \
  CUDA_VISIBLE_DEVICES=0 \
  VLLM_PLUGINS=fl \
  USE_FLAGGEMS=0 \
  VLLM_FL_REFERENCE_MODE=1 \
  VLLM_FL_REFERENCE_EXCLUDE=attention \
  VLLM_FL_PREFER=vendor \
  VLLM_FL_STRICT=1 \
  VLLM_FL_REFERENCE_REPORT_DIR=/tmp/qwen36-selective-reference-routes \
  vllm serve /data/models/Qwen3.6-35B-A3B \
    --served-model-name Qwen3.6-35B-A3B \
    --attention-backend TRITON_ATTN \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --enable-chunked-prefill \
    --language-model-only \
    --max-model-len 2048 \
    --max-num-batched-tokens 256 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000
```

`VLLM_PLUGINS=fl` 加载我们实现的 reference 路由，不代表启用 FlagGems kernel。
CUDA 上 Attention 调用 vLLM 的 CUDA backend selector；其他平台通过已安装 vLLM 的 backend registry
解析显式选择，并保留厂商注册的 backend。两种路径均调用实际 backend 的 `validate_configuration`，
检查 dtype、head size、KV cache、MLA、设备能力等配置。
不支持指定后端时明确失败，不替换成另一种 Attention。默认设置下无需排除 Attention 或额外传 CLI 参数。
如果设置了 `TORCH`，不能同时传优化后端 CLI 参数而不排除 Attention，避免矛盾配置被静默忽略。
`AUTO` 且没有 CLI/per-op 选择时，由原平台选择 backend。
显式选择不再限制为 NVIDIA。实际可用的 backend 仍由当前镜像的实现、依赖和硬件决定；
不能据此认为所有平台都支持 FlashAttention、FlashInfer 或 Triton。

海光 reference 使用 vLLM Triton 的核心配置为：

```bash
export GEMS_VENDOR=hygon
export VLLM_PLUGINS=fl
export USE_FLAGGEMS=0
export VLLM_FL_REFERENCE_MODE=1
export VLLM_FL_PREFER=vendor
export VLLM_FL_STRICT=1
export VLLM_FL_REFERENCE_ATTENTION_BACKEND=TRITON_ATTN  # 默认值，可省略
unset VLLM_FL_CONFIG VLLM_FL_OP_CONFIG VLLM_FL_PER_OP
unset VLLM_FL_REFERENCE_INCLUDE VLLM_FL_REFERENCE_EXCLUDE
# 接原有 vllm serve 命令即可，不需要额外的 Attention 参数。
```

也可以使用 FL 的 per-op 选择（此方式不再同时传 `--attention-backend`）：

```bash
export VLLM_FL_REFERENCE_EXCLUDE=attention
export VLLM_FL_PER_OP='attention_backend=flagos'
export VLLM_FL_USE_FLAGGEMS_ATTN=0
# 当前 default.flagos 选择器在此配置下返回 vLLM 的 TritonAttentionBackend。
```

两种选择都会记录实际 backend 类和 `user_override`。显式选择执行失败或能力校验失败时直接报错，
即使 `VLLM_FL_STRICT=0` 也不在执行后自动重试另一种 Attention。
恢复纯 Torch Attention 时，设置 `VLLM_FL_REFERENCE_ATTENTION_BACKEND=TORCH`，清除 Attention 的排除项，
确保 INCLUDE 包含 Attention，并删除 `--attention-backend`。其余复合算子的选择和严格模式不受影响。
默认 Triton 不可用时会明确报错；可设置 `TORCH`，或选择当前平台安装且支持的其他 backend。

Attention 在 KV cache 分配之前选定；MoE/W8A8 的选择也涉及权重装载和布局。
**所有选择配置在启动时确定，修改后要重启整个服务，不能在已加载模型中热切换。**
上述 Qwen3.6 完整权重模型尚未做整模型验收；这是基于已安装实现及模型配置给出的启动示例。

## 只选择一部分算子

例如只让 RMSNorm 系列走 reference，其余走原平台实现：

```bash
export VLLM_FL_REFERENCE_MODE=1
export VLLM_FL_STRICT=1
export VLLM_FL_PREFER=vendor
export USE_FLAGGEMS=0
export VLLM_FL_REFERENCE_INCLUDE=normalization
unset VLLM_FL_REFERENCE_EXCLUDE VLLM_FL_CONFIG
# 接上原来的 vllm serve 命令。
```

等价的 YAML 核心配置：

```yaml
prefer: vendor
strict: true
reference_include: [normalization]
reference_exclude: []
```

常用选择名称：

| 名称 | 范围 |
| --- | --- |
| `attention` / `attention_backend` | 标准 Attention 的后端选择与计算；不包含 GDN |
| `rms_norm` / `RMSNorm` / `RMSNormFL` | 普通 RMSNorm，包括其残差版本及两个 RMSNorm IR 入口 |
| `normalization` | RMSNorm、GemmaRMSNorm、RMSNormGated |
| `activation` | SiluAndMul、GeluAndMul、SwigluOAIAndMul、SwigluStepAndMul |
| `rope` | 当前审查清单中的普通 RoPE、ApplyRotaryEmb、MRoPE 及扩展变体 |
| `moe` | 路由、对齐、专家计算、激活、求和；包含完整专家计算和对应装载布局选择 |
| `gdn` | causal conv、chunk/recurrent gated delta rule、l2norm、门控和 packed decode 等 9 个逻辑入口 |
| `w8a8` | 动态量化、权重解包和 W8A8 Linear；专家整体计算由 `fused_experts` / `moe` 控制 |

也可单独使用清单中的逻辑名，例如 `silu_and_mul`、`gemma_rms_norm`、`mrope`、`fused_experts`、
`topk_softmax`、`moe_sum`、`chunk_gated_delta_rule`、`l2norm_fwd`、`w8a8_linear`。
支持的完整类名和函数名按显式别名表映射；完整映射见 `vllm_fl/reference/selection.py`。

## 选择粒度与记录

选择作用于实际到达 hook 的入口，不会拆解融合 kernel，也不会把 reference 函数中内联的数学运算重新变成独立算子。
例如完整 torch MoE 已在内部直接求和时，单独排除 `moe_sum` 不会改变这段内联求和；可以先排除完整的
`fused_experts`，在更外层比较整个专家计算。主动排除复合入口时，它的内部调用整体沿原实现执行。
MoE 权重装载与完整专家计算属于同一个选择单元，W8A8 Linear 的初始化和计算也属于同一个单元。
新增选择器不会扩大原有 reference 数值实现的支持范围，也不承诺任意模型的未接管调用均被检查。

每个 worker 的 JSONL 路由记录区分：

- `vllm.native` / `plugin.torch`：实际 reference 候选。
- `vllm.native.dynamic`：动态发现的 vLLM native 接口，见动态发现文档中的检查边界。
- `reference.setup`：reference 所需的布局等初始化约定。
- `user_override`：主动排除或配置的优化 Attention（包括默认 Triton），记录实际函数/provider/backend 及原因。
- `user_override_error`：主动选择的实现执行失败，没有自动重试。
- `reference.mixed`：某个 reference 父调用内实际发生了显式优化子调用，不能将父调用整体视为纯 torch。
- `optimized_fallback`：reference 不可用时的非严格回退，与主动排除分开。

标准 native CustomOp 的初始化选择也由上述环境变量自动控制，用户无需填写
`CompilationConfig`。按实际类隔离配置，处理共享注册名；构造后改变选择会报错并要求重建 worker。
非严格回落只在初始化兼容时执行：没有 native 的算子可按优化路径构造并回落；
已按 native 构造、运行时才发现不支持输入的算子，不会在相同状态上盲目改跑 vendor。

Attention 的记录应包含：

```text
attention_backend -> vllm.v1.attention.backends.triton_attn.TritonAttentionBackend (user_override)
attention -> vllm.v1.attention.backends.triton_attn.TritonAttentionImpl.forward (user_override)
```

## 复现验证

```bash
python3 -m pytest tests/unit_tests/dispatch tests/unit_tests/quantization -q
python3 tools/reference_stage2_make_models.py \
  --output /tmp/my-selective-models --attention-head-size 64
python3 tools/reference_stage2_model_smoke.py \
  --model /tmp/my-selective-models/tiny-qwen3-next \
  --output /tmp/my-selective-result.json --reference --multiprocess \
  --reference-exclude attention --attention-backend TRITON_ATTN
```

模型是本地固定种子的随机小模型，不需要下载权重。选择 64 的 head size 是为了满足 vLLM Triton 后端约束；
旧测试模型的 head size 16 不被该后端支持。这些验证证明路由和推理接通，不代表真实模型的 benchmark 精度对齐。
