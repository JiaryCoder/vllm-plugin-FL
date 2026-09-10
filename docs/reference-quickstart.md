# Reference 模式使用说明（vLLM 0.24）

本说明适用于已按 [README](../README.md#setup) 安装本分支插件、使用 vLLM 0.24.0 的环境。
Reference 用于精度诊断：在已接入的算子入口，优先调用经过审核的 vLLM PyTorch 实现，
缺少可用实现时调用插件补充的 PyTorch 实现。当前仍使用显式支持名单。

**所有开关在启动新服务前设置。修改后必须重启整个服务，不支持在已加载模型中热切换。**
Reference 会关闭模型编译和 CUDA Graph，并阻止使用已激活的 FlagGems ATen 替换。

## 1. 开启 reference，保留 vLLM Triton Attention

以下是 NVIDIA 上的选择方式：标准 Attention 使用 vLLM Triton，其余已支持入口使用 reference。
`attention` 不包含 GDN；GDN 由 `gdn` 组独立控制。

```bash
unset VLLM_FL_CONFIG VLLM_FL_PER_OP
unset VLLM_FL_REFERENCE_INCLUDE VLLM_FL_REFERENCE_EXCLUDE

export VLLM_PLUGINS=fl
export USE_FLAGGEMS=0
export VLLM_FL_REFERENCE_MODE=1
export VLLM_FL_STRICT=1
export VLLM_FL_PREFER=vendor
export VLLM_FL_REFERENCE_EXCLUDE=attention
export VLLM_FL_REFERENCE_REPORT_DIR="/tmp/fl-reference-$(date +%Y%m%d-%H%M%S)"
```

然后启动服务。例如下面保留 Qwen3.6 长输出评测使用的上下文长度和调度参数：

```bash
export CUDA_VISIBLE_DEVICES=0

vllm serve /data/models/Qwen3.6-35B-A3B \
  --served-model-name Qwen3.6-35B-A3B \
  --attention-backend TRITON_ATTN \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --language-model-only \
  --max-model-len 40960 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 9181
```

模型路径、GPU、端口、并发和容量参数按实际环境调整；这些数值不是所有模型或设备的通用配置。
切换 reference 本身不要求缩短上下文或输出长度。请求的输出 token 上限由评测端另行设置，
`--max-model-len` 约束输入与输出的总长度。

## 2. 按算子或算子组切换

保留上一节的公共设置，只调整 INCLUDE 和 EXCLUDE：

| 目标 | `VLLM_FL_REFERENCE_INCLUDE` | `VLLM_FL_REFERENCE_EXCLUDE` |
| --- | --- | --- |
| 标准 Attention 走原实现，其余已支持入口走 reference | unset | `attention` |
| 只让归一化走 reference | `normalization` | unset |
| 只让 RoPE 走 reference | `rope` | unset |
| 只让 MoE 走 reference | `moe` | unset |
| 只让 GDN 走 reference | `gdn` | unset |
| 同时选择归一化和 RoPE | `normalization,rope` | unset |
| 所有已支持入口走 reference | unset | unset |

例如只让归一化走 reference：

```bash
export VLLM_FL_REFERENCE_INCLUDE=normalization
unset VLLM_FL_REFERENCE_EXCLUDE
# 然后启动服务；Attention 未选中，可以保留 --attention-backend TRITON_ATTN。
```

- INCLUDE 未设置或为 `all` 表示选择全部入口；显式空字符串表示一个都不选。
- EXCLUDE 优先于 INCLUDE。名字拼错会报错。
- 未选中的入口主动使用保存的原平台实现；不会关闭其他入口的严格检查。
- 有效 `VLLM_FL_CONFIG` YAML 会覆盖 policy 环境变量。使用上述环境变量方式时先清除旧配置。
- 只设置 INCLUDE/EXCLUDE 或 `VLLM_FL_PREFER=reference` 不会开启此模式；
  总开关是 `VLLM_FL_REFERENCE_MODE=1`。

所有已支持入口都走 reference 时：

```bash
unset VLLM_FL_REFERENCE_INCLUDE VLLM_FL_REFERENCE_EXCLUDE
# 从 vllm serve 命令中移除 --attention-backend TRITON_ATTN。
```

Attention 被选为 reference 时，不能同时指定优化 Attention 后端。
Attention 的布局在 KV cache 分配前确定；MoE/W8A8 的选择也会影响权重装载。

完整名称、别名、组和选择粒度见 [reference-selection.md](reference-selection.md)。
复合算子中的内联计算不会自动成为可单独切换的入口。

## 3. 切回原版 vLLM 对照

在新进程中显式关闭 FL 插件及 reference：

```bash
export VLLM_PLUGINS=""
export USE_FLAGGEMS=0
export VLLM_FL_REFERENCE_MODE=0
unset VLLM_FL_REFERENCE_INCLUDE VLLM_FL_REFERENCE_EXCLUDE
unset VLLM_FL_REFERENCE_REPORT_DIR

# 然后启动原来的 vllm serve 命令。
```

`VLLM_PLUGINS=""` 要显式设为空。仅 unset 可能让 vLLM 自动加载已安装的插件。
如果保留 `VLLM_PLUGINS=fl`、只设置 `VLLM_FL_REFERENCE_MODE=0`，
则进入插件的普通模式，仍受其 backend policy 和兼容性 hooks 影响。

做原版对照时，可保留 `--enforce-eager`，并对齐模型、dtype、Attention 后端、
上下文、采样及并发设置，以减少其他变量的影响。

## 4. 查看实际执行来源

`VLLM_FL_REFERENCE_REPORT_DIR` 是可选项；未设置时不写 JSONL 文件。
设置后每个 worker 写入独立的 `reference-<PID>.jsonl`，建议每次实验使用新目录。

```bash
tail -n 20 "$VLLM_FL_REFERENCE_REPORT_DIR"/reference-*.jsonl
```

| source | 含义 |
| --- | --- |
| `vllm.native` | 使用 vLLM 的 reference 候选，可能经过参数或布局适配 |
| `plugin.torch` | 使用插件补充的 PyTorch 候选 |
| `reference.setup` | 装载或布局初始化事件，不代表执行某个计算 kernel |
| `user_override` | 按 INCLUDE/EXCLUDE 主动使用原实现 |
| `reference.mixed` | reference 父调用包含显式优化子调用 |
| `optimized_fallback` | 非严格模式下使用优化实现回退 |
| `unavailable` / `error` | 无可用 reference / 实现执行出错 |

`implementation` 字段辅助定位实现，精确的调用细节需结合适配代码。
例如 GemmaRMSNorm 的标签是其 native 方法名，实际由适配器准备权重后调用 vLLM IR native。

## 5. 严格模式和支持边界

`VLLM_FL_STRICT=1` 表示选中的入口没有可用 reference 时明确报错。
只有候选发现或输入支持检查失败可以尝试下一候选；实现开始执行后发生异常，
严格和非严格模式都不会重试，避免重复修改缓存或原地张量。

当前支持范围和已知限制见 [reference-mode-stage2.md](reference-mode-stage2.md)。
未知 CustomOp、未接管的自由函数和任意直接 kernel 调用并不会自动获得 reference 支持。
“开启全部 reference”也不表示整个模型进程的每个调用都已被证明为纯 PyTorch。

测试可在空闲设备上运行：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_PLUGINS=fl USE_FLAGGEMS=0 \
  python3 -m pytest tests/unit_tests/dispatch tests/unit_tests/quantization -q
```

更早的 [stage-one 文档](reference-mode-stage1.md) 用于解释第一阶段实现，
当前使用方式以本说明、选择性 reference 文档和第二阶段支持范围为准。
