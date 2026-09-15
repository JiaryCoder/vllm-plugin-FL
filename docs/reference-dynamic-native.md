# 动态复用 vLLM native 实现

在 `VLLM_FL_REFERENCE_MODE=1` 下，新的 **vLLM 内置 CustomOp** 可以自动发现
`forward_native`，无需为每个类添加支持名单。注册到 vLLM IR 的 native 实现也会动态接入。
已有特殊适配器继续负责参数、形状、布局和数值语义修正。

## 开启

沿用 [Reference 使用说明](reference-quickstart.md) 的开关，没有额外的自动发现开关：

```bash
unset VLLM_FL_CONFIG VLLM_FL_PER_OP VLLM_FL_REFERENCE_INCLUDE
export VLLM_PLUGINS=fl
export USE_FLAGGEMS=0
export VLLM_FL_REFERENCE_MODE=1
export VLLM_FL_STRICT=1
export VLLM_FL_PREFER=vendor
export VLLM_FL_REFERENCE_EXCLUDE=attention
export VLLM_FL_REFERENCE_REPORT_DIR=/tmp/reference-dynamic-run01
# 启动原来的 vllm serve 命令，保留 --attention-backend TRITON_ATTN。
```

要让标准 Attention 也走 reference，unset EXCLUDE 并移除显式优化 Attention 后端参数。
修改开关后须重启服务。所有 reference 模式都保留 vLLM 内置类，不注册 FlagGems OOT 类或 MoE 路由。

## 自动发现的范围

| 情况 | 行为 |
| --- | --- |
| 内置类提供标准 `forward_native` | 构造前选择 native 配置，校验初始化与方法接口后调用 |
| 内置子类继承 native 方法 | 解析实际继承的方法，绑定当前对象 |
| staticmethod / classmethod | 按描述符规则绑定，不重复传 self |
| vLLM IR 有 native provider | 调用实际 native 函数，保留 functional / maybe_inplace 约定 |
| 已有特殊适配器 | 维持已验证的语义和支持条件 |
| 没有 native、空实现或平台包装器 | 使用已有适配器；仍不可用时按 strict 策略处理 |

`Llama3RotaryEmbedding` 的专用适配分支已删除。
它和 Linear/NTK/YaRN RoPE 使用同一套发现机制，复用 `RotaryEmbedding.forward_native`，
保留各自对象的频率缩放和 cos/sin 缓存。

发现按实际出现的类进行，不提前导入所有模型模块。
方法解析按类、实际描述符和代码对象缓存；重新绑定 native 方法会重新检查。
原有类名映射用于兼容选择名称或特殊适配，不再是新 CustomOp 必须加入的支持名单。

## 自动保持初始化与执行一致

用户只需配置 reference 环境变量，不需要填写 `CompilationConfig` 或 `-cc.custom_ops`。
插件在实际 CustomOp 类的完整构造函数开始前，建立独立的 vLLM 配置视图，使用官方
`custom_ops` 开关关闭该类注册名对应的优化实现，并保持 eager。构造函数中的 `enabled()`、
缓存初始化和 forward 选择由同一个配置控制；执行时恢复同一配置视图。

例如 DeepSeek RoPE 会自然得到 `use_flashinfer=False` 和 BF16 sin/cos 缓存，
直接调用未修改的 vLLM `forward_native`。此前针对它的输出 dtype 转换补丁已移除。

配置视图按实际类隔离，不改全局注册名：多个 RoPE 子类共享 `rotary_embedding` 时，
只选择其中一个类不会把另一个被排除的类也初始化为 native。嵌套子算子独立应用自己的选择，
退出构造、正常调用或异常路径都会恢复原配置。vLLM 0.24 的当前配置是进程全局变量，
插件用可重入锁串行化这些配置作用域；该模式用于 eager 调试，不提供跨配置并行模型执行保证。

已注册 OOT 类不会抢占被选中的标准 native 类。NVIDIA 上被排除的算子若落到
`CustomOp.forward_oot` 的通用 native 转发桩，会使用真正的 `forward_cuda`，
避免再次组合“优化初始化 + native 函数”。专有 OOT 实现不作这种替换。

普通 native 接口由通用机制接入；GroupedTopk、完整 MoE、GDN 等已有专用适配继续使用
各自的初始化和计算约定，不将它们包装优化 kernel 的 `forward_native` 当作朴素 Torch 实现。

对于覆盖 `enabled`、强制启用、覆盖构造后 dispatch 等不遵循标准协议的类，
插件不会宣称已完成 native 初始化；需要显式适配，否则严格模式报错。
算子实例创建后改变 reference 选择会报错，必须新建 worker。

非严格模式仍可在没有可用 native 时使用优化初始化及回落实现。
如果对象已经按 native 初始化，之后才遇到不支持的输入，且没有可用的 Torch 适配，
不会复用该对象去冒险调用优化 kernel；会明确要求在启动前排除该算子并新建 worker。
这项限制避免把缓存或权重布局错误伪装成成功回落。任意外部 Python 代码直接改写对象状态，
不在自动配置保证范围内。

## 动态选择

`rope`、`activation`、`normalization`、`moe`、`gdn` 根据模块和继承链识别新类；
`custom` 组选择实际接入的 CustomOp。模型导入前即可配置这些组。

```bash
export VLLM_FL_REFERENCE_INCLUDE=custom:vllm.model_executor.layers.activation.NewGELU
unset VLLM_FL_REFERENCE_EXCLUDE
```

- `custom:<完整 vllm 类名>` 或完整类名：选择对应 CustomOp。
- `ir:<IR 注册名>`：选择对应 IR 入口。
- `all` 包含后来才出现的入口；EXCLUDE 优先于 INCLUDE。
- 完整类名和 IR 选择器先检查语法，不会为了验证名称而提前导入模型。
  应对照实际路由确认目标被调用；名称匹配本身不代表 native 可用。
- 复合入口的内联计算不会被拆成可单独选择的算子；内部显式排除的入口仍走原实现，
  父调用会记录为 `reference.mixed`。

## 路由和执行检查

自动发现的路由使用 `vllm.native.dynamic`：

```text
vllm.model_executor.layers.activation.NewGELU
  -> vllm.model_executor.layers.activation.NewGELU.forward_native (vllm.native.dynamic)
vllm.model_executor.layers.rotary_embedding.llama3_rope.Llama3RotaryEmbedding
  -> vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding.forward_native (vllm.native.dynamic)
```

`vllm.native` 表示原有已审核的上游适配路径，`plugin.torch` 表示插件补充实现。
`reference.setup` 记录 native 初始化及实际使用的 `custom_ops`；它不代表算子已经执行。

执行中继续检查 ATen/prims 调用。只解包已注册 IR 的准确 torch overload 后接回 native，
不会放行整个自定义 torch namespace。Reference 计算中的标准 Triton JIT、autotune、
heuristics 和 compiled-kernel 下标调用会被拦截。PyTorch 公共 SDPA 接口使用 math backend；
直接选择 fused SDPA 的 ATen 入口会报错。普通调用和主动排除的原实现按原路径执行。

候选发现失败可以尝试下一候选；一旦开始执行，异常直接抛出，避免修改输入或缓存后重试。

## 边界

自动发现复用的是 vLLM 的标准接口，不表示完整 Python 调用图已经完成纯度或数值审核：

- 类和 native 定义须来自安装的 vLLM Python 包。第三方 OOT 类、实例替换的 native，
  或覆盖公共 `forward` 改变入口语义的类，仍需要适配。
- 原有适配器的 dtype、形状、量化和缓存限制保留，不通过自动发现绕过。
- 标准入口、ATen 和 Triton 检查不能覆盖所有直接 C/CUDA 调用、预先保存的低层 launcher 或新线程执行。
  `vllm.native.dynamic` 不应被解释为“整个模型进程全为纯 PyTorch”。
- 未接管的自由函数、仅有优化 kernel 的实现、采样器、通信和模型初始化，不会被自动重写。
- native 自身执行失败仍会报错，不能承诺所有 vLLM 可运行的模型都具备完整 reference 路径。

实现位于 `native.py`、`lifecycle.py`、`hooks.py`、`guards.py` 和 `selection.py`。
新增非标准路径时维护的是必要适配器，而非每个正常 native 类的名单。

## 验证

单元测试覆盖真实 vLLM 激活类、四种缩放 RoPE、FP32/FP16/BF16、CPU/CUDA 数值、
动态分组、方法替换、嵌套 IR、SDPA 和 Triton 拦截。

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_PLUGINS=fl USE_FLAGGEMS=0 \
  python3 -m pytest tests/unit_tests/dispatch tests/unit_tests/quantization -q
```

生成本地随机小权重并经过实际 worker 验证，无需下载：

```bash
python3 tools/reference_dynamic_make_models.py --output /tmp/dynamic-native-models

CUDA_VISIBLE_DEVICES=0 VLLM_PLUGINS=fl USE_FLAGGEMS=0 \
  python3 tools/reference_stage2_model_smoke.py \
    --model /tmp/dynamic-native-models/tiny-gpt2-gelu-new \
    --output /tmp/dynamic-gpt2-reference.json --reference --multiprocess
```

生成器还提供 `tiny-llama3-rope` 和 `tiny-llama-yarn`，替换路径即可验证继承方法。
Smoke 工具尊重调用者的 `CUDA_VISIBLE_DEVICES`。这些验证不代表完整权重模型的 benchmark 精度验收。
