# Stage-one torch reference mode

This document describes the first implementation stage. For current usage,
see the [Reference mode guide](reference-quickstart.md) and
[stage-two coverage](reference-mode-stage2.md).

This opt-in mode routes audited composite operations to existing vLLM torch code,
then to audited plugin torch code. It is independent of `VLLM_FL_PREFER`, which
retains its original backend-preference semantics.

The validated source target is vLLM **0.24.0**, PyTorch **2.11.0**, with the FL
plugin based on commit **13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0**.
Validation includes CPU and NVIDIA H20, FP32/FP16/BF16. Other vendor platforms
need their own numerical validation before being declared supported.

## Start a new reference process

```bash
export VLLM_FL_REFERENCE_MODE=1
export VLLM_FL_STRICT=1
export VLLM_FL_REFERENCE_REPORT_DIR=/tmp/fl-reference-report
# Then run your normal vLLM entry point.
```

The worker disables model compilation and CUDA graphs and skips
`flag_gems.enable()`. The IR hooks select audited native torch implementations,
including when callers use an IR `maybe_inplace` overload.
Already-active FlagGems ATen overrides cause an explicit error: use a fresh
worker. Changing the mode after a model has been constructed is unsupported.

The mode flag is environment-only in this first version. Existing
`SelectionPolicy`, YAML `strict`, `VLLM_FL_STRICT`, vendor filters and
per-op backend orders still control strictness and permitted optimized fallbacks.
A valid `VLLM_FL_CONFIG` file overrides the policy environment variables.
For example, a non-strict policy file can contain:

```yaml
prefer: vendor
strict: false
op_backends:
  moe_sum: [vendor:cuda, flagos]
```

Reference mode always tries audited vLLM then plugin candidates first. The
listed non-reference implementations are fallback choices, not reference code.
An explicit order containing only `reference` permits no optimized fallback.

Setting `VLLM_FL_PREFER=reference` alone does **not** enable this mode. It keeps
the previous preference-based behavior, including platform per-op overrides.
`USE_FLAGGEMS=0` is optional here: the reference worker already skips ATen
replacement. That variable additionally removes FlagGems composite candidates
from the ordinary dispatch registry, so leave it unset if you want FlagGems
composites available as non-strict fallbacks.

## Frozen stage-one inventory

| Interface | Operation | Preferred reference | Plugin fallback |
| --- | --- | --- | --- |
| FL dispatch | silu_and_mul | SiluAndMul.forward_native | torch SiLU and multiply |
| FL dispatch | gelu_and_mul | GeluAndMul.forward_native | torch GELU and multiply |
| FL dispatch | rms_norm, including residual | IR native RMSNorm/Add RMSNorm | FP32 torch normalization |
| FL dispatch | rotary_embedding | ApplyRotaryEmb.forward_static with layout adapter | functional torch RoPE |
| FL dispatch | dynamic_per_token_quant_int8 | None audited: upstream helper uses Triton | existing W8A8 torch quantization |
| CustomOp | SiluAndMul, GeluAndMul, RMSNorm | audited upstream implementations | above plugin implementations |
| CustomOp | RotaryEmbedding | upstream native/static chain | normalized plugin RoPE with interface adapter |
| CustomOp | ApplyRotaryEmb | upstream native/static chain | none |
| CustomOp | MRotaryEmbedding | upstream native, no FlagGems kernel required | none |
| IR | rms_norm, fused_add_rms_norm | native provider directly | plugin normalization |

The four existing FL layer classes are aliases of their audited upstream
counterparts. Unknown subclasses are not certified by inheritance or by sharing
the registered name `rotary_embedding`.

Covered variations include residual and weightless RMSNorm, variance-size
override, ordinary/partial RoPE, optional key for ordinary RoPE, neox/interleaved
rotation, rank-3 and rank-4 normalized dispatch layouts, MRoPE text positions and
three-axis multimodal positions, and MRoPE's `mrope_interleaved` parameter.
MRoPE requires a key in the upstream implementation. YaRN-scaled MRoPE and the
separate `MRotaryEmbeddingInterleaved`/Ernie classes remain unaudited.
Numerical tests cover FP32, FP16 and BF16; unsupported floating dtypes are
reported as unavailable instead of silently counted as verified references.

The plugin RMSNorm fallback now performs residual addition and normalization
intermediates in FP32, preserving upstream cast ordering. This fixes FP16
squaring overflow. The plugin RoPE fallback fixes compact interleaved cache
expansion and rank-4 broadcasting; both fallbacks are functional.

## Strictness and boundaries

1. Discover a candidate and check availability/input support before execution.
2. Try the vLLM candidate, then the plugin candidate.
3. If neither is usable: strict mode raises `ReferenceUnavailable`; non-strict
   mode uses the next permitted ordinary backend and records its actual source.
4. Once execution begins, any exception propagates in both modes. Retrying after
   in-place mutation is unsafe and can hide the original precision problem.

The runtime rejects opaque non-ATen torch custom operations reached from an
audited reference. It also uses an explicit audited class/operator inventory;
the runtime guard alone cannot detect every direct Triton/C++ launch in
arbitrary Python code. A function name or a `reference.torch` label is not a
certificate of a pure torch path.

The standard CustomOp constructor hook also catches classes overriding
`forward`; an unknown variant is rejected rather than executing an unrelated
inherited native implementation. For non-strict CustomOps outside FL dispatch,
fallback means the original platform/forward route saved at construction.
IR fallbacks preserve the normal provider-selection behavior.

**This is not yet a full-model pure-torch mode.** Attention, MoE, GDN/FLA,
quantized expert/linear computation and arbitrary direct kernel calls are
stage-two work. At intercepted Attention/MoE dispatch entries, strict mode
explicitly rejects the two legacy `reference.torch` wrappers that call optimized
backends. Calls completely outside these hooks are not automatically audited.
Full-model coverage requires later execution-path inventory and validation.

## Evidence and integration

The common engine is in `vllm_fl/reference/engine.py`. Audited candidates and
calling-convention adapters are in `adapters.py`; the constructor/IR hooks are
in `hooks.py`; FL manager integration is in `dispatch.py`.

Hooks are installed before model construction by platform configuration and
worker initialization. Installation is idempotent. Spawned workers install
their own hooks; report files are separated by PID. Plain Python unit
experiments can explicitly call:

```python
from vllm_fl.reference.hooks import configure_reference
configure_reference(vllm_config)
```

`get_records()` returns actual successful reference paths, unavailable paths,
errors and optimized fallbacks with call counts. Optional JSONL reports contain
metadata only, never tensor values. Report directories should be unique per
experiment. Source labels are `vllm.native`, `plugin.torch`,
`optimized_fallback`, `reference_unavailable`, `unavailable` and `error`.

## Reproduce validation without model weights

```bash
python3 tools/reference_stage1_smoke.py --device cuda --dtype bfloat16 \
  --output /tmp/reference-stage1-smoke.json

python3 -m pytest tests/unit_tests/dispatch -q --tb=short
```

The smoke program enables the mode only in its own process. It exercises the
five existing dispatch ops plus generic CustomOp/IR/MRoPE routes and checks the
reported sources. The test suite includes upstream-first selection, plugin
fallback, strict rejection, non-strict ordering, failure without retry, active
ATen-override rejection, CPU/GPU numerical cases and two spawned workers.
No model benchmark score is inferred from these operator-level tests.
