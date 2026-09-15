# MetaX MLA backend compatibility

The bundled MetaX FlashMLA implementation uses an older single `forward`
interface. vLLM 0.24.0 requires separate `forward_mha` and `forward_mqa`
methods, with cache updates and decode projections handled by the attention
layer. Selecting the old implementation fails while constructing DeepSeek:

```text
TypeError: Can't instantiate abstract class FlashMLAImpl without an
implementation for abstract methods 'forward_mha', 'forward_mqa'
```

The dense MLA vendor selector now checks the installed vLLM interface. When
the split methods are required, it reuses
`vllm_metax.v1.attention.backends.mla.flashmla.MacaFlashMLABackend` from the
vendor package. The implementation must be a concrete subclass of the current
`MLAAttentionImpl`. Missing or incompatible vendor packages produce an explicit
error; the selector does not fall back to the known-incompatible implementation.
Older vLLM interfaces keep the bundled backend.

A thin subclass bridges the optional `output_scale` argument that vLLM passes
even for unquantized output. If the vendor method does not accept it, only None
is omitted; a non-None scale is rejected. Vendor versions accepting the argument
receive it unchanged. No attention arithmetic is changed by this bridge.

New vLLM also selects a separate MLA prefill backend. On this OOT platform,
its stock FlashAttention availability check cannot find a CUDA/ROCm backend.
The selector therefore registers `MacaFlashAttnPrefillBackend` through vLLM's
prefill registry. This adapter inherits vLLM's prefill methods and initializes
them with the installed MetaX `flash_attn_varlen_func`, including its
`return_attn_probs` calling convention and value-head padding. It does not
advertise fused output quantization or alter NVIDIA's FlashAttention globals.

This change selects existing vendor attention code. It does not implement a
Torch MLA reference or change the reference math for normalization, RoPE, or
MoE. With `VLLM_FL_REFERENCE_EXCLUDE=attention`, the selected MLA backend is
still an intentional optimized vendor path. Standard attention and sparse MLA
selection are unchanged.

Use a compatible vendor vLLM / vllm-metax installation. For the DeepSeek
comparison, keep BF16, eager, `--block-size 64`, `reference_exclude=attention`,
and omit an explicit `--attention-backend` in selective reference mode.

The dependency-free interface regression can be run with:

```bash
python3 tests/unit_tests/dispatch/test_metax_mla_compat.py
```

## Validation on MetaX C550

- Seven dependency-free compatibility tests cover old/new interfaces, missing
  vendor packages, abstract implementations, prefill registration, and the
  optional output-scale bridge.
- FP16 and BF16 ragged causal prefill and unmasked context attention match an
  independent Torch implementation (atol/rtol 0.02), including unequal QK/V head
  dimensions 192/128 and the return-LSE path.
- DeepSeek-V2-Lite-Chat starts with TP2, BF16, eager, block size 64, and strict
  reference excluding attention. Short arithmetic, an equation with 117 generated
  tokens, and a 7005-token prompt spanning chunked prefill all finish normally.
- The validation image uses vLLM 0.24.0+empty, Torch 2.8.0+metax3.7.0.7, and
  Triton 3.0.0+metax3.7.0.7. Its base packages are unchanged.

These checks validate the integration and selected numerical cases; they are
not a full benchmark accuracy comparison.
