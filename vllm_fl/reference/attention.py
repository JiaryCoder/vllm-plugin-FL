# Copyright (c) 2026 BAAI. All rights reserved.
"""Eager ATen attention with vLLM paged-KV semantics."""
from dataclasses import dataclass
import functools

import torch

from vllm.v1.attention.backend import (
    AttentionBackend, AttentionCGSupport, AttentionImpl, AttentionMetadataBuilder,
    AttentionType, CommonAttentionMetadata,
)

from .engine import (
    Candidate, ReferenceUnavailable, reference_enabled, run_reference, tensor_support,
    record_route, run_user_override,
)
from .selection import selection_reason


@dataclass
class TorchAttentionMetadata:
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool | torch.Tensor = True


class TorchAttentionMetadataBuilder(AttentionMetadataBuilder[TorchAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.NEVER
    supports_update_block_table = False

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        common = common_attn_metadata
        if common.dcp_local_seq_lens is not None or common.mm_req_doc_ranges:
            raise ReferenceUnavailable("context-parallel and PrefixLM metadata are unaudited")
        return TorchAttentionMetadata(
            common.num_actual_tokens, common.query_start_loc, common.seq_lens,
            common.block_table_tensor, common.slot_mapping, common.causal,
        )


class TorchAttentionBackend(AttentionBackend):
    supported_dtypes = [torch.float32, torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "float16", "bfloat16", "float32"]
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name():
        # vLLM requires a member of AttentionBackendEnum.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return TorchAttentionImpl

    @staticmethod
    def get_builder_cls():
        return TorchAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size,
                           cache_dtype_str="auto"):
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        return (1, 0, 2, 3, 4, 5) if include_num_layers_dimension else (0, 1, 2, 3, 4)

    @classmethod
    def supports_non_causal(cls):
        return True

    @classmethod
    def supports_kv_connector(cls):
        return False


class TorchAttentionImpl(AttentionImpl[TorchAttentionMetadata]):
    def __init__(self, num_heads, head_size, scale, num_kv_heads=None,
                 alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
                 logits_soft_cap=None, attn_type=AttentionType.DECODER,
                 kv_sharing_target_layer_name=None, **kwargs):
        if attn_type != AttentionType.DECODER:
            raise ReferenceUnavailable("stage-two paged reference supports decoder self-attention")
        if kv_cache_dtype not in TorchAttentionBackend.supported_kv_cache_dtypes:
            raise ReferenceUnavailable("quantized KV cache requires a separate reference")
        if self.total_cp_world_size != 1:
            raise ReferenceUnavailable("context-parallel attention is outside this reference")
        if any(v is not None for v in kwargs.values()):
            raise ReferenceUnavailable(f"unaudited attention options: {sorted(kwargs)}")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.scale = scale
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap or 0.
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        if num_heads % self.num_kv_heads:
            raise ValueError("query heads must be a multiple of KV heads")

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None,
                output_scale=None, output_block_scale=None):
        args = (query, key, value, kv_cache, attn_metadata, output)
        def supports(q, k, v, cache, meta, out):
            reason = tensor_support(q, k, v, cache, out)
            if reason:
                return reason
            if output_scale is not None or output_block_scale is not None:
                return "quantized attention output is outside the reference audit"
            if k is not None and (k.shape[-1] != self.head_size or v.shape[-1] != self.head_size):
                return "different key/value head widths require a separate backend"
        return run_reference(
            "attention", args, {},
            (lambda: Candidate("plugin.torch", "TorchAttentionImpl._forward",
                               self._forward, supports),),
        )

    def _forward(self, query, key, value, kv_cache, meta, output):
        if output is None:
            output = torch.empty_like(query)
        if meta is None:
            # vLLM memory-profiling forward: no requests and no cache writes.
            return output.zero_()
        count = meta.num_actual_tokens
        if count == 0:
            return output
        block_size = kv_cache.shape[2]
        # Cache stores the logical [block, K/V, token, head, dim] layout.
        # Advanced indexing preserves arbitrary strides and padded block pages.
        if key is not None and self.kv_sharing_target_layer_name is None:
            slots = meta.slot_mapping[:count].long()
            valid = slots >= 0
            selected = slots[valid]
            if bool((selected >= kv_cache.shape[0] * block_size).any()):
                raise ValueError("KV cache slot is out of bounds")
            blocks, offsets = selected // block_size, selected % block_size
            kv_cache[blocks, 0, offsets] = key[:count][valid].to(kv_cache.dtype)
            kv_cache[blocks, 1, offsets] = value[:count][valid].to(kv_cache.dtype)
        offsets = meta.query_start_loc.tolist()
        lengths = meta.seq_lens.tolist()
        repeats = self.num_heads // self.num_kv_heads
        for seq, (begin, end) in enumerate(zip(offsets, offsets[1:])):
            end = min(end, count)
            if begin >= end:
                continue
            length = int(lengths[seq])
            qlen = end - begin
            if length < qlen:
                raise ValueError("attention sequence length is shorter than its query")
            positions = torch.arange(length, device=query.device)
            block_ids = meta.block_table[seq, positions // block_size].long()
            keys = kv_cache[block_ids, 0, positions % block_size].float()
            values = kv_cache[block_ids, 1, positions % block_size].float()
            keys = keys.repeat_interleave(repeats, dim=1).transpose(0, 1)
            values = values.repeat_interleave(repeats, dim=1).transpose(0, 1)
            causal = bool(meta.causal[seq]) if isinstance(meta.causal, torch.Tensor) else meta.causal
            for chunk in range(0, qlen, 128):
                stop = min(chunk + 128, qlen)
                qpos = torch.arange(length - qlen + chunk, length - qlen + stop,
                                    device=query.device)
                q = query[begin+chunk:begin+stop].float().transpose(0, 1)
                scores = (q @ keys.transpose(-1, -2)) * self.scale
                if self.logits_soft_cap > 0:
                    scores = self.logits_soft_cap * (scores / self.logits_soft_cap).tanh()
                relative = positions[None, :] - qpos[:, None]
                if self.alibi_slopes is not None:
                    slopes = torch.as_tensor(self.alibi_slopes, device=query.device,
                                             dtype=torch.float32)
                    scores -= slopes[:, None, None] * relative.abs()[None]
                allowed = torch.ones_like(relative, dtype=torch.bool)
                if causal:
                    allowed &= relative <= 0
                if self.sliding_window is not None:
                    allowed &= relative >= -(self.sliding_window - 1)
                    if not causal:
                        allowed &= relative <= self.sliding_window - 1
                probs = scores.masked_fill(~allowed[None], -torch.inf).softmax(-1)
                probs = torch.nan_to_num(probs, nan=0.)
                out = (probs @ values).transpose(0, 1).to(output.dtype)
                output[begin+chunk:begin+stop].copy_(out)
        return output


PATH = "vllm_fl.reference.attention.TorchAttentionBackend"


def backend_candidate():
    def supports(use_mla=False, use_sparse=False):
        if use_mla or use_sparse:
            return "MLA and sparse attention are outside the standard-attention reference"
    def select(use_mla=False, use_sparse=False):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
        register_backend(AttentionBackendEnum.CUSTOM, class_path=PATH)
        return PATH
    return Candidate("plugin.torch", PATH, select, supports)


def _validate_optimized_backend(path, config):
    """Use the installed backend's capability checks on every device family."""
    from vllm.platforms import current_platform
    from vllm.utils.import_utils import resolve_obj_by_qualname

    backend = resolve_obj_by_qualname(path)
    if not isinstance(backend, type) or not issubclass(backend, AttentionBackend):
        raise ReferenceUnavailable(f"Not a vLLM AttentionBackend: {path}")
    if issubclass(backend, TorchAttentionBackend):
        raise ReferenceUnavailable("Excluded attention cannot re-enter TorchAttentionBackend")
    invalid = backend.validate_configuration(
        device_capability=current_platform.get_device_capability(), **config._asdict(),
    )
    if invalid:
        raise ValueError(f"Selected backend {path} is not valid for this configuration: {invalid}")
    return path


def _select_optimized_backend(config, fallback, selected_backend, num_heads, reason):
    from vllm.platforms import current_platform
    from vllm_fl.dispatch.policy import get_policy

    policy = get_policy()
    order = policy.get_per_op_order("attention_backend")
    # An explicit dispatch choice is independent of the device family. In
    # particular, Hygon registers vLLM Triton under default.flagos, not vendor.
    if selected_backend is None and (order is not None or not current_platform.is_cuda()):
        path = fallback()
    else:
        vendor = getattr(current_platform, "vendor_name", current_platform.device_name)
        vendor = {"nvidia": "cuda", "mthreads": "musa"}.get(vendor, vendor)
        label = "CUDA" if current_platform.is_cuda() else vendor
        if not policy.is_vendor_allowed(vendor):
            raise ReferenceUnavailable(f"{label} attention is denied by dispatch policy")
        if order is not None and not any(
            token in {"vendor", f"vendor:{vendor}", f"impl:vendor.{vendor}"} for token in order
        ):
            raise ReferenceUnavailable(
                f"attention_backend policy does not permit the {label} vendor; "
                "choose --attention-backend or an FL per-op backend order, not conflicting selectors"
            )
        if current_platform.is_cuda():
            from vllm.platforms.cuda import CudaPlatform
            path = CudaPlatform.get_attn_backend_cls(selected_backend, config, num_heads)
            if not path.startswith("vllm."):
                raise ReferenceUnavailable(f"Expected an upstream vLLM attention backend, got {path}")
        else:
            # Respect vendor registrations in vLLM's backend registry; there is
            # no recursive call to PlatformFL's own backend selector here.
            path = selected_backend.get_path()
    path = _validate_optimized_backend(path, config)
    record_route("attention_backend", "user_override", path, reason)
    return path


def select_backend(config, fallback, selected_backend=None, num_heads=None):
    reason = selection_reason("attention_backend")
    if reason is not None:
        return run_user_override(
            "attention_backend",
            lambda: _select_optimized_backend(config, fallback, selected_backend, num_heads, reason),
            "explicit attention selector", reason,
        )
    if selected_backend is not None and selected_backend.name != "CUSTOM":
        raise ValueError("An explicit optimized attention backend conflicts with reference attention; "
                         "set VLLM_FL_REFERENCE_EXCLUDE=attention to use it")
    def supports():
        if config.use_mla or config.use_sparse:
            return "MLA and sparse attention are outside stage two"
        if config.has_sink or config.use_mm_prefix or config.use_per_head_quant_scales:
            return "sinks, PrefixLM and quantized KV require separate attention references"
        if config.attn_type != AttentionType.DECODER or config.use_kv_connector:
            return "only local decoder self-attention is audited"
        if not TorchAttentionBackend.supports_dtype(config.dtype):
            return f"unaudited attention dtype {config.dtype}"
        if not TorchAttentionBackend.supports_kv_cache_dtype(config.kv_cache_dtype):
            return f"unaudited KV cache dtype {config.kv_cache_dtype}"
        from vllm.config import get_current_vllm_config
        parallel = get_current_vllm_config().parallel_config
        if parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1:
            return "context-parallel attention requires a separate reference"
    return run_reference("attention_backend", (), {}, (
        lambda: Candidate("plugin.torch", PATH, lambda: backend_candidate().fn(), supports),
    ), fallback)


def install():
    from vllm.model_executor.layers.attention import Attention
    if getattr(Attention.__init__, "_fl_reference_attention", False):
        return
    original = Attention.__init__
    @functools.wraps(original)
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if reference_enabled() and isinstance(self.impl, TorchAttentionImpl):
            self.use_direct_call = True
        elif reference_enabled() and selection_reason("attention_backend") is not None:
            selected = self.impl.forward
            reason = selection_reason("attention_backend")
            implementation = type(self.impl).__module__ + "." + type(self.impl).__qualname__ + ".forward"
            @functools.wraps(selected)
            def forward(*a, **k):
                return run_user_override("attention", lambda: selected(*a, **k),
                                         implementation, reason)
            self.impl.forward = forward
    init._fl_reference_attention = True
    Attention.__init__ = init
