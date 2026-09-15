# Copyright (c) 2026 BAAI. All rights reserved.
"""MetaX FlashAttention MLA prefill against independent Torch attention."""
import os
from types import SimpleNamespace
import unittest


@unittest.skipUnless(os.environ.get("GEMS_VENDOR") == "metax", "requires MetaX")
class MetaXMLAPrefill(unittest.TestCase):
    def test_ragged_causal_prefill_and_unmasked_context(self):
        import torch
        from vllm_fl.dispatch.backends.vendor.metax.impl.attention.mla.prefill import (
            MacaFlashAttnPrefillBackend,
        )

        torch.manual_seed(42)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                impl = MacaFlashAttnPrefillBackend(
                    num_heads=2, scale=192 ** -0.5, kv_lora_rank=512,
                    qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
                    vllm_config=None,
                )
                cu = torch.tensor([0, 3, 8], device="cuda", dtype=torch.int32)
                meta = SimpleNamespace(
                    query_start_loc=cu, max_query_len=5,
                    chunked_context=SimpleNamespace(cu_seq_lens=[cu], max_seq_lens=[5]),
                )
                impl.prepare_metadata(meta)
                q, k = [torch.randn(8, 2, 192, device="cuda", dtype=dtype)
                        for _ in range(2)]
                v = torch.randn(8, 2, 128, device="cuda", dtype=dtype)
                for causal in (True, False):
                    if causal:
                        actual, lse = impl.run_prefill_new_tokens(q, k, v, True)
                        plain = impl.run_prefill_new_tokens(q, k, v, False)
                        torch.testing.assert_close(plain, actual)
                    else:
                        actual, lse = impl.run_prefill_context_chunk(0, q, k, v)
                    self.assertIsNotNone(lse)
                    expected = torch.empty_like(v)
                    for begin, end in ((0, 3), (3, 8)):
                        score = torch.einsum(
                            "thd,shd->hts", q[begin:end].float(), k[begin:end].float(),
                        ) * impl.scale
                        if causal:
                            mask = torch.ones(end-begin, end-begin, device="cuda",
                                              dtype=torch.bool).triu(1)
                            score.masked_fill_(mask, float("-inf"))
                        expected[begin:end] = torch.einsum(
                            "hts,shv->thv", score.softmax(-1), v[begin:end].float(),
                        ).to(dtype)
                    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
                self.assertFalse(impl.supports_quant_output(None))


if __name__ == "__main__":
    unittest.main()
