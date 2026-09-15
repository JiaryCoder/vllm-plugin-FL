# Copyright (c) 2026 BAAI. All rights reserved.
"""MLA interface compatibility without GPU or vendor package dependencies."""
from abc import ABC, abstractmethod
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


path = Path(__file__).resolve().parents[3] / (
    "vllm_fl/dispatch/backends/vendor/metax/mla_compat.py"
)
spec = importlib.util.spec_from_file_location("metax_mla_compat_under_test", path)
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


class SplitMLA(ABC):
    @abstractmethod
    def forward_mha(self): ...

    @abstractmethod
    def forward_mqa(self): ...


class VendorMLA(SplitMLA):
    def forward_mha(self): return "vendor prefill"

    def forward_mqa(self): return "vendor decode"


class LegacyMLA(ABC):
    @abstractmethod
    def forward(self): ...


class MetaXMLACompatibility(unittest.TestCase):
    def setUp(self):
        self.base = patch.dict(sys.modules, {
            "vllm.v1.attention.backend": SimpleNamespace(MLAAttentionImpl=SplitMLA),
        })
        self.base.start()
        self.addCleanup(self.base.stop)

    def test_current_vllm_reuses_concrete_vendor_backend(self):
        backend = SimpleNamespace(get_impl_cls=lambda: VendorMLA)
        with patch.object(compat, "import_module", return_value=SimpleNamespace(
            MacaFlashMLABackend=backend,
        )) as load:
            self.assertEqual(compat.dense_mla_backend_path(), compat.COMPAT_BACKEND)
            load.assert_called_once_with(compat.VENDOR_BACKEND.rsplit(".", 1)[0])
        self.assertEqual(VendorMLA().forward_mha(), "vendor prefill")
        self.assertEqual(VendorMLA().forward_mqa(), "vendor decode")

    def test_legacy_vllm_does_not_require_external_vendor_package(self):
        with patch.dict(sys.modules, {
            "vllm.v1.attention.backend": SimpleNamespace(MLAAttentionImpl=LegacyMLA),
        }), patch.object(compat, "import_module") as load:
            self.assertEqual(compat.dense_mla_backend_path(), compat.LEGACY_BACKEND)
            load.assert_not_called()

    def test_missing_vendor_does_not_fall_back_to_incompatible_legacy(self):
        with patch.object(compat, "import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(RuntimeError, "matching vendor vllm-metax"):
                compat.dense_mla_backend_path()

    def test_abstract_or_unrelated_vendor_is_rejected(self):
        for impl in (SplitMLA, object, object()):
            with self.subTest(impl=impl), patch.object(
                compat, "import_module", return_value=SimpleNamespace(
                    MacaFlashMLABackend=SimpleNamespace(get_impl_cls=lambda: impl),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "incompatible"):
                    compat.dense_mla_backend_path()

    def test_prefill_registration_is_limited_to_current_vendor_mla(self):
        from unittest.mock import Mock
        register = Mock()
        registry = SimpleNamespace(
            MLAPrefillBackendEnum=SimpleNamespace(FLASH_ATTN="flash"),
            register_mla_prefill_backend=register,
        )
        with patch.object(compat, "import_module", return_value=registry) as load:
            compat.register_mla_prefill(compat.LEGACY_BACKEND)
            load.assert_not_called()
            compat.register_mla_prefill(compat.VENDOR_BACKEND)
            register.assert_called_once_with("flash", class_path=compat.PREFILL_BACKEND)

    def test_missing_prefill_registry_is_distinct_from_broken_dependency(self):
        missing = ModuleNotFoundError(name="vllm.v1.attention.backends.mla.prefill")
        with patch.object(compat, "import_module", side_effect=missing):
            compat.register_mla_prefill(compat.VENDOR_BACKEND)
        broken = ModuleNotFoundError(name="broken_dependency")
        with patch.object(compat, "import_module", side_effect=broken):
            with self.assertRaises(ModuleNotFoundError):
                compat.register_mla_prefill(compat.VENDOR_BACKEND)

    def test_optional_output_scale_bridge_preserves_vendor_calls(self):
        bridge_path = path.parent / "impl/attention/mla/vendor_flashmla.py"
        calls = []

        class OldVendor:
            def forward_mha(self, value, output):
                calls.append((value, output))
                return output

        class NewVendor:
            def forward_mha(self, value, output, output_scale=None):
                calls.append((value, output, output_scale))
                return output

        for cls in (OldVendor, NewVendor):
            with self.subTest(cls=cls), patch.dict(sys.modules, {
                compat.VENDOR_BACKEND.rsplit(".", 1)[0]: SimpleNamespace(
                    FlashMLAImpl=cls, MacaFlashMLABackend=object,
                ),
            }):
                entry = importlib.util.spec_from_file_location("metax_mla_bridge_test", bridge_path)
                bridge = importlib.util.module_from_spec(entry)
                entry.loader.exec_module(bridge)
                obj = bridge.MacaFlashMLABackend.get_impl_cls()()
                output, scale = object(), object()
                self.assertIs(obj.forward_mha("q", output=output, output_scale=None), output)
                if cls is OldVendor:
                    count = len(calls)
                    with self.assertRaises(NotImplementedError):
                        obj.forward_mha("q", output=output, output_scale=scale)
                    self.assertEqual(len(calls), count)
                else:
                    obj.forward_mha("q", output=output, output_scale=scale)
                    self.assertIs(calls[-1][-1], scale)


if __name__ == "__main__":
    unittest.main()
