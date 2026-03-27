"""Tests for trainer runtime compatibility workarounds."""

import dataclasses
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            get_device_name=lambda _index: "stub",
        )
    ),
)

from staged_rl.trainer_runtime import (
    _configure_generation_cache_behavior,
    _count_trainable_parameters,
    _has_active_peft_adapters,
    _install_trl_prepare_peft_workaround,
    _warm_start_peft_adapter,
)


@dataclasses.dataclass
class _FakeArgs:
    gradient_checkpointing: bool = True
    generation_batch_size: int | None = 8
    steps_per_generation: int | None = 4


class _FakeDataclassesModule:
    @staticmethod
    def replace(obj, /, **changes):
        if (
            changes == {"gradient_checkpointing": False}
            and getattr(obj, "generation_batch_size", None) is not None
            and getattr(obj, "steps_per_generation", None) is not None
        ):
            raise ValueError("'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time")
        return dataclasses.replace(obj, **changes)


class TrainerRuntimePatchTests(unittest.TestCase):
    def test_prepare_peft_workaround_mutates_in_place_for_grpo_conflict(self):
        fake_module = types.SimpleNamespace(dataclasses=_FakeDataclassesModule())
        changed = _install_trl_prepare_peft_workaround(fake_module)
        self.assertTrue(changed)

        args = _FakeArgs()
        result = fake_module.dataclasses.replace(args, gradient_checkpointing=False)

        self.assertIs(result, args)
        self.assertFalse(args.gradient_checkpointing)

    def test_prepare_peft_workaround_delegates_for_non_conflicting_replace(self):
        fake_module = types.SimpleNamespace(dataclasses=_FakeDataclassesModule())
        _install_trl_prepare_peft_workaround(fake_module)

        args = _FakeArgs(generation_batch_size=None, steps_per_generation=4)
        result = fake_module.dataclasses.replace(args, gradient_checkpointing=False)

        self.assertIsNot(result, args)
        self.assertFalse(result.gradient_checkpointing)
        self.assertIsNone(result.generation_batch_size)
        self.assertEqual(result.steps_per_generation, 4)

    def test_has_active_peft_adapters_requires_non_empty_config(self):
        self.assertFalse(_has_active_peft_adapters(types.SimpleNamespace(peft_config=None)))
        self.assertFalse(_has_active_peft_adapters(types.SimpleNamespace(peft_config={})))
        self.assertTrue(_has_active_peft_adapters(types.SimpleNamespace(peft_config={"default": object()})))

    def test_count_trainable_parameters(self):
        class _FakeParam:
            def __init__(self, n, requires_grad):
                self._n = n
                self.requires_grad = requires_grad

            def numel(self):
                return self._n

        fake_model = types.SimpleNamespace(
            parameters=lambda: [_FakeParam(5, True), _FakeParam(7, False), _FakeParam(3, True)]
        )
        trainable, total = _count_trainable_parameters(fake_model)
        self.assertEqual(trainable, 8)
        self.assertEqual(total, 15)

    def test_configure_generation_cache_behavior_clears_static_cache(self):
        generation_config = types.SimpleNamespace(cache_implementation="static", use_cache=False)
        config = types.SimpleNamespace(cache_implementation="static", use_cache=False)
        nested = types.SimpleNamespace(generation_config=generation_config, config=config)
        fake_model = types.SimpleNamespace(
            generation_config=types.SimpleNamespace(cache_implementation="static", use_cache=False),
            config=types.SimpleNamespace(cache_implementation="static", use_cache=False),
            base_model=nested,
        )

        result = _configure_generation_cache_behavior(fake_model)

        self.assertIsNone(fake_model.generation_config.cache_implementation)
        self.assertTrue(fake_model.generation_config.use_cache)
        self.assertIsNone(fake_model.config.cache_implementation)
        self.assertTrue(fake_model.config.use_cache)
        self.assertIsNone(nested.generation_config.cache_implementation)
        self.assertTrue(nested.generation_config.use_cache)
        self.assertIn("SimpleNamespace", result["patched_wrappers"])
        self.assertIsNone(result["cache_implementation"])
        self.assertTrue(result["use_cache"])

    def test_warm_start_peft_adapter_uses_peft_state_dict_loader(self):
        save_and_load_module = types.ModuleType("peft.utils.save_and_load")
        observed = {}

        def load_peft_weights(path, device="cpu"):
            observed["load_peft_weights"] = {"path": path, "device": device}
            return {"adapter": "weights"}

        def set_peft_model_state_dict(model, adapter_state, adapter_name="default", ignore_mismatched_sizes=False):
            observed["set_peft_model_state_dict"] = {
                "model": model,
                "adapter_state": adapter_state,
                "adapter_name": adapter_name,
                "ignore_mismatched_sizes": ignore_mismatched_sizes,
            }
            return {"missing_keys": [], "unexpected_keys": []}

        save_and_load_module.load_peft_weights = load_peft_weights
        save_and_load_module.set_peft_model_state_dict = set_peft_model_state_dict
        sys.modules["peft"] = types.ModuleType("peft")
        sys.modules["peft.utils"] = types.ModuleType("peft.utils")
        sys.modules["peft.utils.save_and_load"] = save_and_load_module

        class _FakeModel:
            def __init__(self):
                self.selected_adapter = None

            def set_adapter(self, adapter_name):
                self.selected_adapter = adapter_name

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_dir = Path(tmpdir)
                model = _FakeModel()
                _warm_start_peft_adapter(model, str(checkpoint_dir))
        finally:
            sys.modules.pop("peft.utils.save_and_load", None)
            sys.modules.pop("peft.utils", None)
            sys.modules.pop("peft", None)

        self.assertEqual(observed["load_peft_weights"]["path"], str(checkpoint_dir))
        self.assertEqual(observed["load_peft_weights"]["device"], "cpu")
        self.assertEqual(observed["set_peft_model_state_dict"]["adapter_name"], "default")
        self.assertFalse(observed["set_peft_model_state_dict"]["ignore_mismatched_sizes"])
        self.assertEqual(model.selected_adapter, "default")


if __name__ == "__main__":
    unittest.main()
