# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import pytest

from QEfficient.finetune.experimental.core.component_registry import ComponentRegistry, get_object, registry


class TestComponentRegistry:
    @pytest.fixture(autouse=True)
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.registry = ComponentRegistry()

    @pytest.mark.parametrize(
        "register_method, get_method, object_name",
        [
            ("trainer_module", "get_trainer_module", "test_trainer"),
            ("optimizer", "get_optimizer", "test_optimizer"),
            ("scheduler", "get_scheduler", "test_scheduler"),
            ("dataset", "get_dataset", "test_dataset"),
            ("model", "get_model", "test_model"),
            ("data_collator", "get_data_collator", "test_collator"),
            ("loss_function", "get_loss_function", "test_loss"),
            ("callback", "get_callback", "test_callback"),
        ],
    )
    def test_object_success(self, register_method: str, get_method: str, object_name: str):
        """Test object registration decorator."""

        class MockObject:
            pass

        # Register with decorator
        getattr(self.registry, register_method)(object_name)(MockObject)

        # Verify registration
        retrieved = getattr(self.registry, get_method)(object_name)
        if register_method == "trainer_module":
            retrieved = retrieved["trainer_cls"]
        assert retrieved == MockObject

    @pytest.mark.parametrize(
        "object_type, get_method",
        [
            ("trainer module", "get_trainer_module"),
            ("optimizer", "get_optimizer"),
            ("scheduler", "get_scheduler"),
            ("dataset", "get_dataset"),
            ("model", "get_model"),
            ("data collator", "get_data_collator"),
            ("loss function", "get_loss_function"),
            ("callback", "get_callback"),
        ],
    )
    def test_object_failure(self, object_type: str, get_method: str, object_name: str = "non_existent"):
        """Test failure when retrieving non-existent object."""
        with pytest.raises(ValueError) as exc_info:
            getattr(self.registry, get_method)(object_name)

        assert f"Unknown {object_type}" in str(exc_info.value)

    def test_init_empty_registries(self):
        """Test that all registries are initialized as empty dictionaries."""
        assert len(self.registry._optimizers) == 0
        assert len(self.registry._schedulers) == 0
        assert len(self.registry._datasets) == 0
        assert len(self.registry._models) == 0
        assert len(self.registry._data_collators) == 0
        assert len(self.registry._metrics) == 0
        assert len(self.registry._loss_functions) == 0
        assert len(self.registry._callbacks) == 0
        assert len(self.registry._hooks) == 0
        assert len(self.registry._trainer_modules) == 0

    def test_trainer_module_with_args_and_kwargs(self):
        """Test trainer module registration with args class and required kwargs."""

        class MockArgs:
            pass

        class MockTrainer:
            pass

        # Register with decorator including args class and required kwargs
        self.registry.trainer_module(
            "test_trainer_with_args", args_cls=MockArgs, required_kwargs={"param1": "default1", "param2": "default2"}
        )(MockTrainer)

        # Verify registration details
        module_info = self.registry.get_trainer_module("test_trainer_with_args")
        assert module_info["trainer_cls"] == MockTrainer
        assert module_info["args_cls"] == MockArgs
        assert module_info["required_kwargs"] == {"param1": "default1", "param2": "default2"}

    def test_list_methods(self):
        """Test all list methods return correct keys."""

        # Register some dummy items
        class DummyClass:
            pass

        self.registry.optimizer("opt1")(DummyClass)
        self.registry.scheduler("sched1")(DummyClass)
        self.registry.dataset("ds1")(DummyClass)
        self.registry.model("model1")(DummyClass)
        self.registry.data_collator("coll1")(lambda x: x)
        self.registry.loss_function("loss1")(DummyClass)
        self.registry.callback("cb1")(DummyClass)
        self.registry.trainer_module("tm1")(DummyClass)

        # Test lists
        assert self.registry.list_optimizers() == ["opt1"]
        assert self.registry.list_schedulers() == ["sched1"]
        assert self.registry.list_datasets() == ["ds1"]
        assert self.registry.list_models() == ["model1"]
        assert self.registry.list_data_collators() == ["coll1"]
        assert self.registry.list_loss_functions() == ["loss1"]
        assert self.registry.list_callbacks() == ["cb1"]
        assert self.registry.list_trainer_modules() == ["tm1"]

    def test_logging_on_registration(self, mocker):
        """Test that registration logs messages."""
        mock_logger = mocker.patch("QEfficient.finetune.experimental.core.component_registry.logger")

        class MockClass:
            pass

        # Test optimizer registration logging
        self.registry.optimizer("test_opt")(MockClass)
        mock_logger.info.assert_called_with("Registered optimizer: test_opt")

        # Reset mock
        mock_logger.reset_mock()

        # Test trainer module registration logging
        self.registry.trainer_module("test_tm")(MockClass)
        mock_logger.info.assert_called_with("Registered trainer module: test_tm")


class TestGetObjectFunction:
    def test_get_object_success(self):
        """Test get_object function success case."""
        test_dict = {"key1": "value1", "key2": "value2"}

        result = get_object(test_dict, "key1", "test_type", lambda: ["key1", "key2"])
        assert result == "value1"

    def test_get_object_failure(self):
        """Test get_object function failure case."""
        test_dict = {"key1": "value1"}

        with pytest.raises(ValueError) as exc_info:
            get_object(test_dict, "nonexistent", "test_type", lambda: ["key1", "key2"])

        assert "Unknown test_type: nonexistent" in str(exc_info.value)
        assert "Available: ['key1', 'key2']" in str(exc_info.value)


class TestGlobalRegistry:
    def test_global_registry_instance(self):
        """Test that global registry instance exists and is of correct type."""
        assert isinstance(registry, ComponentRegistry)
