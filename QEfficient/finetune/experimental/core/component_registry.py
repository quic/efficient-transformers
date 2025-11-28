# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import logging
from typing import Callable, Dict, Optional, Type

# from QEfficient.finetune.experimental.core.logger import get_logger

# logger = get_logger()
logger = logging.getLogger(__name__)


def get_object(obj_dict: Dict, name: str, object_type: str, list_fn: Callable) -> Optional[Type]:
    """Utility to get object from a dictionary with error handling."""
    obj = obj_dict.get(name)
    if obj is None:
        raise ValueError(f"Unknown {object_type}: {name}. Available: {list_fn()}")
    return obj


class ComponentRegistry:
    """Registry for managing different training components."""

    def __init__(self):
        self._optimizers: Dict[str, Type] = {}
        self._schedulers: Dict[str, Type] = {}
        self._datasets: Dict[str, Type] = {}
        self._models: Dict[str, Type] = {}
        self._data_collators: Dict[str, Type] = {}
        self._metrics: Dict[str, Type] = {}
        self._loss_functions: Dict[str, Type] = {}
        self._callbacks: Dict[str, Type] = {}
        self._hooks: Dict[str, Type] = {}
        self._trainer_modules: Dict[str, Type] = {}

    def trainer_module(self, name: str, args_cls=None, required_kwargs=None):
        """
        Decorator to register a trainer module with its configuration.
        Each trainer module has to be binded to its args class and required kwargs.

        Args:
            name: Name of the trainer type
            args_cls: The arguments class for this trainer
            required_kwargs: Dictionary of required keyword arguments and their default values
        """
        required_kwargs = required_kwargs or {}

        def decorator(trainer_cls):
            self._trainer_modules[name] = {
                "trainer_cls": trainer_cls,
                "args_cls": args_cls,
                "required_kwargs": required_kwargs,
            }
            logger.info(f"Registered trainer module: {name}")
            return self._trainer_modules[name]

        return decorator

    def optimizer(self, name: str):
        """Decorator to register an optimizer class."""

        def decorator(cls: Type):
            self._optimizers[name] = cls
            logger.info(f"Registered optimizer: {name}")
            return cls

        return decorator

    def scheduler(self, name: str):
        """Decorator to register a scheduler class."""

        def decorator(cls: Type):
            self._schedulers[name] = cls
            logger.info(f"Registered scheduler: {name}")
            return cls

        return decorator

    def dataset(self, name: str):
        """Decorator to register a dataset class."""

        def decorator(cls: Type):
            self._datasets[name] = cls
            logger.info(f"Registered dataset: {name}")
            return cls

        return decorator

    def model(self, name: str):
        """Decorator to register a model class."""

        def decorator(cls: Type):
            self._models[name] = cls
            logger.info(f"Registered model: {name}")
            return cls

        return decorator

    def data_collator(self, name: str):
        """Decorator to register a data collator class."""

        def decorator(fn_pointer: Type):
            self._data_collators[name] = fn_pointer
            logger.info(f"Registered data collator: {name}")
            return fn_pointer

        return decorator

    def loss_function(self, name: str):
        """Decorator to register a loss function class."""

        def decorator(cls: Type):
            self._loss_functions[name] = cls
            logger.info(f"Registered loss function: {name}")
            return cls

        return decorator

    def callback(self, name: str):
        """Decorator to register a callback class."""

        def decorator(cls: Type):
            self._callbacks[name] = cls
            logger.info(f"Registered callback: {name}")
            return cls

        return decorator

    def get_trainer_module(self, name: str) -> Optional[Type]:
        """Get trainer module class by name."""
        return get_object(self._trainer_modules, name, "trainer module", self.list_trainer_modules)

    def get_optimizer(self, name: str) -> Optional[Type]:
        """Get optimizer class by name."""
        return get_object(self._optimizers, name, "optimizer", self.list_optimizers)

    def get_scheduler(self, name: str) -> Optional[Type]:
        """Get scheduler class by name."""
        return get_object(self._schedulers, name, "scheduler", self.list_schedulers)

    def get_dataset(self, name: str) -> Optional[Type]:
        """Get dataset class by name."""
        return get_object(self._datasets, name, "dataset", self.list_datasets)

    def get_model(self, name: str) -> Optional[Type]:
        """Get model class by name."""
        return get_object(self._models, name, "model", self.list_models)

    def get_data_collator(self, name: str) -> Optional[Type]:
        """Get data collator class by name."""
        return get_object(self._data_collators, name, "data collator", self.list_data_collators)

    def get_loss_function(self, name: str) -> Optional[Type]:
        """Get loss function class by name."""
        return get_object(self._loss_functions, name, "loss function", self.list_loss_functions)

    def get_callback(self, name: str) -> Optional[Type]:
        """Get callback class by name."""
        return get_object(self._callbacks, name, "callback", self.list_callbacks)

    def list_trainer_modules(self) -> list[str]:
        """List all registered trainer modules."""
        return list(self._trainer_modules.keys())

    def list_optimizers(self) -> list[str]:
        """List all registered optimizers."""
        return list(self._optimizers.keys())

    def list_schedulers(self) -> list[str]:
        """List all registered schedulers."""
        return list(self._schedulers.keys())

    def list_datasets(self) -> list[str]:
        """List all registered datasets."""
        return list(self._datasets.keys())

    def list_models(self) -> list[str]:
        """List all registered models."""
        return list(self._models.keys())

    def list_data_collators(self) -> list[str]:
        """List all registered data collators."""
        return list(self._data_collators.keys())

    def list_loss_functions(self) -> list[str]:
        """List all registered loss functions."""
        return list(self._loss_functions.keys())

    def list_callbacks(self) -> list[str]:
        """List all registered callbacks."""
        return list(self._callbacks.keys())


# Global registry instance
registry = ComponentRegistry()
