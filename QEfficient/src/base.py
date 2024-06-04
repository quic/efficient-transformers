from abc import ABC, abstractmethod
from typing import Any


class QEFFBaseModel(ABC):
    """
    This class acts as parent class for all the varieties of model class (i.e. LLMs, SD, quantized etc.).
    Enforces certain methods to be implemented by child classes.

    All the child classes must provide way to load, transform(optimize), exoprt to ONNX etc. capabilities.
    """
    def __init__(self) -> None:
        super().__init__()
        # Users can call generate or execute
        self.generate = self.execute

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        raise NotImplementedError("Must implement for child classes")

    @property
    def is_transformed(self) -> bool:
        raise NotImplementedError("Must implement for child classes")

    @abstractmethod
    def transform_export(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def transform_export_compile(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def transform(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def export(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def compile(self, *args, **kwargs) -> Any:
        pass