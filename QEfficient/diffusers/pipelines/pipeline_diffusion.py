# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from diffusers import DiffusionPipeline
from tqdm import tqdm

from QEfficient.diffusers.pipelines.pipeline_utils import (
    ONNX_SUBFUNCTION_MODULE,
    compile_modules_parallel,
    compile_modules_sequential,
    config_manager,
    set_execute_params,
)


class QEffDiffusionPipeline:
    """
    Generic QEfficient Diffusers pipeline facade.

    This class provides:
    1) an auto-dispatch `from_pretrained` factory that returns concrete child pipelines,
    2) shared export/compile plumbing reused by model-specific child pipelines.
    """

    _HF_PIPELINE_TO_QEFF_CLASS: Dict[str, Tuple[str, str]] = {
        "FluxPipeline": ("QEfficient.diffusers.pipelines.flux.pipeline_flux", "QEffFluxPipeline"),
        "WanPipeline": ("QEfficient.diffusers.pipelines.wan.pipeline_wan", "QEffWanPipeline"),
        "WanImageToVideoPipeline": (
            "QEfficient.diffusers.pipelines.wan.pipeline_wan_i2v",
            "QEffWanImageToVideoPipeline",
        ),
    }

    @staticmethod
    def _filter_kwargs_for_callable(func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        sig = inspect.signature(func)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
            return dict(kwargs)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}

    @classmethod
    def _resolve_qeff_pipeline_class(
        cls, pretrained_model_name_or_path: Optional[Union[str, Any]], **kwargs
    ) -> Type["QEffDiffusionPipeline"]:
        load_cfg_kwargs = cls._filter_kwargs_for_callable(DiffusionPipeline.load_config, kwargs)
        config = DiffusionPipeline.load_config(pretrained_model_name_or_path, **load_cfg_kwargs)
        if isinstance(config, tuple):
            config = config[0]
        hf_pipeline_class_name = config.get("_class_name")
        if not hf_pipeline_class_name:
            raise ValueError("Unable to infer Hugging Face diffusion pipeline class from model config.")

        mapped = cls._HF_PIPELINE_TO_QEFF_CLASS.get(hf_pipeline_class_name)
        if not mapped:
            supported = ", ".join(sorted(cls._HF_PIPELINE_TO_QEFF_CLASS.keys()))
            raise NotImplementedError(
                f"Unsupported diffusion pipeline '{hf_pipeline_class_name}'. Supported pipelines: {supported}"
            )

        module_path, class_name = mapped
        module = importlib.import_module(module_path)
        qeff_cls = getattr(module, class_name)
        return qeff_cls

    @classmethod
    def register_pipeline(cls, hf_pipeline_class_name: str, module_path: str, class_name: str) -> None:
        """
        Register a new HF pipeline class -> QEff pipeline class mapping.
        """
        cls._HF_PIPELINE_TO_QEFF_CLASS[hf_pipeline_class_name] = (module_path, class_name)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, Any]], *args, **kwargs):
        """
        Auto-dispatch to a concrete QEff diffusion pipeline.

        Note:
            Calling this method on QEffDiffusionPipeline always returns a concrete child object.
        """
        if cls is not QEffDiffusionPipeline:
            raise NotImplementedError(f"{cls.__name__}.from_pretrained must be implemented by concrete child classes.")

        qeff_cls = cls._resolve_qeff_pipeline_class(pretrained_model_name_or_path, **kwargs)
        return qeff_cls.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def _export_modules(
        self,
        export_dir: Optional[str] = None,
        use_onnx_subfunctions: bool = False,
        skip_if_qpc_exists: bool = True,
    ) -> None:
        """
        Shared module export loop for diffusion child pipelines.
        """
        for module_name, module_obj in tqdm(self.modules.items(), desc="Exporting modules", unit="module"):
            example_inputs, dynamic_axes, output_names = module_obj.get_onnx_params()
            export_params = {
                "inputs": example_inputs,
                "output_names": output_names,
                "dynamic_axes": dynamic_axes,
                "export_dir": export_dir,
            }
            if use_onnx_subfunctions and module_name in ONNX_SUBFUNCTION_MODULE:
                export_params["use_onnx_subfunctions"] = True

            if skip_if_qpc_exists and module_obj.qpc_path is not None:
                continue
            module_obj.export(**export_params)

    def _compile_modules(
        self,
        *,
        compile_config: Optional[str] = None,
        parallel: bool = False,
        use_onnx_subfunctions: bool = False,
        specialization_updates: Optional[Dict[str, Any]] = None,
        required_module_names: Optional[list[str]] = None,
        pre_compile_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Shared module compile flow for diffusion child pipelines.
        """
        config_manager(self, config_source=compile_config, use_onnx_subfunctions=use_onnx_subfunctions)
        set_execute_params(self)

        required_module_names = required_module_names or list(self.modules.keys())
        if any(getattr(self.modules[name], "onnx_path", None) is None for name in required_module_names):
            self.export(use_onnx_subfunctions=use_onnx_subfunctions)

        if pre_compile_hook is not None:
            pre_compile_hook()

        if parallel:
            compile_modules_parallel(self.modules, self.custom_config, specialization_updates)
        else:
            compile_modules_sequential(self.modules, self.custom_config, specialization_updates)
