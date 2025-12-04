# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import gc
import inspect
import logging
import re
import shutil
import subprocess
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import onnx
import torch

from QEfficient.base.onnx_transforms import (
    BaseOnnxTransform,
    CustomOpTransform,
    OnnxTransformPipeline,
    RenameFunctionOutputsTransform,
)
from QEfficient.base.pytorch_transforms import PytorchTransform
from QEfficient.compile.qnn_compiler import compile as qnn_compile
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.cache_utils import InvalidIndexProvider
from QEfficient.transformers.models.pytorch_transforms import get_decoder_layer_classes_for_export
from QEfficient.utils import (
    constants,
    create_json,
    create_model_params,
    dump_qconfig,
    export_wrapper,
    generate_mdp_partition_config,
    hash_dict_params,
    load_json,
)
from QEfficient.utils.torch_patches import apply_torch_patches, undo_torch_patches

logger = logging.getLogger(__name__)


class QEFFBaseModel(ABC):
    """
    Base class for all the model classes (i.e. LLMs, SD, quantized etc.).
    Provides certain utility methods to be used by child classes.

    Class variables:
    :_pytorch_transforms: Pytorch transformations to be applied after initialization.
    :_onnx_transforms: ONNX transformations to be applied after ONNX export.
    """

    _pytorch_transforms: List[PytorchTransform]
    _onnx_transforms = [BaseOnnxTransform]

    @classmethod
    def _transform_names(cls) -> List[str]:
        return [x.__name__ for x in cls._pytorch_transforms + cls._onnx_transforms]

    def __init__(self, model: torch.nn.Module, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.hash_params = create_model_params(self, **kwargs)
        self.onnx_path: Optional[str] = None
        self.qpc_path: Optional[str] = None
        self.qpc_session: Optional[QAICInferenceSession] = None
        self.model_architecture = (
            (arch := getattr(self.model.config, "architectures", None)) and len(arch) > 0 and arch[0]
        ) or None

        # Flag for checking if weights are offloaded
        self._is_weights_offloaded: bool = False

        # Apply the transformations
        any_transformed = False
        for transform in self._pytorch_transforms:
            self.model, transformed = transform.apply(self.model)
            any_transformed = any_transformed or transformed

        if not any_transformed:
            warnings.warn(f"No transforms applied to model: {self.model_name}. It may be an unsupported model!")
        else:
            logger.info(f"Pytorch transforms applied to model: {self.model_name}")

    def _offload_model_weights(self, offload_pt_weights: bool) -> bool:
        """Clear PyTorch model weights to reduce memory usage after ONNX export."""
        if offload_pt_weights and not self._is_weights_offloaded:
            try:
                for param in self.model.parameters():
                    if param.storage():
                        param.storage().resize_(0)
                for buffer in self.model.buffers():
                    if buffer.storage():
                        buffer.storage().resize_(0)

                meta_model = self.model.to("meta")
                del self.model
                gc.collect()

                self.model = meta_model
                self._is_weights_offloaded = True
                return True
            except Exception as e:
                logger.warning(f"Weight clearing failed, continuing: {e}")
                return False
        return False

    def _model_offloaded_check(self) -> None:
        """
        Check if the model is in meta state or weights are offloaded.

        Raises:
            RuntimeError: If model is in meta state or if weights are offloaded
        """
        if self._is_weights_offloaded or any(param.is_meta for param in self.model.parameters()):
            error_msg = (
                "Cannot re-export model: weights have been offloaded to save memory. "
                "To re-export, please create a new model instance using from_pretrained() method."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def export(self, export_dir: Optional[str] = None) -> Path:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.

        Args:
            :export_dir (str): Specify the export directory. The export_dir will be suffixed with a hash corresponding to current model.

        Returns:
            :Path: Path of the generated ``ONNX`` file.
        """

    @abstractmethod
    def compile(self, *args, **kwargs) -> Path:
        """
        Compile the exported onnx to run on AI100.
        If the model has not been exported yet, this method will handle the export process.

        Args:
            :onnx_path (str): Onnx file to compile
            :compile_dir (str): Directory path to compile the qpc. A suffix is added to the directory path to avoid reusing same qpc for different parameters.
            :num_devices (int): Number of devices to compile for. ``Defaults to 1``.
            :num_cores (int): Number of cores to utilize in each device ``Defaults to 16``.
            :mxfp6_matmul (bool): Use MXFP6 to compress weights for MatMul nodes to run faster on device. ``Defaults to False``.
            :mxint8_kv_cache (bool): Use MXINT8 to compress KV-cache on device to access and update KV-cache faster. ``Defaults to False``.
            :compiler_options: Pass any compiler option as input.

                Following flag can be passed in compiler_options to enable QNN Compilation path.
                    :enable_qnn (bool): Enables QNN Compilation. ``Defaults to False. if not passed.``
                    :qnn_config (str): Path of QNN Config parameters file. ``Defaults to None. if not passed``

                for QAIC compilation path, any flag that is supported by ``qaic-exec`` can be passed. Params are converted to flags as below:

                    - aic_num_cores=16 -> -aic-num-cores=16
                    - convert_to_fp16=True -> -convert-to-fp16
                    - aic_hw_version=ai100 -> -aic-hw-version=ai100
                    - aic_hw_version=ai200 -> -aic-hw-version=ai200

        ``QEFFAutoModelForCausalLM`` Args:

            :full_batch_size (int): Full batch size to allocate cache lines.
            :batch_size (int): Batch size to compile for. ``Defaults to 1``.
            :prefill_seq_len (int): Prefill sequence length to compile for. Prompt will be chunked according to this length.
            :ctx_len (int): Context length to allocate space for KV-cache tensors.

        Returns:
            :str: Path of the compiled ``qpc`` package.
        """

    @export_wrapper
    def _export(
        self,
        example_inputs: Dict[str, torch.Tensor],
        output_names: List[str],
        dynamic_axes: Dict[str, Dict[int, str]],
        export_kwargs: Optional[Dict[str, any]] = None,
        onnx_transform_kwargs: Optional[Dict[str, any]] = None,
        export_dir: Optional[str] = None,
        offload_pt_weights: bool = True,
        use_onnx_subfunctions: bool = False,
    ) -> str:
        """
        Export the PyTorch model to ONNX and apply ONNX transforms

        This method:
        1. Exports PyTorch model to ONNX using torch.onnx.export
        2. Clears PyTorch weights after export
        3. Applies ONNX transforms with reduced memory footprint

        Args:
            :example_inputs (dict): Sample inputs to trace the model.
            :output_names (list): names to assign to the output nodes of the graph, in order.
            :dynamic_axes (dict): Same as dynamic_axes parameter to be passed to `torch.onnx.export`.
            :export_kwargs (dict): Additional arguments to be passed to `torch.onnx.export`.
            :onnx_transform_kwargs (dict): Additional arguments to be passed to `Transform.apply` for this class.
            :export_dir (str): Specify the export directory. The export_dir will be suffixed with a hash corresponding to current model.
            :offload_pt_weights (bool): If True, offload PyTorch model weights to meta device
            after successful export to reduce memory usage. Set to False if you need to
            keep weights for further operations. Defaults to True.
            Note:
            Once weights are offloaded, the model cannot be re-exported. Create a new
            instance using from_pretrained() for re-export.

        """
        onnx_path = export_dir / f"{self.model_name}.onnx"

        # Return early if ONNX already exists
        if onnx_path.is_file():
            self.onnx_path = onnx_path
            return onnx_path

        # check if the model is in meta state or weights are offloaded
        self._model_offloaded_check()

        # Setup temporary paths
        tmp_onnx_dir = export_dir / "onnx_tmp"
        tmp_onnx_path = tmp_onnx_dir / f"{self.model_name}.onnx"
        tmp_onnx_dir.mkdir(parents=True, exist_ok=True)

        # Create input_names from example_inputs
        input_names = []
        for param in inspect.signature(self.model.forward).parameters:
            if param in example_inputs:
                if param == "past_key_values":
                    for i in range(len(example_inputs["past_key_values"])):
                        if len(example_inputs["past_key_values"][0]) == 2:
                            input_names.extend([f"past_key.{i}", f"past_value.{i}"])
                        elif len(example_inputs["past_key_values"][0]) == 4:
                            input_names.extend(
                                [
                                    f"past_key_self.{i}",
                                    f"past_value_self.{i}",
                                    f"past_key_cross.{i}",
                                    f"past_value_cross.{i}",
                                ]
                            )
                        else:
                            raise ValueError(
                                f"Unknown shape of past_key_values! Expected length of past_key_values for each layer to be either 2 or 4 but got {len(example_inputs['past_key_values'][0])}"
                            )
                else:
                    input_names.append(param)

        try:
            # Initialize the registry with your custom ops
            export_kwargs = {} if export_kwargs is None else export_kwargs
            if use_onnx_subfunctions:
                warnings.warn(
                    "The subfunction feature is experimental. Please note that using compile consecutively with and without subfunction may produce inconsistent results."
                )
                apply_torch_patches()
                InvalidIndexProvider.SUBFUNC_ENABLED = True
                output_names = [re.sub("_RetainedState", "_InternalRetainedState", s) for s in output_names]
                export_kwargs["export_modules_as_functions"] = get_decoder_layer_classes_for_export(self.model)
                self._onnx_transforms.append(RenameFunctionOutputsTransform)
                self._onnx_transforms.append(CustomOpTransform)

            torch.onnx.export(
                self.model,
                (example_inputs,),
                str(tmp_onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=constants.ONNX_EXPORT_OPSET,
                **export_kwargs,
            )
            logger.info("PyTorch export successful")
            _ = self._offload_model_weights(offload_pt_weights)
            model = onnx.load(tmp_onnx_path, load_external_data=False)

            # Clear temporary references
            transform_kwargs = {
                "onnx_base_dir": str(tmp_onnx_dir),
                "model_name": self.model_name,
            }
            if onnx_transform_kwargs is not None:
                transform_kwargs.update(onnx_transform_kwargs)

            onnx_transforms = OnnxTransformPipeline(transforms=self._onnx_transforms)
            model, transformed = onnx_transforms.apply(model, **transform_kwargs)

            # Add metadata to the model
            model.metadata_props.append(
                onnx.StringStringEntryProto(key="qeff_transforms", value=",".join(self._transform_names()))
            )
            logger.info("ONNX transforms applied")

            onnx.save(model, onnx_path)
            del model
            gc.collect()
            logger.info("Transformed ONNX saved")

        except Exception as e:
            logger.error(f"ONNX export or transforms failed: {e}")
            raise e

        finally:
            shutil.rmtree(tmp_onnx_dir, ignore_errors=True)

        if use_onnx_subfunctions:
            undo_torch_patches()
            InvalidIndexProvider.SUBFUNC_ENABLED = False
            self._onnx_transforms.remove(CustomOpTransform)
            self._onnx_transforms.remove(RenameFunctionOutputsTransform)

        self.onnx_path = onnx_path
        return onnx_path

    @dump_qconfig
    def _compile(
        self,
        onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        *,
        mxint8_kv_cache: bool = False,
        specializations: Optional[List[Dict[str, int]]] = None,
        custom_io: Optional[Dict[str, str]] = None,
        mdp_ts_num_devices: int = 1,
        num_speculative_tokens: Optional[int] = None,
        enable_qnn: Optional[bool] = False,
        qnn_config: Optional[str] = None,
        use_onnx_subfunctions: bool = False,
        **compiler_options,
    ) -> str:
        """
        Interface for qaic-exec compiler

        Args:
            :onnx_path (str): Onnx file to compile
            :compile_dir (str): Directory path to compile the qpc. A suffix is added to the directory path to avoid reusing same qpc for different parameters.
            :mxint8_kv_cache (bool, optional): Whether to use ``mxint8`` compression for KV cache. ``Defaults to False``.
            :specializations (list): List of specializations to compile for
            :custom_io (dict): Custom IO to specify the input and outputs in different formats than default
            :mdp_ts_num_devices (int): Number of devices to partition to use Multi-Device Partitioning with tensor-slicing.
            :num_speculative_tokens (int, optional): Number of speculative tokens to take as input for Speculative Decoding Target Language Model.
            :enable_qnn (bool): Enables QNN Compilation. ``Defaults to False.``
            :qnn_config (str): Path of QNN Config parameters file. Any extra parameters for QNN compilation can be passed via this file. ``Defaults to None.``
            :compiler_options: Pass any compiler option as input.
                Any flag that is supported by `qaic-exec` can be passed. Params are converted to flags as below:

                - aic_num_cores=16 -> -aic-num-cores=16
                - convert_to_fp16=True -> -convert-to-fp16
                - aic_hw_version=ai100 -> -aic-hw-version=ai100
                - aic_hw_version=ai200 -> -aic-hw-version=ai200

                For QNN Compilation path, when enable_qnn is set to True, any parameter passed in compiler_options will be ignored.
        """

        if onnx_path is None and self.onnx_path is None:
            self.export(use_onnx_subfunctions=use_onnx_subfunctions)

        onnx_path = Path(onnx_path or self.onnx_path)
        compile_dir = Path(compile_dir or onnx_path.parent)
        qpc_path = compile_dir / "qpc"
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX file not found at: {onnx_path}")

        if enable_qnn:
            if compiler_options:
                logger.warning(
                    f"Extra arguments to QNN compilation are supported only via qnn_config file. Ignoring {compiler_options}"
                )

            self.qpc_path = qnn_compile(
                onnx_path=onnx_path,
                qpc_base_path=compile_dir,
                specializations=specializations,
                custom_io=custom_io,
                device_group=list(range(mdp_ts_num_devices)),
                num_cores=compiler_options.get("aic_num_cores", constants.DEFAULT_AIC_NUM_CORES),
                mxfp6=compiler_options.get("mxfp6_matmul", constants.DEFAULT_AIC_MXPF6_MATMUL),
                mxint8=mxint8_kv_cache,
                qnn_config=qnn_config,
            )

            return self.qpc_path

        command = (
            constants.COMPILER
            + [
                f"-aic-hw-version={compiler_options.pop('aic_hw_version', compiler_options.pop('aic-hw-version', constants.DEFAULT_AIC_HW_VERSION))}"
            ]
            + [f"-m={onnx_path}"]
        )

        if mdp_ts_json_path := compiler_options.pop("mdp_load_partition_config", None):
            command.append(f"-mdp-load-partition-config={mdp_ts_json_path}")

        for key, value in compiler_options.items():
            option = "-" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(option)
                continue
            command.append(f"{option}={value}")

        # Create a dummy mdp_ts_json if mdp-load-partition-config not provided and num_devices > 1
        if mdp_ts_json_path is not None:
            mdp_ts_json = load_json(str(mdp_ts_json_path))
        elif mdp_ts_num_devices > 1:
            mdp_ts_json = generate_mdp_partition_config(
                mdp_ts_num_devices, compiler_options.get("aic_num_cores", constants.DEFAULT_AIC_NUM_CORES)
            )
        else:
            mdp_ts_json = None

        compile_hash_params = {
            "command": command,
            "specializations": specializations,
            "custom_io": custom_io,
            "mdp_ts_num_devices": mdp_ts_num_devices,
            "mdp_ts_json": mdp_ts_json,
            "num_speculative_tokens": num_speculative_tokens,
        }
        compile_hash = hash_dict_params(compile_hash_params)

        compile_dir = qpc_path.with_name(qpc_path.name + "-" + compile_hash)
        qpc_path = compile_dir / "qpc"
        qpc_path.mkdir(parents=True, exist_ok=True)

        if qpc_path.is_dir():
            if (qpc_path / "programqpc.bin").is_file():
                self.qpc_path = qpc_path
                return qpc_path
            # Probably compilation failure last time, delete directory to start over
            shutil.rmtree(qpc_path)

        # write the MDP partition config file if not provided
        if mdp_ts_json is not None:
            mdp_ts_json_path = compile_dir / f"mdp_ts_{mdp_ts_num_devices}.json"
            create_json(str(mdp_ts_json_path), mdp_ts_json)
            command.append(f"-mdp-load-partition-config={mdp_ts_json_path}")

        # Write specializations.json file
        if specializations is not None:
            specializations_json = compile_dir / "specializations.json"
            specializations_data = {
                "specializations": [{k: str(v) for k, v in spec.items()} for spec in specializations]
            }
            create_json(str(specializations_json), specializations_data)
            command.append(f"-network-specialization-config={specializations_json}")

        # Write custom_io.yaml file
        if custom_io is not None:
            custom_io_yaml = compile_dir / "custom_io.yaml"
            with open(custom_io_yaml, "w") as fp:
                for io_name, dtype in custom_io.items():
                    fp.write(f" - IOName: {io_name}\n   Precision: {dtype}\n\n")
            command.append(f"-custom-IO-list-file={custom_io_yaml}")

        command.append(f"-aic-binary-dir={qpc_path}")
        logger.info(f"Running compiler: {' '.join(command)}")
        try:
            subprocess.run(command, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "\n".join(
                    [
                        "Compilation failed!",
                        f"Compiler command: {e.cmd}",
                        f"Compiler exitcode: {e.returncode}",
                        "Compiler stderr:",
                        e.stderr.decode(),
                    ]
                )
            )
        # Dump JSON file with hashed parameters
        hashed_compile_params_path = compile_dir / "hashed_compile_params.json"
        create_json(hashed_compile_params_path, compile_hash_params)
        logger.info("Hashed parameters exported successfully.")

        self.qpc_path = qpc_path

        return qpc_path
