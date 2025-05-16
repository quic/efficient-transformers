# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import hashlib
import inspect
import json
import logging
import shutil
import subprocess
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import onnx
import torch

from QEfficient.base.onnx_transforms import OnnxTransform
from QEfficient.base.pytorch_transforms import PytorchTransform
from QEfficient.compile.qnn_compiler import compile as qnn_compile
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import constants, dump_qconfig
from QEfficient.utils.cache import QEFF_HOME, to_hashable

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
    _onnx_transforms: List[OnnxTransform]

    @classmethod
    def _transform_names(cls) -> List[str]:
        return [x.__name__ for x in cls._pytorch_transforms + cls._onnx_transforms]

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.onnx_path: Optional[str] = None
        self.qpc_path: Optional[str] = None
        self.qpc_session: Optional[QAICInferenceSession] = None

        # Apply the transformations
        any_transformed = False
        for transform in self._pytorch_transforms:
            self.model, transformed = transform.apply(self.model)
            any_transformed = any_transformed or transformed

        if not any_transformed:
            warnings.warn(f"No transforms applied to model: {self.model_name}. It may be an unsupported model!")
        else:
            logger.info(f"Pytorch transforms applied to model: {self.model_name}")

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def model_hash(self) -> str: ...

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

        ``QEFFAutoModelForCausalLM`` Args:
            :full_batch_size (int): Full batch size to allocate cache lines.
            :batch_size (int): Batch size to compile for. ``Defaults to 1``.
            :prefill_seq_len (int): Prefill sequence length to compile for. Prompt will be chunked according to this length.
            :ctx_len (int): Context length to allocate space for KV-cache tensors.

        Returns:
            :str: Path of the compiled ``qpc`` package.
        """

    def _export(
        self,
        example_inputs: Dict[str, torch.Tensor],
        output_names: List[str],
        dynamic_axes: Dict[str, Dict[int, str]],
        export_kwargs: Optional[Dict[str, any]] = None,
        onnx_transform_kwargs: Optional[Dict[str, any]] = None,
        export_dir: Optional[str] = None,
    ) -> str:
        """
        Export the Pytorch model to ONNX.

        Args:
            :example_inputs (dict): Sample inputs to trace the model.
            :output_names (list): names to assign to the output nodes of the graph, in order.
            :dynamic_axes (dict): Same as dynamic_axes parameter to be passed to `torch.onnx.export`.
            :export_kwargs (dict): Additional arguments to be passed to `torch.onnx.export`.
            :onnx_transform_kwargs (dict): Additional arguments to be passed to `Transform.apply` for this class.
            :export_dir (str): Specify the export directory. The export_dir will be suffixed with a hash corresponding to current model.
        """
        export_dir = Path(export_dir or (QEFF_HOME / self.model_name))
        export_dir = export_dir.with_name(export_dir.name + "-" + self.model_hash)
        onnx_path = export_dir / f"{self.model_name}.onnx"
        if onnx_path.is_file():
            self.onnx_path = onnx_path
            return onnx_path

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
            export_kwargs = {} if export_kwargs is None else export_kwargs
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
            logger.info("Pytorch export successful")

            model = onnx.load(tmp_onnx_path, load_external_data=False)
            transform_kwargs = {
                "onnx_base_dir": str(tmp_onnx_dir),
                "model_name": self.model_name,
            }
            if onnx_transform_kwargs is not None:
                transform_kwargs.update(onnx_transform_kwargs)

            for transform in self._onnx_transforms:
                model, transformed = transform.apply(model, **transform_kwargs)
            model.metadata_props.append(
                onnx.StringStringEntryProto(key="qeff_transforms", value=",".join(self._transform_names()))
            )
            logger.info("ONNX transforms applied")

            onnx.save(model, onnx_path)
            logger.info("Transformed onnx saved")

        except Exception as e:
            logger.error(f"ONNX export (or) ONNXTransforms failed: {e}")

            raise e

        finally:
            shutil.rmtree(tmp_onnx_dir, ignore_errors=True)

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
            :qnn_config (str): Path of QNN Config parameters file. ``Defaults to None.``
            :compiler_options: Pass any compiler option as input. Any flag that is supported by `qaic-exec` can be passed. Params are converted to flags as below:
                - aic_num_cores=16 -> -aic-num-cores=16
                - convert_to_fp16=True -> -convert-to-fp16
        """
        if onnx_path is None and self.onnx_path is None:
            self.export()

        onnx_path = Path(onnx_path or self.onnx_path)
        compile_dir = Path(compile_dir or onnx_path.parent)
        qpc_path = compile_dir / "qpc"
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX file not found at: {onnx_path}")

        if enable_qnn:
            self.qpc_path = qnn_compile(
                onnx_path=onnx_path,
                qpc_base_path=compile_dir,
                specializations=specializations,
                custom_io=custom_io,
                device_group=list(range(mdp_ts_num_devices)),
                num_cores=compiler_options.get("aic_num_cores", 16),
                mxfp6=compiler_options.get("mxfp6_matmul", False),
                mxint8=mxint8_kv_cache,
                qnn_config=qnn_config,
            )

            return self.qpc_path

        command = constants.COMPILER + [f"-m={onnx_path}"]
        if mdp_ts_json_path := compiler_options.pop("mdp_ts_json_path", None):
            mdp_ts_num_devices = None
            command.append(f"-mdp-load-partition-config={mdp_ts_json_path}")

        for key, value in compiler_options.items():
            option = "-" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(option)
                continue
            command.append(f"{option}={value}")
        compile_hash = hashlib.sha256(to_hashable(command))

        if specializations is not None:
            compile_hash.update(to_hashable(specializations))

        if custom_io is not None:
            compile_hash.update(to_hashable(custom_io))

        if num_speculative_tokens:
            compile_hash.update(to_hashable({"num_speculative_tokens": num_speculative_tokens}))

        # Check if already compiled
        compile_hash = compile_hash.hexdigest()[:16]
        compile_dir = qpc_path.with_name(qpc_path.name + "-" + compile_hash)
        qpc_path = compile_dir / "qpc"
        qpc_path.mkdir(parents=True, exist_ok=True)
        if qpc_path.is_dir():
            if (qpc_path / "programqpc.bin").is_file():
                self.qpc_path = qpc_path
                return qpc_path
            # Probably compilation failure last time, delete directory to start over
            shutil.rmtree(qpc_path)

        # Write specializations.json file
        if specializations is not None:
            specializations_json = compile_dir / "specializations.json"
            with open(specializations_json, "w") as fp:
                json.dump(
                    {"specializations": [{k: str(v) for k, v in spec.items()} for spec in specializations]},
                    fp,
                    indent=4,
                )
            command.append(f"-network-specialization-config={specializations_json}")

        # Write custom_io.yaml file
        if custom_io is not None:
            custom_io_yaml = compile_dir / "custom_io.yaml"
            with open(custom_io_yaml, "w") as fp:
                for io_name, dtype in custom_io.items():
                    fp.write(f" - IOName: {io_name}\n   Precision: {dtype}\n\n")
            command.append(f"-custom-IO-list-file={custom_io_yaml}")

        # Write mdp_config.json file
        if not mdp_ts_json_path and mdp_ts_num_devices > 1:
            num_cores = compiler_options.get("aic_num_cores", 16)
            mdp_ts_json = compile_dir / f"mdp_ts_{mdp_ts_num_devices}.json"
            with open(mdp_ts_json, "w") as fp:
                json.dump(
                    {
                        "connections": [{"devices": list(range(mdp_ts_num_devices)), "type": "p2p"}],
                        "partitions": [
                            {
                                "name": "Partition0",
                                "devices": [{"deviceId": d, "numCores": num_cores} for d in range(mdp_ts_num_devices)],
                            }
                        ],
                    },
                    fp,
                    indent=4,
                )
            command.append(f"-mdp-load-partition-config={mdp_ts_json}")

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

        self.qpc_path = qpc_path

        return qpc_path
