import argparse
import json
import os
import subprocess
from typing import List

import onnx
import yaml

from QEfficient.compile.compile_helper import create_and_dump_specializations
from QEfficient.compile.qnn_compiler import QnnConstants


class QNN:
    def __init__(
        self,
        onnx_path: str,
        qpc_path: str,
        num_cores: int,
        qnn_sdk_path: str,
        specialization: str,
        custom_io_path: str,
        device_group: List[int],  #  FIXME: use num_devices instead
        aic_enable_depth_first: bool = False,
        mos: int = -1,
        batch_size: int = 1,
        prompt_len: int = 32,
        ctx_len: int = 128,
        mxfp6: bool = True,
        mxint8: bool = False,
        qnn_target : str = QnnConstants.TARGET,
        converter_args : str = "",
        context_bin_args : str = "",
        runtime_args: str = "",
        **kwargs,
    ) -> None:
        self.onnx_path = onnx_path
        self.qpc_path = qpc_path
        self.qnn_sdk_path = qnn_sdk_path
        self.num_cores = num_cores
        self.device_group = device_group
        self.aic_enable_depth_first = aic_enable_depth_first
        self.mos = mos
        self.batch_size = batch_size
        self.prompt_len = prompt_len
        self.ctx_len = ctx_len
        self.mxfp6 = mxfp6
        self.mxint8 = mxint8
        self.specialization_path = specialization
        self.custom_io_path = custom_io_path
        self.qnn_target = qnn_target
        self.converter_args = converter_args
        self.context_bin_args = context_bin_args
        self.runtime_args = runtime_args

        # Handle additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @staticmethod
    def execute_command(command): # TODO move this to util function. 
        return subprocess.run(command, capture_output=True, text=True)

    def compile(self):
        converter_cmd = self.converter()

        print(f"Running converter command : \n {converter_cmd}")
        result = subprocess.run(converter_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Conversion failed Failed!!\n\nSTDOUT\n{result.stdout}\n\nSTDERR\n{result.stderr}")

        ctx_bin_cmd = self.generate_context_binary()

        print(f"Running Context binary command : \n {ctx_bin_cmd}")
        result = subprocess.run(ctx_bin_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Context binary file generation failed Failed!!\n\nSTDOUT\n{result.stdout}\n\nSTDERR\n{result.stderr}")

        print("\n===================== Compilation Done! =====================\n")
        return result.returncode == 0


    def convert(self):
        converter_tool = QnnConstants.QAIRT_CONVERTER.format(self.qnn_sdk_path, self.qnn_target)

        # If execution method is DLC we need to add --export_format dlc and change the output file name to ".dlc"

        cmd = f"{converter_tool} --input_network {self.model_path} --output_path {self.qpc_path} --io_config {self.specialization_path}"

        if self.mxfp6:
            cmd += "--float_bitwidth 16 "
        
        if self.converter_args:
            cmd+=self.converter_args

        return cmd

    def quantize():
        raise NotImplementedError("QNN Quantization is not supported")

    def generate_context_binary():
        pass
    
    def execute():
        raise NotImplementedError("QNN Execution is not supported")
    
    def generate_profiling():
        raise NotImplementedError("QNN profiling is not supported")



def compile(
    onnx_path: str,
    qpc_path: str,
    num_cores: int,
    qnn_sdk_path: str,
    device_group: List[int],  #  FIXME: use num_devices instead
    aic_enable_depth_first: bool = False,
    mos: int = -1,
    batch_size: int = 1,
    prompt_len: int = 32,
    ctx_len: int = 128,
    mxfp6: bool = True,
    mxint8: bool = False,
    **kwargs,
) -> str:
    base_path = qpc_path
    aic_binary_dir = os.path.join(base_path, "qpcs")

    os.makedirs(qpc_path, exist_ok=True)
    specialization_json_path = os.path.join(qpc_path, "specializations.json")
    create_and_dump_specializations(
        batch_size=batch_size, prompt_len=prompt_len, ctx_len=ctx_len, path=specialization_json_path
    )

    # Select the customIO config based on the mx flag.
    if mxint8:
        custom_io_file_name = "custom_io_int8.yaml"
    else:
        custom_io_file_name = "custom_io_fp16.yaml"

    custom_io_file_path = os.path.join(os.path.dirname(onnx_path), custom_io_file_name)

    if not os.path.isfile(custom_io_file_path):
        raise FileNotFoundError(
            f"file {custom_io_file_path} needs to exist in the same directory as onnx model files. Please rerun infer/export Api"
        )

    command = QNN()

    if len(device_group) > 1:
        mdp_ts_config = {
            "connections": [{"devices": list(range(len(device_group))), "type": "p2p"}],
            "partitions": [
                {
                    "name": "Partition0",
                    "devices": [{"deviceId": device, "numCores": num_cores} for device in range(len(device_group))],
                }
            ],
        }
        mdp_ts_config_path = os.path.join(base_path, "mdp_ts_config.json")
        with open(mdp_ts_config_path, "w") as file:
            json.dump(mdp_ts_config, file, indent=4)
        command.append(f"-mdp-load-partition-config={mdp_ts_config_path}")

    print("Running AI 100 compiler:", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Compilation Failed!!\n\nSTDOUT\n{result.stdout}\n\nSTDERR\n{result.stderr}")

    print("\n===================== Compilation Done! =====================\n")
    return result.returncode == 0, aic_binary_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compilation script.")
    parser.add_argument("--onnx_path", "--onnx-path", required=True, help="Onnx Model Path")
    parser.add_argument(
        "--qpc-path",
        "--qpc_path",
        required=True,
        help="Compiled qpc binaries will be stored under this folder",
    )
    parser.add_argument("--batch_size", "--batch-size", type=int, default=1, help="Batch size for text generation")
    parser.add_argument(
        "--prompt_len",
        "--prompt-len",
        default=32,
        type=int,
        help="Sequence length for text generation.",
    )
    parser.add_argument("--ctx_len", "--ctx-len", default=128, type=int, help="Context length for text generation.")
    parser.add_argument(
        "--mxfp6",
        action="store_true",
        help="Compress constant MatMul weights to MXFP6 E2M3, default is no compression",
    )
    parser.add_argument(
        "--mxint8",
        action="store_true",
        help="Compress Present/Past KV to MXINT8 using CustomIO config, default is False",
    )
    parser.add_argument(
        "--num_cores",
        "--num-cores",
        required=True,
        type=int,
        help="num cores to compile the model on",
    )
    parser.add_argument(
        "--device_group",
        "--device-group",
        required=True,
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        help="Cloud AI 100 device ids (comma-separated) e.g. [0] ",
    )
    parser.add_argument(
        "--aic_enable_depth_first",
        "--aic-enable-depth-first",
        action="store_true",
        help="If passed, this option will be enabled during compilation, disabled by default",
    )
    parser.add_argument(
        "--mos",
        type=int,
        default=-1,
        help=" Effort level to reduce the on-chip memory",
    )

    parser.add_argument("--qnn_sdk_path", "--qnn-sdk-path", required=True, help="QNN SDK path ")
    # FIXME(ochougul): Allow extra compilation arguments
    args = parser.parse_args()
    compile(**vars(args))
