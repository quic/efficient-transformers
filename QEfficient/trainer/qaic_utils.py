import subprocess
from typing import List


class QAICCompiler:
    def __init__(self, num_cores: int, fp16: bool, mxfp6_matmul: bool):
        self.num_cores = num_cores
        self.fp16 = fp16
        self.mxfp6_matmul = mxfp6_matmul

    def compile(self, onnx_path: str, qpc_path: str, custom_io_path: str = None):
        args = [
            "/opt/qti-aic/exec/qaic-exec",
            "-aic-hw",
            "-aic-hw-version=2.0",
            f"-onnx-define-symbol=batch_size,{self._train_batch_size}",
            f"-onnx-define-symbol=seq_len,{self.args.max_ctx_len}",
            f"-aic-num-cores={self.num_cores}",
            "-compile-only",
            f"-m={onnx_path}",
            f"-aic-binary-dir={qpc_path}",
        ]
        if self.fp16:
            args.append("-convert-to-fp16")
        if self.mxfp6_matmul:
            args.append("-mxfp6-matmul")
        if custom_io_path:
            args.append(f"-custom-IO-list-file={custom_io_path}")

        # or use LRT HL Apis()
        # qaic.session(onnx_path, qpc_path)

        subprocess.run(args).check_returncode()


class QAICLoader:
    def __init__(self, qpc_path: str, device_ids: List[int]):
        self.qpc_path = qpc_path
        self.device_ids = device_ids

    def load_model(self):
        # Implement QAIC model loading here
        pass

    def get_session(self):
        # Return the loaded QAIC session
        pass
