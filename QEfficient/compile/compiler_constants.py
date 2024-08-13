from dataclasses import dataclass


@dataclass
class QnnConstants():
    QAIRT_CONVERTER = "{}/bin/{}/qairt-converter"
    QAIRT_QUANTIZER = "{}/bin/{}/qairt-quantizer"

    # Other QNN Tools
    QNN_MODEL_LIB = "{}/bin/{}/qnn-model-lib-generator"
    QNN_CONTEXT_BIN = "{}/bin/{}/qnn-context-binary-generator"
    QNN_NET_RUNNER = "{}/bin/{}/qnn-net-run"
    QNN_PROFILE_VIEWER = "{}/bin/{}/qnn-profile-viewer"

    MODEL_NAME_DLC = "model.dlc"
    MODEL_NAME_DLC_QUANT = "model_quantized.dlc"

    TARGET = "x86_64-linux-clang"