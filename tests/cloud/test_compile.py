import os
import pytest
import QEfficient
import json
from typing import List

from QEfficient.utils.constants import Constants, QEFF_MODELS_DIR
import QEfficient.cloud.compile
from QEfficient.cloud.compile import main

@pytest.mark.parametrize("model_name", ["gpt2"])
@pytest.mark.parametrize("num_cores", [8])
@pytest.mark.parametrize("prompt",["My name is"])
@pytest.mark.parametrize("aic_enable_depth_first",[False])
@pytest.mark.parametrize("mos",[-1])
@pytest.mark.parametrize("cache_dir",[Constants.CACHE_DIR])
@pytest.mark.parametrize("hf_token",[None])
@pytest.mark.parametrize("batch_size",[1])
@pytest.mark.parametrize("prompt_len",[32])
@pytest.mark.parametrize("ctx_len",[128])
@pytest.mark.parametrize("mxfp6",[False])
@pytest.mark.parametrize("device_group",[[1]])
@pytest.mark.run(after="test_export")

def test_main(setup, mocker):
    ms = setup

    create_and_dump_specializations_spy = mocker.spy(QEfficient.cloud.compile,"create_and_dump_specializations")
    compile_kv_model_on_cloud_ai_100_spy = mocker.spy(QEfficient.cloud.compile,"compile_kv_model_on_cloud_ai_100")
    
    main(onnx_path=ms.onnx_model_path(),
            qpc_path=os.path.dirname(ms.qpc_dir_path()),
            num_cores=ms.num_cores,
            batch_size=ms.batch_size,
            prompt_len=ms.prompt_len,
            ctx_len=ms.ctx_len,
            mxfp6=ms.mxfp6,
            aic_enable_depth_first=ms.aic_enable_depth_first,
            mos=ms.mos,
            device_group=ms.device_group,)
    
    assert os.path.isdir(os.path.join(ms.model_card_dir(), ms.qpc_base_dir_name()))
    assert os.path.isfile(ms.specialization_json_path())
    create_and_dump_specializations_spy.assert_called_once()

    if not os.path.isfile(ms.custom_io_file_path()):
        print(f"file {ms.custom_io_file_path()} needs to exist in the same directory as onnx model files.")
        assert True
    else:
        compile_kv_model_on_cloud_ai_100_spy.assert_called_once()
        assert compile_kv_model_on_cloud_ai_100_spy.spy_return[1] == ms.generated_qpc_path()


    


