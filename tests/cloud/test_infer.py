import os
import pytest
import QEfficient

from QEfficient.utils.constants import Constants, QEFF_MODELS_DIR
from typing import List
import QEfficient.cloud.infer
from QEfficient.cloud.infer import main


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
@pytest.mark.run(after="test_execute")

def test_main(setup, mocker):
    ms = setup

    hf_download_spy = mocker.spy(QEfficient.cloud.infer,"hf_download")
    qpc_exists_spy = mocker.spy(QEfficient.cloud.infer,"qpc_exists")
    onnx_exists_spy = mocker.spy(QEfficient.cloud.infer,"onnx_exists")
    transform_spy = mocker.spy(QEfficient,"transform")
    qualcomm_efficient_converter_spy = mocker.spy(QEfficient.cloud.infer,"qualcomm_efficient_converter")
    compile_spy = mocker.spy(QEfficient.cloud.infer,"compile")
    latency_stats_kv_spy = mocker.spy(QEfficient.cloud.infer,"latency_stats_kv")

    # AutoTokenizer_from_pretrained_spy = mocker.spy(QEfficient.cloud.infer,"AutoTokenizer.from_pretrained")
    # AutoModelForCausalLM_from_pretrained_spy = mocker.spy(QEfficient.cloud.infer,"AutoModelForCausalLM.from_pretrained")

    main(
            model_name = ms.model_name,
            num_cores = ms.num_cores,
            prompt = ms.prompt,
            aic_enable_depth_first = ms.aic_enable_depth_first,
            mos = ms.mos,
            hf_token = ms.hf_token,
            batch_size = ms.batch_size,
            prompt_len = ms.prompt_len,
            ctx_len = ms.ctx_len,
            mxfp6 = ms.mxfp6,
            device_group = ms.device_group)
    
    assert os.path.isdir(ms.model_hf_path())
    assert os.path.isdir(ms.model_card_dir())
    assert os.path.isdir(ms.qpc_dir_path())
    assert os.path.isdir(ms.onnx_dir_path())
    assert os.path.isfile(ms.onnx_model_path())


    hf_download_spy.assert_called_once()

    # if qpc already exist then only execute function will be run
    if qpc_exists_spy.spy_return == True:
        print("____________qpc exist_________________")
        latency_stats_kv_spy.assert_called_once()
    
    # if onnx already exist then only compile and execute function will be run
    elif onnx_exists_spy.spy_return == True:
        print("____________onnx exist_________________")
        compile_spy.assert_called_once()
        assert compile_spy.spy_return == ms.generated_qpc_path()
        latency_stats_kv_spy.assert_called_once()
    
    # otherwise every low lwvel api will be run
    else:
        print("____________otherwise_________________")
        transform_spy.assert_called_once()
        qualcomm_efficient_converter_spy.assert_called_once()
        assert qualcomm_efficient_converter_spy.spy_return == ms.base_path_and_generated_onnx_path()
        compile_spy.assert_called_once()
        assert compile_spy.spy_return == ms.generated_qpc_path()
        latency_stats_kv_spy.assert_called_once()

    
