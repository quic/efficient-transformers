import os
import pytest
import QEfficient

from QEfficient.utils.constants import Constants, QEFF_MODELS_DIR
from typing import List
import QEfficient.cloud.export
from QEfficient.cloud.export import main


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
@pytest.mark.order(1)

def test_main(setup, mocker):
    ms = setup
    
    hf_download_spy = mocker.spy(QEfficient.cloud.export,"hf_download")
    transform_spy = mocker.spy(QEfficient,"transform")
    qualcomm_efficient_converter_spy = mocker.spy(QEfficient.cloud.export,"qualcomm_efficient_converter")

    main(model_name=ms.model_name,cache_dir=ms.cache_dir)

    hf_download_spy.assert_called_once()
    assert hf_download_spy.spy_return == ms.model_hf_path()
    
    transform_spy.assert_called_once()

    qualcomm_efficient_converter_spy.assert_called_once()
    assert qualcomm_efficient_converter_spy.spy_return == ms.base_path_and_generated_onnx_path()


    


