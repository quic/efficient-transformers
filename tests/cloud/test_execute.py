import os
import pytest
import QEfficient
import json
from typing import List

from QEfficient.utils.constants import Constants, QEFF_MODELS_DIR
import QEfficient.cloud.execute
from QEfficient.cloud.execute import main


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
@pytest.mark.run(after="test_compile")

def test_main(setup, mocker):
    ms = setup
    
    login_spy = mocker.spy(QEfficient.cloud.execute,"login")
    hf_download_spy = mocker.spy(QEfficient.cloud.execute,"hf_download")
    latency_stats_kv_spy = mocker.spy(QEfficient.cloud.execute,"latency_stats_kv")
    
    main(model_name=ms.model_name,
    prompt=ms.prompt,
    qpc_path=ms.qpc_dir_path(),
    devices=ms.device_group,
    cache_dir=ms.cache_dir,
    hf_token=ms.hf_token,)
    
    if ms.hf_token is not None:
        login_spy.assert_called_once_with(ms.hf_token)
    
    hf_download_spy.assert_called_once()
    assert hf_download_spy.spy_return == ms.model_hf_path()

    latency_stats_kv_spy.assert_called_once()

