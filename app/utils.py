# -----------------------------------------------------------------------------
#
# Copyright (c)  2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json5 as json
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.generation.llm_generator import LLMGenerator

from transformers import TextIteratorStreamer

generator_hub = {}


def get_app_config():
    f = open("app_config.json")
    app_config = json.load(f)
    f.close()
    return app_config


def get_list_of_tasks(app_config=None):
    if app_config is None:
        app_config = get_app_config()
    return list(app_config.keys())


def get_list_of_models_all(app_config=None):
    if app_config is None:
        app_config = get_app_config()
    list_of_models = []
    for task in app_config:
        for model in app_config[task].keys():
            list_of_models.append(model)
    return list_of_models


def get_list_of_models_task(app_config, task):
    return list(app_config[task].keys())


def get_data(task, model):
    app_config = get_app_config()
    return app_config[task][model]


def load_models_artifacts():
    app_config = get_app_config()
    for task in app_config:
        generator_hub[task] = {}
        for model in app_config[task].keys():
            data = app_config[task][model]
            try:
                generator_hub[task][model] = LLMGenerator(
                    qpc_path=data["qpc_path"],
                    model_name=data["model_name"],
                    device_id=data["device_id"],
                    prompt_len=data["prompt_len"],
                    ctx_len=data["ctx_len"],
                    streamer=TextIteratorStreamer,
                )
            except Exception as err:
                print(err)
                generator_hub[task][model] = None

    print(generator_hub)


def get_generator(task, model):
    app_config = get_app_config()
    if task in generator_hub.keys():
        if model in generator_hub[task].keys():
            if generator_hub[task][model] is None:
                data = app_config[task][model]
                generator_hub[task][model] = LLMGenerator(
                    qpc_path=data["qpc_path"],
                    model_name=data["model_name"],
                    device_id=data["device_id"],
                    prompt_len=data["prompt_len"],
                    ctx_len=data["ctx_len"],
                    streamer=TextIteratorStreamer,
                )
            return generator_hub[task][model]
    return None
