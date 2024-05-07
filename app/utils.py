# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json

def get_app_config():
    f= open("app_config.json")
    app_config = json.load(f)
    f.close()
    return app_config

def get_list_of_tasks(app_config = None):
    if app_config is None:
        app_config = get_app_config()
    return list(app_config.keys())

def get_list_of_models(app_config = None): 
    if app_config is None:
        app_config = get_app_config()
    list_of_models = []  
    for task in app_config:
        for model in app_config[task].keys():
            list_of_models.append(model)
            
def get_list_of_model_task(app_config, task):
    return list(app_config[task].keys())