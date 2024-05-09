# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import os
import time
import json
import gradio as gr

from pathlib import Path
from threading import Thread
from typing import List, Tuple
from dotenv import load_dotenv
from jinja2.exceptions import TemplateError

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    TextStreamer,
)

from utils import (
    get_list_of_models_task,
    get_list_of_tasks,
    get_list_of_models_all,
    get_data,
    get_generator,
    load_models_artifacts,
    get_app_config
)


# Load .env file
load_dotenv()

# Load app config
app_config = get_app_config()
list_of_tasks = get_list_of_tasks()
list_of_models = get_list_of_models_all()

load_models_artifacts()

# Global variable for book keeping
qeff_generator_model = None
qeff_flags = set()
summary_text = ""
previous_current_ctx_len = 0
last_prompt = ""
last_state_generation_ids = []

# main title of app
title = """
# <span style="color:#3253DC;">  Developer Applications on Cloud AI 100 using Transformers Library </span> 
"""
# title for left container of app
subtitle_left = """
##  Developer Application </span>
"""
# title for right container of app
subtitle_right = """
##  <span style="color:white;"> Optimizing and Compiling Model using Qualcomm Transformers Library </span> 

"""


def update_model(task, model):
    global qeff_generator_model
    new_obj = get_generator(task, model)
    if new_obj is not None:
        qeff_generator_model = new_obj
        print("Updating qeff generator, ", qeff_generator_model.model_name)


def get_prompt(message : str, system_prompt:str):
    prompt = message
    chat = []
    if system_prompt:
        chat.append({"role":"system", "content":f"{system_prompt}"})
    chat.append({"role":"user", "content":f"{message}"})

    try :
        prompt = qeff_generator_model.tokenizer.apply_chat_template(chat, tokenize=False)
    except TemplateError:
        prompt = qeff_generator_model.tokenizer.apply_chat_template(chat[1:], tokenize=False)
    except Exception as err:
        print(err)
    
    return prompt
        


def run_qeff_check(task, model_name, progress=gr.Progress()):
    global summary_text, qeff_flags
    summary_text = ""
    
    model_info = get_data(task, model_name)

    if model_name not in qeff_flags:
        qeff_flags.add(model_name)
        
        # TODO : call QEfficient transform api
        # TODO : take model_info as args
        progress(0, desc="Downloading...")
        for i in progress.tqdm(range(100), desc="Downloading..."):
            time.sleep(0.0005)
        summary_text += f"$ Downloaded {model_name} from cache directory\n"
        # TODO : call QEfficient compile api
        # TODO : take model info as arguments
        # TODO : we can update the outputs from execute api in summary text
        # TODO : app_config[task][model_name]['qpc_path'] = <update qpc from qeff api>
        progress(0, desc="Optimizing and Compiling...")
        time.sleep(0.5)
        for i in progress.tqdm(range(100), desc="Optimizing and Compiling..."):
            time.sleep(0.07)

        summary_text += f"$ Optimized {model_name}\n"


    progress(0, desc="Generating Inference Container...")
    for i in progress.tqdm(range(100), desc="Generating Inference Container..."):
        pass

    summary_text += f"$ Compiled {model_name} and generated inference container\n"
    
    update_model(task, model_name)
    print(qeff_generator_model.model_name)
    
    return Path("./img/box.png")


def summary():
    return summary_text


def infer_prompt(msg, chat_history, task, model):
    global last_prompt, previous_current_ctx_len, last_state_generation_ids
    
    qeff_generator_model.curr_cache_index = 0
    qeff_generator_model.generated_ids = []

    if qeff_generator_model.curr_cache_index >= qeff_generator_model.ctx_len - 1:
        qeff_generator_model.curr_cache_index = 0

    qeff_generator_model.curr_cache_index = 0
    qeff_generator_model.stop_indicator = True
    last_prompt = msg
    previous_current_ctx_len = qeff_generator_model.curr_cache_index
    last_state_generation_ids = qeff_generator_model.generated_ids

    if not check():
        return msg, chat_history

    output = ""
    yield "", chat_history + [(msg, output)]

    generate_args = {
        "prompt": get_prompt(msg, "Give an brief answer."),
        "sample": True,
        "max_new_tokens": None,
    }

    t = Thread(target=qeff_generator_model.generate, kwargs=generate_args)
    t.start()

    for each in qeff_generator_model.streamer:
        output += each
        yield "", chat_history + [(msg, output)]

    t.join()


def stop():
    qeff_generator_model.stop_indicator = False
    return


def check():
    if qeff_generator_model.curr_cache_index >= qeff_generator_model.ctx_len - 1:
        gr.Warning(
            f"Reached max token generation limit of {qeff_generator_model.ctx_len}, Kindly press clear!"
        )
        qeff_generator_model.curr_cache_index = 0
        return False
    return True


def run_clear():
    global qeff_flags
    qeff_generator_model.curr_cache_index = 0
    qeff_generator_model.generated_ids = []
    qeff_flags = set()
    return


def clear_img(img):
    img.clear()


# Combined Interface
with gr.Blocks(theme=gr.themes.Soft(), css="demo.css") as demo:
    gr.Markdown(title)

    with gr.Row():

        with gr.Column(scale=7, variant="compact"):
            gr.Markdown(subtitle_left)

            dropdown1 = gr.Dropdown(
                list_of_tasks,
                value=list_of_tasks[0],
                label="Developer Use Case",
                elem_id="task_id",
            )

            with gr.Row():
                textbox = gr.Textbox(
                    container=False,
                    show_label=False,
                    placeholder="Type your prompt here...",
                    interactive=True,
                    lines=2,
                )

            with gr.Row():
                chat = gr.Button("Launch on AI 100", variant="primary", size="sm")

                clear = gr.Button("Reset", size="sm")

                stop_btn = gr.Button("Stop", size="sm")
            with gr.Column():
                # with gr.Group():
                chatbot = gr.Chatbot(
                    label="Response",
                    elem_id="chuanhu_chatbot",
                )
        with gr.Column(variant="compact", scale=3, elem_id="qeff_id"):
            gr.Markdown(subtitle_right)

            dropdown2 = gr.Dropdown(
                list_of_models,
                value=list_of_models[-1],
                label="Pretrained model catalogue from Qualcomm Transformers Library",
                elem_id="model_id",
            )
            img = gr.Image(
                show_label=False,
                show_download_button=False,
                container=True,
                height=260,
                width=480,
                elem_id="qpc_id",
            )
            # "block-size: inherit;"
            qeff_output = gr.Textbox(
                container=True,
                show_label=False,
                lines=4,
            )
    with gr.Row():
        gr.Image(
            "./img/full.png",
            show_label=False,
            show_download_button=False,
            container=False,
        )

    chat.click(
        run_qeff_check, inputs=[dropdown1, dropdown2], outputs=[img]
    ).then(summary, inputs=[], outputs=[qeff_output]).then(
        infer_prompt,
        inputs=[textbox, chatbot, dropdown1, dropdown2],
        outputs=[textbox, chatbot],
    )

    textbox.submit(
        run_qeff_check, inputs=[dropdown1, dropdown2], outputs=[img]
    ).then(summary, inputs=[], outputs=[qeff_output]).then(
        infer_prompt,
        inputs=[textbox, chatbot, dropdown1, dropdown2],
        outputs=[textbox, chatbot],
    )

    stop_btn.click(fn=stop)

    clear.click(lambda: None, None, chatbot, queue=False).then(
        lambda x: gr.update(value=""), [], [textbox]
    ).then(lambda x: gr.update(value=""), [], [qeff_output]).then(fn=run_clear).then(
        lambda: None, None, img, queue=False
    )
    dropdown2.change(lambda x: gr.update(value=""), [], [qeff_output]).then(
        lambda: None, None, img, queue=False
    )


demo.queue()
demo.launch(
    server_name="0.0.0.0",
    server_port=8085,
    ssl_certfile="cert.pem",
    ssl_keyfile="key.pem",
    ssl_verify=False,
    allowed_paths=[f"{os.getcwd()}"],
)
# launch()
