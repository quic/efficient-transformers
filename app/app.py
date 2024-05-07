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


from QEfficient.generation.LLMGenerator import LLMGenerator

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
)

list_of_tasks = get_list_of_tasks()
list_of_models = get_list_of_models_all()

load_models_artifacts()

codellama = get_generator(list_of_tasks[0], "codellama")

assert codellama is not None

title = """
# <span style="color:#3253DC;">  Developer Applications on Cloud AI 100 using Transformers Library </span> 

"""

subtitle_left = """
##  Developer Application </span> 

"""

subtitle_right = """
##  <span style="color:white;"> Optimizing and Compiling Model using Qualcomm Transformers Library </span> 

"""

qeff_flags = set()
summary_text = ""


previous_current_ctx_len = 0
last_prompt = ""
last_state_generation_ids = []


def update_model(task, model):
    new_obj = get_generator(task, model)
    if new_obj is not None:
        codellama = new_obj


def get_prompt(
    message: str, chat_history: List[Tuple[str, str]], system_prompt: str
) -> str:
    texts = [f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
    # The first user input is _not_ stripped
    do_strip = False
    if chat_history:
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(f"{user_input} [/INST] {response.strip()} </s><s>[INST] ")
    message = message.strip() if do_strip else message
    texts.append(f"{message} [/INST]")
    return "".join(texts)


def run_qeff_check(model_name, progress=gr.Progress()):
    global summary_text, qeff_flags
    summary_text = ""

    if model_name not in qeff_flags:
        qeff_flags.add(model_name)
        progress(0, desc="Downloading...")
        # time.sleep(1)
        for i in progress.tqdm(range(100), desc="Downloading..."):
            time.sleep(0.0005)
        summary_text += f"$ Downloaded {model_name} from cache directory\n"
        progress(0, desc="Optimizing and Compiling...")
        time.sleep(0.5)
        for i in progress.tqdm(range(100), desc="Optimizing and Compiling..."):
            time.sleep(0.07)

        summary_text += f"$ Optimized {model_name}\n"
        # progress(0, desc="Compiling...")
        # for i in progress.tqdm(range(100), desc="Compiling..."):
        #     time.sleep(0.2)
        # summary_text += f"Optimized {model_name}\n"

    progress(0, desc="Generating Inference Container...")
    for i in progress.tqdm(range(100), desc="Generating Inference Container..."):
        pass

    summary_text += f"$ Compiled {model_name} and generated inference container\n"

    return Path("./img/box.png")


def summary():
    return summary_text


def run_codellama(msg, chat_history, task, model):

    # DEBUG
    # print(task, model)
    # output = "Hi there!"
    # return "", chat_history + [(msg, output)]
    # print(codellama)
    codellama.curr_cache_index = 0
    codellama.generated_ids = []

    if codellama.curr_cache_index >= codellama.ctx_len - 1:
        codellama.curr_cache_index = 0

    codellama.curr_cache_index = 0
    codellama.stop_indicator = True
    global last_prompt, previous_current_ctx_len, last_state_generation_ids
    last_prompt = msg
    previous_current_ctx_len = codellama.curr_cache_index
    last_state_generation_ids = codellama.generated_ids

    if not check():
        return msg, chat_history

    output = ""
    yield "", chat_history + [(msg, output)]

    generate_args = {
        "prompt": get_prompt(msg, None, "Give an brief answer."),
        "sample": True,
        "max_new_tokens": None,
    }

    t = Thread(target=codellama.generate, kwargs=generate_args)
    t.start()

    for each in codellama.streamer:
        output += each
        yield "", chat_history + [(msg, output)]

    t.join()


def stop():
    codellama.stop_indicator = False
    return


def check():
    if codellama.curr_cache_index >= codellama.ctx_len - 1:
        gr.Warning(
            f"Reached max token generation limit of {codellama.ctx_len}, Kindly press clear!"
        )
        codellama.curr_cache_index = 0
        return False
    return True


def reset_cache_index():
    codellama.curr_cache_index = previous_current_ctx_len
    codellama.generated_ids = last_state_generation_ids
    gr.Warning(f"Regenerating output for last prompt")
    return


def run_clear():
    global qeff_flags
    codellama.curr_cache_index = 0
    codellama.generated_ids = []
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

    chat.click(update_model, inputs=[dropdown1, dropdown2], outputs=[]).then(
        run_qeff_check, inputs=[dropdown2], outputs=[img]
    ).then(summary, inputs=[], outputs=[qeff_output]).then(
        run_codellama,
        inputs=[textbox, chatbot, dropdown1, dropdown2],
        outputs=[textbox, chatbot],
    )

    textbox.submit(update_model, inputs=[dropdown1, dropdown2], outputs=[]).then(
        run_qeff_check, inputs=[dropdown2], outputs=[img]
    ).then(summary, inputs=[], outputs=[qeff_output]).then(
        run_codellama,
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
    server_port=7881,
    ssl_certfile="cert.pem",
    ssl_keyfile="key.pem",
    ssl_verify=False,
    allowed_paths=[f"{os.getcwd()}"],
)
# launch()
