# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from ast import Not
from socket import MsgFlag
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union
import time
from loguru import logger
from sse_starlette.sse import EventSourceResponse
#from QEfficient.cloud.qaic_infer import QAICInferenceSession
from QEfficient.generation.cloud_infer import QAICInferenceSession

import numpy as np
import transformers

# Set up limit request time
EventSourceResponse.DEFAULT_PING_INTERVAL = 1000


app = FastAPI()
global_context={}

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_list=[]
    for key, items in global_context.items():
        model_name=key
        model_card = ModelCard(id=f'{model_name}')
        model_list.append(model_card)
    return ModelList(data=model_list)

class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: str = None
    name: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None

class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None
    repetition_penalty: Optional[float] = 1.1

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]]
    index: int

class ChatCompletionResponse(BaseModel):
    model: str
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):

    global model, global_tokenizers

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        model=request.model,
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
        tools=request.tools,
    )
    logger.debug(f"==== request ====\n{gen_params}")

    if request.stream:
        # Use the stream mode to read the first few characters, if it is not a function call, direct stram output
        predict_stream_generator = predict_stream(request.model, gen_params)
        return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")
    else:
        #Need to implement for non stream
        pass
def send_trunk(send_msg,model_name,stop_flag=False):
    pass

def predict_stream(model_id, gen_params):
    """
    The function call is compatible with stream mode output.

    The first seven characters are determined.
    If not a function call, the stream output is directly generated.
    Otherwise, the complete character content of the function call is returned.

    :param model_id:
    :param gen_params:
    :return:
    """
    
    messages=''
    logger.debug(f"messages is {gen_params['messages']}")

    if True:
        #Only get the last message 
        messages=gen_params['messages'][-1].content
    else:
        for msg in gen_params['messages']:
            messages= messages + f' {msg.role}:' + f' {msg.content}'
    
    #messages="Who is the president of USA in 2019?"

    print(f'messages:{messages}')
    #messages = [ChatMessage(role=msg.role, content=msg.content) for msg in gen_params['messages']]
    model_name=gen_params['model']
    tokenizer=global_context[model_name]['tokenizer']

    session=global_context[model_name]['session']
    prompt_len = max([x[session.binding_index_map["input_ids"]][1][1] for x in session.allowed_shapes])
    ctx_len = session.allowed_shapes[0][session.binding_index_map["attention_mask"]][1][1]
    input_len = tokenizer(messages, return_tensors="np", padding=True).input_ids.shape[1]
    generation_len = ctx_len - input_len

    num_chunks = -(input_len // -prompt_len)  # ceil divide without float
    input_len = num_chunks * prompt_len  # Convert input_len to a multiple of prompt_len
    assert input_len <= ctx_len, "input_len should be less than ctx_len"
    # Prepare inputs for first iteration
    inputs = tokenizer(messages, return_tensors="np", padding="max_length", max_length=input_len)
    batch_size = inputs["input_ids"].shape[0]
    inputs["position_ids"] = (np.cumsum(inputs["attention_mask"], 1) - 1) * inputs["attention_mask"]
    inputs["attention_mask"] = np.concatenate(
        [
            inputs["attention_mask"].astype(bool),
            np.zeros((batch_size, ctx_len - input_len), dtype=bool),
        ],
        1,
    )
    cache_index = np.array([0])
    inputs["cache_index"] = cache_index
    generated_ids = np.full((batch_size, generation_len - input_len + 1), tokenizer.pad_token_id)
    # Run prefill
    for i in range(num_chunks):
        chunk_inputs = inputs.copy()
        chunk_inputs["input_ids"] = inputs["input_ids"][:, i * prompt_len : (i + 1) * prompt_len]
        chunk_inputs["position_ids"] = inputs["position_ids"][:, i * prompt_len : (i + 1) * prompt_len]
        chunk_inputs["attention_mask"] = inputs["attention_mask"].copy()
        chunk_inputs["attention_mask"][:, (i + 1) * prompt_len :] = False
        outputs = session.run(chunk_inputs)
        cache_index += prompt_len
    # Get first token
    logits = outputs["logits"]
    if len(logits.shape) == 2:
        logits = np.expand_dims(logits, 1)
    next_token_id = logits.argmax(2)
    inputs["input_ids"] = next_token_id
    inputs["position_ids"] = inputs.pop("attention_mask").sum(1, keepdims=True)



    #Run the decode 

    session.skip_buffers(["attention_mask"])

    finished_sequences = next_token_id == tokenizer.eos_token_id
    while not finished_sequences.all() and cache_index[0] < generation_len:
        outputs = session.run(inputs)
        # Prepare inputs for next iteration
        logits = outputs["logits"]
        if len(logits.shape) == 2:
            logits = np.expand_dims(logits, 1)
        next_token_id = logits.argmax(2)

        finished_sequences |= next_token_id == tokenizer.eos_token_id
        inputs["input_ids"] = next_token_id
        inputs["position_ids"] += 1
        cache_index += 1

        send_msg=tokenizer.decode(next_token_id[0])

        #Send out the result 
        if ( send_msg is not None):
            message = DeltaMessage(content=send_msg,  role="assistant", function_call=None, )
            if next_token_id == tokenizer.eos_token_id:
                choice_data = ChatCompletionResponseStreamChoice(index=0,delta=message, finish_reason=None)
            else:
                choice_data = ChatCompletionResponseStreamChoice(index=0,delta=message, finish_reason='stop')
            chunk = ChatCompletionResponse(model=model_name,id="",choices=[choice_data],created=int(time.time()),object="chat.completion.chunk" )
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))

def start_server(model_name=None, tokenizer=None, qpc=None, device_id=None, port=8088):
#model_name is unique. We can set it as qwen_1.5_8_core_1BS_256PL_1024CL
#
    global_context[model_name]={}

    session=QAICInferenceSession(qpc, device_id)

    session.skip_buffers([x for x in session.input_names if x.startswith("past_")])
    session.skip_buffers([x for x in session.output_names if x.endswith("_RetainedState")])

    #session.skip_inputs(set([x for x in session.input_names if x.startswith("past_")]))
    #session.skip_outputs(set([x for x in session.output_names if x.endswith("_RetainedState")]))

    global_context[model_name]['session']= session
    tokenizer=tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    global_context[model_name]['tokenizer']=tokenizer
    global_context[model_name]['skip']=False

    uvicorn.run(app, host='0.0.0.0', port=port, workers=1)

    
