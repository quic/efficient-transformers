# QEfficient.cloud.infer
```{eval-rst}
   - Check if compiled qpc for given config already exists, if it does jump to execute, else
   - Check if exported ONNX file already exists, if true, jump to compilation -> execution, else
   - Check if HF model exists in cache, if true, start transform -> export -> compilation -> execution, else,
   - Download HF model -> transform -> export -> compile -> execute
```
```bash
   python -m QEfficient.cloud.infer OPTIONS
```

```{eval-rst}

   ---------
    :model_name: str. Hugging Face Model Card name, Example: "gpt2"
    :num_cores: int. :num_cores: int. Number of cores to compile model on.
    :device_group: List[int]. Device Ids to be used for compilation. if len(device_group) > 1. Multiple Card setup is enabled.
    :prompt: str. Sample prompt for the model text generation
    :prompts_txt_file_path: str. Path to txt file for multiple input prompts
    :aic_enable_depth_first: bool. Enables DFS with default memory size, disabled by default.
    :mos: int. Effort level to reduce the on-chip memory.
    :batch_size: int. Batch size to compile the model for.
    :prompt_len: int. prompt length for the model to compile.
    :ctx_len: int. Maximum context length to compile the model.
    :generation_len: int. Number of tokens to be generated.
    :mxfp6: bool. Enable compilation for MXFP6 precision
    :mxint8: Compress Present/Past KV to MXINT8 using CustomIO config, default is False.
    :local_model_dir: str. Path to custom model weights and config files.
    :cache_dir: str. Cache dir where downloaded HuggingFace files are stored.
    :hf_token: str. HuggingFace login token to access private repos.
 
```

# QEfficient.cloud.execute
```{eval-rst}
   - Check if compiled qpc for given config already exists, if it does jump to execute, else
   - Check if exported ONNX file already exists, if true, jump to compilation -> execution, else
   - Check if HF model exists in cache, if true, start transform -> export -> compilation -> execution, else,
   - Download HF model -> transform -> export -> compile -> execute
```
```bash
   python -m QEfficient.cloud.execute OPTIONS
```
```{eval-rst}

   ---------
    :model_name: str. Hugging Face Model Card name, Example: "gpt2"
    :num_cores: int. :num_cores: int. Number of cores to compile model on.
    :device_group: List[int]. Device Ids to be used for compilation. if len(device_group) > 1. Multiple Card setup is enabled.
    :prompt: str. Sample prompt for the model text generation
    :prompts_txt_file_path: str. Path to txt file for multiple input prompts
    :aic_enable_depth_first: bool. Enables DFS with default memory size, disabled by default.
    :mos: int. Effort level to reduce the on-chip memory.
    :batch_size: int. Batch size to compile the model for.
    :prompt_len: int. prompt length for the model to compile.
    :ctx_len: int. Maximum context length to compile the model.
    :generation_len: int. Number of tokens to be generated.
    :mxfp6: bool. Enable compilation for MXFP6 precision
    :mxint8: Compress Present/Past KV to MXINT8 using CustomIO config, default is False.
    :local_model_dir: str. Path to custom model weights and config files.
    :cache_dir: str. Cache dir where downloaded HuggingFace files are stored.
    :hf_token: str. HuggingFace login token to access private repos.
```
# QEfficient.cloud.compile
```{eval-rst}
   CLI app to compiling the Onnx Model on Cloud AI 100 Platform with given config.
```
```bash
   python -m QEfficient.cloud.compile OPTIONS
```
```{eval-rst}

   ---------
   :onnx_path: str. Generated Onnx Model Path.
   :qpc_path: str. Path for saving compiled qpc binaries.
   :num_cores: int. Number of cores to compile model on.
   :device_group: List[int]. Used for finding number of devices to compile for.
   :aic_enable_depth_first: bool. Enables DFS with default memory size, disabled by default.
   :mos: int. Effort level to reduce the on-chip memory.
   :batch_size: int. Batch size to compile the model for.
   :prompt_len: int. prompt len for the model to compile.
   :ctx_len: int. Maximum context length to compile the model.
   :mxfp6: bool. Enable compilation for MXFP6 precision
   :mxint8: Compress Present/Past KV to MXINT8 using CustomIO config, default is False.
   :custom_io_file_path: str. Path to custom IO file.
```
# QEfficient.cloud.export
```{eval-rst}
   CLI app to convert torch Bert style and KV style model to ONNX.
```
```bash
   python -m QEfficient.cloud.export OPTIONS
```
```{eval-rst}

   ---------
   :model_name: str. The name of the model to be used.
   :model_kv: torch.nn.Module. Transformed KV torch model to be used.
   :local_model_dir: str. Path of local model.
   :tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]. model tokenizer.
   :cache_dir: str. Path of the cache directory.
   :onnx_dir_path: str. Path to store ONNX file.
   :hf_token: str. Huggingface token to access gated models. Default=None.
   :seq_len: int. The length of the sequence. Default is 128.
   :kv: bool. If false, It will export to Bert style. Default=true.
   :form_factor: str. form_factor of the hardware, currently only "cloud" is accepted.

```