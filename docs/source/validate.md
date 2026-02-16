(validated_models)=
# Validated Models

## Text-only Language Models

### Text Generation Task
**QEff Auto Class:** `QEFFAutoModelForCausalLM`

| Architecture            | Model Family       | Representative Models                                                                 | [vLLM Support](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/vLLM/vLLM/index.html) |
|-------------------------|--------------------|--------------------------------------------------------------------------------------|--------------|
| **MolmoForCausalLM** | Molmo① | [allenai/Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924) | ✕           |
| **Olmo2ForCausalLM**   |       OLMo-2       | [allenai/OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B)                                                               | ✔️         |
| **FalconForCausalLM**   | Falcon②            | [tiiuae/falcon-40b](https://huggingface.co/tiiuae/falcon-40b)                                                                | ✔️          |
| **Qwen3MoeForCausalLM**   | Qwen3Moe             | [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)                                                                | ✔️          |
| **GemmaForCausalLM**    | CodeGemma          | [google/codegemma-2b](https://huggingface.co/google/codegemma-2b)<br>[google/codegemma-7b](https://huggingface.co/google/codegemma-7b)                                           | ✔️          |
|                         | Gemma③             | [google/gemma-2b](https://huggingface.co/google/gemma-2b)<br>[google/gemma-7b](https://huggingface.co/google/gemma-7b)<br>[google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b)<br>[google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b)<br>[google/gemma-2-27b](https://huggingface.co/google/gemma-2-27b)        | ✔️          |
| **GptOssForCausalLM** | GPT-OSS            | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)                                                   | ✔️          |
| **GPTBigCodeForCausalLM** | Starcoder1.5      | [bigcode/starcoder](https://huggingface.co/bigcode/starcoder)                                                                   | ✔️          |
|                         | Starcoder2         | [bigcode/starcoder2-15b](https://huggingface.co/bigcode/starcoder2-15b)                                                              | ✔️          |
| **GPTJForCausalLM**     | GPT-J              | [EleutherAI/gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b)                                                                 | ✔️          |
| **GPT2LMHeadModel**     | GPT-2              | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)                                                               | ✔️          |
| **GraniteForCausalLM**  | Granite 3.1        | [ibm-granite/granite-3.1-8b-instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct)<br>[ibm-granite/granite-guardian-3.1-8b](https://huggingface.co/ibm-granite/granite-guardian-3.1-8b)          | ✔️          |
|                         | Granite 20B        | [ibm-granite/granite-20b-code-base-8k](https://huggingface.co/ibm-granite/granite-20b-code-base-8k)<br>[ibm-granite/granite-20b-code-instruct-8k](https://huggingface.co/ibm-granite/granite-20b-code-instruct-8k)    | ✔️          |
| **InternVLChatModel**   | Intern-VL①         | [OpenGVLab/InternVL2_5-1B](https://huggingface.co/OpenGVLab/InternVL2_5-1B) <br> [OpenGVLab/InternVL3_5-1B](https://huggingface.co/OpenGVLab/InternVL3_5-1B)  | ✔️          |                                                         |            |
| **LlamaForCausalLM**    | CodeLlama          | [codellama/CodeLlama-7b-hf](https://huggingface.co/codellama/CodeLlama-7b-hf)<br>[codellama/CodeLlama-13b-hf](https://huggingface.co/codellama/CodeLlama-13b-hf)<br>[codellama/CodeLlama-34b-hf](https://huggingface.co/codellama/CodeLlama-34b-hf) | ✔️          |
|                         | DeepSeek-R1-Distill-Llama | [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)                                      | ✔️          |
|                         | InceptionAI-Adapted | [inceptionai/jais-adapted-7b](https://huggingface.co/inceptionai/jais-adapted-7b)<br>[inceptionai/jais-adapted-13b-chat](https://huggingface.co/inceptionai/jais-adapted-13b-chat)<br>[inceptionai/jais-adapted-70b](https://huggingface.co/inceptionai/jais-adapted-70b) | ✔️          |
|                         | Llama 3.3          | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)                                                   | ✔️          |
|                         | Llama 3.2          | [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)<br>[meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)                                  | ✔️          |
|                         | Llama 3.1          | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)<br>[meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)                                 | ✔️          |
|                         | Llama 3            | [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)<br>[meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)                           | ✔️          |
|                         | Llama 2            | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)<br>[meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)<br>[meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) | ✔️          |
|                         | Vicuna             | [lmsys/vicuna-13b-delta-v0](https://huggingface.co/lmsys/vicuna-13b-delta-v0)<br>[lmsys/vicuna-13b-v1.3](https://huggingface.co/lmsys/vicuna-13b-v1.3)<br>[lmsys/vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5)         | ✔️          |
| **MistralForCausalLM**  | Mistral            | [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)                                                  | ✔️          |
| **MixtralForCausalLM**  | Codestral<br>Mixtral | [mistralai/Codestral-22B-v0.1](https://huggingface.co/mistralai/Codestral-22B-v0.1)<br>[mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)                        | ✔️          |
| **Phi3ForCausalLM**     | Phi-3②, Phi-3.5②     | [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)                                                    | ✔️          |
| **QwenForCausalLM**     | DeepSeek-R1-Distill-Qwen | [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)                                                   | ✔️          |
|                         | Qwen2, Qwen2.5     | [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)                                                            | ✔️          |
| **LlamaSwiftKVForCausalLM**  | swiftkv            | [Snowflake/Llama-3.1-SwiftKV-8B-Instruct](https://huggingface.co/Snowflake/Llama-3.1-SwiftKV-8B-Instruct)                                                  | ✔️          |
| **Grok1ModelForCausalLM**  |  grok-1②          | [hpcai-tech/grok-1](https://huggingface.co/hpcai-tech/grok-1)                                                  | ✕          |


---

## Embedding Models

### Text Embedding Task
**QEff Auto Class:** `QEFFAutoModel`

| Architecture | Model Family | Representative Models          | vLLM Support |
|--------------|--------------|---------------------------------|--------------|
| **BertModel** | BERT-based   | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)<br> [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)<br>[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) <br>[e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) | ✔️          |
| **MPNetForMaskedLM** | MPNet | [sentence-transformers/multi-qa-mpnet-base-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-cos-v1) | ✔️         |
| **NomicBertModel** | NomicBERT② | [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | ✕          |
| **RobertaModel**     | RoBERTa |  [ibm-granite/granite-embedding-30m-english](https://huggingface.co/ibm-granite/granite-embedding-30m-english)<br> [ibm-granite/granite-embedding-125m-english](https://huggingface.co/ibm-granite/granite-embedding-125m-english) | ✔️          |
| **XLMRobertaForSequenceClassification** | XLM-RoBERTa | [bge-reranker-v2-m3bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | ✔️          |
| **XLMRobertaModel**    | XLM-RoBERTa  |[ibm-granite/granite-embedding-107m-multilingual](https://huggingface.co/ibm-granite/granite-embedding-107m-multilingual)<br> [ibm-granite/granite-embedding-278m-multilingual](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual) <br> [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | ✔️          |

---

## Sequence Classification Models

### Text Classification Task
**QEff Auto Class:** `QEFFAutoModelForSequenceClassification`

| Architecture | Model Family | Representative Models | vLLM Support |
|--------------|--------------|----------------------|--------------|
| **DebertaV2ForSequenceClassification** | Llama Prompt Guard | [meta-llama/Llama-Prompt-Guard-2-22M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M) | ✕ |

---

## Multimodal Language Models

### Vision-Language Models (Text + Image Generation)
**QEff Auto Class:** `QEFFAutoModelForImageTextToText`

| Architecture                        | Model Family | Representative Models                                                                 | Qeff Single Qpc | Qeff Dual Qpc | vllm Single Qpc | vllm Dual Qpc |
|------------------------------------|--------------|----------------------------------------------------------------------------------------|------------|---------------------|-------------------|-----------------|
| **LlavaForConditionalGeneration**  | LLaVA-1.5   | [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)         | ✔️               | ✔️                       | ✔️                      | ✔️                      |
| **MllamaForConditionalGeneration** | Llama 3.2   | [meta-llama/Llama-3.2-11B-Vision Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)<br>[meta-llama/Llama-3.2-90B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct)           | ✔️                       | ✔️                      | ✔️                      | ✔️                      |
| **LlavaNextForConditionalGeneration** | Granite Vision | [ibm-granite/granite-vision-3.2-2b](https://huggingface.co/ibm-granite/granite-vision-3.2-2b)  | ✕                       | ✔️                      | ✕                       | ✔️                      |
| **Llama4ForConditionalGeneration** | Llama-4-Scout | [Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)  | ✔️                       | ✔️                      | ✔️                       | ✔️                      |
| **Gemma3ForConditionalGeneration** | Gemma3③       | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)  | ✔️               | ✔️                       |                ✕        |                 ✕       |
| **Qwen2_5_VLForConditionalGeneration** | Qwen2.5-VL | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)  | ✔️               | ✔️                       |             ✕           |          ✔️             |
| **Mistral3ForConditionalGeneration** | Mistral3| [mistralai/Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)| ✕ | ✔️ | ✕  | ✕  |



**Dual QPC:**
In the Dual QPC(Qualcomm Program Container) setup, the model is split across two  configurations:

- The **Vision Encoder** runs in one QPC.
- The **Language Model** (responsible for output generation) runs in a separate QPC.
- The outputs from the Vision Encoder are transferred to the Language Model.
- The dual QPC approach introduces the flexibility to run the vision and language components independently.



**Single QPC:**
In the single QPC(Qualcomm Program Container) setup, the entire model—including both image encoding and text generation—runs within a single QPC. There is no model splitting, and all components operate within the same execution environment.



```{NOTE}
The choice between Single and Dual QPC is determined during model instantiation using the `kv_offload` setting.
If the `kv_offload` is set to `True` it runs in dual QPC and if its set to `False` model runs in single QPC mode.
```

### Audio Models
(Automatic Speech Recognition) - Transcription Task

**QEff Auto Class:** `QEFFAutoModelForSpeechSeq2Seq`

| Architecture | Model Family | Representative Models                                                                 | vLLM Support |
|--------------|--------------|----------------------------------------------------------------------------------------|--------------|
| **Whisper**  | Whisper      | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)<br>[openai/whisper-base](https://huggingface.co/openai/whisper-base)<br>[openai/whisper-small](https://huggingface.co/openai/whisper-small)<br>[openai/whisper-medium](https://huggingface.co/openai/whisper-medium)<br>[openai/whisper-large](https://huggingface.co/openai/whisper-large)<br>[openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | ✔️          |
| **Wav2Vec2** | Wav2Vec2     | [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)<br>[facebook/wav2vec2-large](https://huggingface.co/facebook/wav2vec2-large) |           |

---

## Diffusion Models

### Image Generation Models
**QEff Auto Class:** `QEffFluxPipeline`

| Architecture | Model Family | Representative Models                                                                 | vLLM Support |
|--------------|--------------|----------------------------------------------------------------------------------------|--------------|
| **FluxPipeline**  | FLUX.1     | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/stabilityai/stable-diffusion-2-1) |          |

### Video Generation Models
**QEff Auto Class:** `QEffWanPipeline`

| Architecture | Model Family | Representative Models                                                                 | vLLM Support |
|--------------|--------------|----------------------------------------------------------------------------------------|--------------|
| **WanPipeline**  | Wan2.2     | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/stabilityai/stable-diffusion-2-1) |         |

---

```{NOTE}
① Intern-VL and Molmo models are Vision-Language Models but use `QEFFAutoModelForCausalLM` for inference to stay compatible with HuggingFace Transformers.

② Set `trust_remote_code=True` for end-to-end inference with vLLM.

③ Pass `disable_sliding_window` for few family models when using vLLM.
```
---



(models_coming_soon)=
# Models Coming Soon

| Architecture            | Model Family | Representative Models                      |
|-------------------------|--------------|--------------------------------------------|
| **NemotronHForCausalLM** | NVIDIA Nemotron v3   | [NVIDIA Nemotron v3](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3)             |
| **Sam3Model**   | facebook/sam3   | [facebook/sam3](https://huggingface.co/facebook/sam3)             |
| **StableDiffusionModel**     | HiDream-ai         | [HiDream-ai/HiDream-I1-Full](https://huggingface.co/HiDream-ai/HiDream-I1-Full)                       |
| **MistralLarge3Model**    | Mistral Large 3   | [mistralai/mistral-large-3](https://huggingface.co/collections/mistralai/mistral-large-3) |
