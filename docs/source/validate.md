(validated_models)=
# Validated Models

## Text-only Language Models

### Text Generation Task
**QEff Auto Class:** `QEFFAutoModelForCausalLM`

| Architecture            | Model Family       | Representative Models                                                                 | CB Support |
|-------------------------|--------------------|--------------------------------------------------------------------------------------|------------|
| **FalconForCausalLM**   | Falcon             | tiiuae/falcon-40b                                                                    | ✔️          |
| **GemmaForCausalLM**    | CodeGemma          | google/codegemma-2b<br>google/codegemma-7b                                           | ✔️          |
|                         | Gemma              | google/gemma-2b<br>google/gemma-7b<br>google/gemma-2-9b<br>google/gemma-2-27b        | ✔️          |
| **GPTBigCodeForCausalLM** | Starcoder1.5      | bigcode/starcoder                                                                   | ✔️          |
|                         | Starcoder2         | bigcode/starcoder2-15b                                                              | ✔️          |
| **GPTJForCausalLM**     | GPT-J              | EleutherAI/gpt-j-6b                                                                 | ✔️          |
| **GPT2LMHeadModel**     | GPT-2              | openai-community/gpt2                                                               | ✔️          |
| **GraniteForCausalLM**  | Granite 3.1        | ibm-granite/granite-3.1-8b-instruct<br>ibm-granite/granite-guardian-3.1-8b          | ✔️          |
|                         | Granite 20B        | ibm-granite/granite-20b-code-base-8k<br>ibm-granite/granite-20b-code-instruct-8k    | ✔️          |
| **InternVLChatModel**   | Intern-VL          | OpenGVLab/InternVL2_5-1B                                                            |            |
| **LlamaForCausalLM**    | CodeLlama          | codellama/CodeLlama-7b-hf<br>codellama/CodeLlama-13b-hf<br>codellama/CodeLlama-34b-hf | ✔️          |
|                         | DeepSeek-R1-Distill-Llama | deepseek-ai/DeepSeek-R1-Distill-Llama-70B                                      | ✔️          |
|                         | InceptionAI-Adapted | inceptionai/jais-adapted-7b<br>inceptionai/jais-adapted-13b-chat<br>inceptionai/jais-adapted-70b | ✔️          |
|                         | Llama 3.3          | meta-llama/Llama-3.3-70B-Instruct                                                   | ✔️          |
|                         | Llama 3.2          | meta-llama/Llama-3.2-1B<br>meta-llama/Llama-3.2-3B                                  | ✔️          |
|                         | Llama 3.1          | meta-llama/Llama-3.1-8B<br>meta-llama/Llama-3.1-70B                                 | ✔️          |
|                         | Llama 3            | meta-llama/Meta-Llama-3-8B<br>meta-llama/Meta-Llama-3-70B                           | ✔️          |
|                         | Llama 2            | meta-llama/Llama-2-7b-chat-hf<br>meta-llama/Llama-2-13b-chat-hf<br>meta-llama/Llama-2-70b-chat-hf | ✔️          |
|                         | Vicuna             | lmsys/vicuna-13b-delta-v0<br>lmsys/vicuna-13b-v1.3<br>lmsys/vicuna-13b-v1.5         | ✔️          |
| **MistralForCausalLM**  | Mistral            | mistralai/Mistral-7B-Instruct-v0.1                                                  | ✔️          |
| **MixtralForCausalLM**  | Codestral<br>Mixtral | mistralai/Codestral-22B-v0.1<br>mistralai/Mixtral-8x7B-v0.1                        | ✔️          |
| **MPTForCausalLM**      | MPT                | mosaicml/mpt-7b                                                                     | ✔️          |
| **Phi3ForCausalLM**     | Phi-3, Phi-3.5     | microsoft/Phi-3-mini-4k-instruct                                                    | ✔️          |
| **QwenForCausalLM**     | DeepSeek-R1-Distill-Qwen | DeepSeek-R1-Distill-Qwen-32B                                                   | ✔️          |
|                         | Qwen2, Qwen2.5     | Qwen/Qwen2-1.5B-Instruct                                                            | ✔️          |

## Embedding Models

### Text Embedding Task
**QEff Auto Class:** `QEFFAutoModel`

| Architecture | Model Family | Representative Models          |
|--------------|--------------|---------------------------------|
| **BertModel** | BERT-based   | BAAI/bge-base-en-v1.5           |
| **LlamaModel** | Llama-based  | intfloat/e5-mistral-7b-instruct |

## Multimodal Language Models

### Vision-Language Models (Text + Image Generation)
**QEff Auto Class:** `QEFFAutoModelImageTextToText`

| Architecture                | Model Family | Representative Models                  |
|-----------------------------|--------------|----------------------------------------|
| **LlavaForConditionalGeneration** | LLaVA-1.5   | llava-hf/llava-1.5-7b-hf               |
| **MllamaForConditionalGeneration** | Llama 3.2   | meta-llama/Llama-3.2-11B-Vision Instruct<br>meta-llama/Llama-3.2-90B-Vision |

### Audio Models
(Automatic Speech Recognition) - Transcription Task
**QEff Auto Class:** `QEFFAutoModelForSpeechSeq2Seq`

| Architecture | Model Family | Representative Models                                                                 |
|--------------|--------------|----------------------------------------------------------------------------------------|
| **Whisper**  | Whisper      | openai/whisper-tiny<br>openai/whisper-base<br>openai/whisper-small<br>openai/whisper-medium<br>openai/whisper-large<br>openai/whisper-large-v3-turbo |

(models_coming_soon)=
# Models Coming Soon

| Architecture            | Model Family | Representative Models                      |
|-------------------------|--------------|--------------------------------------------|
| **BaichuanForCausalLM** | Baichuan2    | baichuan-inc/Baichuan2-7B-Base             |
| **CohereForCausalLM**   | Command-R    | CohereForAI/c4ai-command-r-v01             |
| **DbrxForCausalLM**     | DBRX         | databricks/dbrx-base                       |