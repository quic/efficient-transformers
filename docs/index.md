# Welcome to Efficient-Transformers Documentation!

## Release Documents

- [Efficient Transformer Library - 1.21.0 Release Notes](source/release_docs.md)
    - [Newly Supported Models](source/release_docs.md#newly-supported-models)
    - [Key Features & Enhancements](source/release_docs.md#key-features-enhancements)
    - [Embedding Model Upgrades](source/release_docs.md#embedding-model-upgrades)
    - [Fine-Tuning Support](source/release_docs.md#fine-tuning-support)

- [Efficient Transformer Library - 1.20.0 Release Notes](source/release_docs.md#efficient-transformer-library-1200-release-notes)
    - [Newly Supported Models](source/release_docs.md#newly-supported-models_1)
    - [Key Features & Enhancements](source/release_docs.md#key-features-enhancements_1)
    - [Embedding Model Upgrades](source/release_docs.md#embedding-model-upgrades_1)
    - [Fine-Tuning Support](source/release_docs.md#fine-tuning-support_1)

## Getting Started

- [Introduction Qualcomm efficient-transformers library](source/introduction.md)
- [Supported Features](source/supported_features.md)
- [Validated Models](source/validate.md)
- [Models Coming Soon](source/validate.md#models-coming-soon)

## Installation

- [Pre-requisites](source/installation.md#pre-requisites)
- [Installation](source/installation.md#installation)
- [Sanity Check](source/installation.md#sanity-check)

## Inference on Cloud AI 100

- [Quick Start](source/quick_start.md)
    - [Transformed models and QPC storage](source/quick_start.md#transformed-models-and-qpc-storage)
    - [Command Line Interface Execution](source/quick_start.md#command-line-interface-execution)
        - [Inference](source/quick_start.md#inference)
        - [Export](source/quick_start.md#export)
        - [Compile](source/quick_start.md#compile)
        - [Execute](source/quick_start.md#execute)
        - [Infer](source/quick_start.md#infer)
    - [QEFF Auto Class Execution](source/quick_start.md#qeff-auto-class-execution)
        - [1. Model download and Optimize for Cloud AI 100](source/quick_start.md#model-download-and-optimize-for-cloud-ai-100)
        - [2. Export and Compile with one API](source/quick_start.md#export-and-compile-with-one-api)
        - [3. Execute](source/quick_start.md#execute_1)
    - [Local Model Execution](source/quick_start.md#local-model-execution)

- [Features Enablement Guide](source/features_enablement.md)
    - [Continuous Batching](source/features_enablement.md#continuous-batching)
    - [Multi-Qranium Inference](source/features_enablement.md#multi-qranium-inference)
    - [QNN Compilation via Python API](source/features_enablement.md#qnn-compilation-via-python-api)
    - [Draft-Based Speculative Decoding](source/features_enablement.md#draft-based-speculative-decoding)

## API Reference

- [QEfficient Auto Classes](source/qeff_autoclasses.md)
    - [QEFFAutoModelForCausalLM](source/qeff_autoclasses.md#QEFFAutoModelForCausalLM)
    - [QEFFAutoModel](source/qeff_autoclasses.md#QEFFAutoModel)
    - [QEffAutoPeftModelForCausalLM](source/qeff_autoclasses.md#QEffAutoPeftModelForCausalLM)
    - [QEffAutoLoraModelForCausalLM](source/qeff_autoclasses.md#QEffAutoLoraModelForCausalLM)
    - [QEFFAutoModelForImageTextToText](source/qeff_autoclasses.md#QEFFAutoModelForImageTextToText)
    - [QEFFAutoModelForSpeechSeq2Seq](source/qeff_autoclasses.md#QEFFAutoModelForSpeechSeq2Seq)
    - [QEFFAutoModelForCTC](source/qeff_autoclasses.md#QEFFAutoModelForCTC)

- [Diffuser Classes](source/diffuser_classes.md)
    - [QEffTextEncoder](source/diffuser_classes.md#QEffTextEncoder)
    - [QEffUNet](source/diffuser_classes.md#QEffUNet)
    - [QEffVAE](source/diffuser_classes.md#QEffVAE)
    - [QEffFluxTransformerModel](source/diffuser_classes.md#QEffFluxTransformerModel)
    - [QEffWanUnifiedTransformer](source/diffuser_classes.md#QEffWanUnifiedTransformer)
    - [QEffWanPipeline](source/diffuser_classes.md#QEffWanPipeline)
    - [QEffWanImageToVideoPipeline](source/diffuser_classes.md#QEffWanImageToVideoPipeline)
    - [QEffFluxPipeline](source/diffuser_classes.md#QEffFluxPipeline)

- [CLI API Reference](source/cli_api.md)
    - [QEfficient.cloud.infer](source/cli_api.md#infer_api)
    - [QEfficient.cloud.execute](source/cli_api.md#execute_api)
    - [QEfficient.compile](source/cli_api.md#compile_api)
    - [QEfficient.cloud.export](source/cli_api.md#export_api)
    - [QEfficient.cloud.finetune](source/cli_api.md#finetune_api)

## QAIC Finetune

- [Finetune Infra](source/finetune.md)
    - [Expose QAIC accelerator devices](source/finetune.md#expose-qaic-accelerator-devices)
    - [Start Docker container](source/finetune.md#start-docker-container)

## Blogs

- [Train anywhere, Infer on Qualcomm Cloud AI 100](https://www.qualcomm.com/developer/blog/2024/01/train-anywhere-infer-qualcomm-cloud-ai-100)
- [How to Quadruple LLM Decoding Performance with Speculative Decoding (SpD) and Microscaling (MX) Formats on Qualcomm® Cloud AI 100](https://statics.teams.cdn.office.net/evergreen-assets/safelinks/1/atp-safelinks.html)
- [Power-efficient acceleration for large language models – Qualcomm Cloud AI SDK](https://www.qualcomm.com/developer/blog/2023/11/power-efficient-acceleration-large-language-models-qualcomm-cloud-ai-sdk)
- [Qualcomm Cloud AI 100 Accelerates Large Language Model Inference by ~2x Using Microscaling (Mx) Formats](https://www.qualcomm.com/developer/blog/2024/01/qualcomm-cloud-ai-100-accelerates-large-language-model-inference-2x-using-microscaling-mx)
- [Qualcomm Cloud AI Introduces Efficient Transformers: One API, Infinite Possibilities](https://www.qualcomm.com/developer/blog/2024/05/qualcomm-cloud-ai-introduces-efficient-transformers-one-api)

## Reference

- [Qualcomm Cloud AI home](https://www.qualcomm.com/products/technology/processors/cloud-artificial-intelligence)
- [Qualcomm Cloud AI SDK download](https://www.qualcomm.com/products/technology/processors/cloud-artificial-intelligence/cloud-ai-100#Software)
- [Qualcomm Cloud AI API reference](https://quic.github.io/cloud-ai-sdk-pages/latest/API/)
- [User Guide](https://quic.github.io/cloud-ai-sdk-pages/)
- [OCP Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
