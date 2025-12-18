Supported Features
===================
.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Feature
     - Impact
   * - `Compute Context Length (CCL) <https://github.com/quic/efficient-transformers/blob/main/examples/performance/compute_context_length/README.md>`_
     - Optimizes inference by using different context lengths during prefill and decode phases, reducing memory footprint and computation for shorter sequences while maintaining support for longer contexts. Supports both text-only and vision-language models. Refer `sample script <https://github.com/quic/efficient-transformers/blob/main/examples/performance/compute_context_length/basic_inference.py>`_ for more **details**.
   * - Sentence embedding, Flexible Pooling configuration and compilation with multiple sequence lengths
     - Supports standard/custom pooling with AI 100 acceleration and sentence embedding. Enables efficient sentence embeddings via Efficient-Transformers. Compile with one or multiple seq_len; optimal graph auto-selected at runtime. Refer `sample script <https://github.com/quic/efficient-transformers/blob/main/examples/embeddings/sentence_embeddings.py>`_ for more **details**.
   * - `SpD, multiprojection heads <https://quic.github.io/efficient-transformers/source/quick_start.html#draft-based-speculative-decoding>`_
     - Implemented post-attention hidden size projections to speculate tokens ahead of the base model. Refer `sample script <https://github.com/quic/efficient-transformers/blob/main/examples/performance/speculative_decoding/multi_projection.py>`_ for more **details**.
   * - `QNN Compilation support <https://github.com/quic/efficient-transformers/pull/374>`_
     - Enabled for AutoModel classes QNN compilation capabilities for multi-models, embedding models and causal models.
   * - `Disaggregated serving <https://github.com/quic/efficient-transformers/pull/365>`_
     - It support for separate prefill and decode compilation for encoder (vision) and language models.
   * - `GGUF model execution <https://github.com/quic/efficient-transformers/pull/368>`_
     - Supported GGUF model execution (without quantized weights). Refer `sample script <https://github.com/quic/efficient-transformers/blob/main/examples/text_generation/gguf_models.py>`_ for more **details**.
   * - Replication of KV
     - Enabled FP8 model support on `replicate_kv_heads script <https://github.com/quic/efficient-transformers/tree/main/scripts/replicate_kv_head>`_.
   * - `gradient checkpointing <https://github.com/quic/efficient-transformers/pull/338>`_
     - Supports gradient checkpointing in the finetuning script
   * - Swift KV `Snowflake/Llama-3.1-SwiftKV-8B-Instruct <https://huggingface.co/Snowflake/Llama-3.1-SwiftKV-8B-Instruct>`_
     - Reduces computational overhead during inference by optimizing key-value pair processing, leading to improved throughput. Support for both `continuous and non-continuous batching execution <https://github.com/quic/efficient-transformers/pull/367>`_ in SwiftKV
   * - :ref:`Vision Language Model <QEFFAutoModelForImageTextToText>`
     - Provides support for the AutoModelForImageTextToText class from the transformers library, enabling advanced vision-language tasks. Refer `sample script <https://github.com/quic/efficient-transformers/blob/main/examples/image_text_to_text/basic_vlm_inference.py>`_ for more **details**.
   * - :ref:`Speech Sequence to Sequence Model <QEFFAutoModelForSpeechSeq2Seq>`
     - Provides support for the QEFFAutoModelForSpeechSeq2Seq Facilitates speech-to-text sequence models. Refer `sample script <https://github.com/quic/efficient-transformers/blob/main/examples/audio/speech_to_text.py>`_ for more **details**.
   * - Support for FP8 Execution
     - Enables execution with FP8 precision, significantly improving performance and reducing memory usage for computational tasks.
   * - Prefill caching
     - Enhances inference speed by caching key-value pairs for shared prefixes, reducing redundant computations and improving efficiency.
   * - On Device Sampling
     - Enables sampling operations to be executed directly on the QAIC device rather than the host CPU for QEffForCausalLM models. This enhancement significantly reduces host-device communication overhead and improves inference throughput and scalability. Refer `sample script <https://github.com/quic/efficient-transformers/blob/main/examples/on_device_sampling.py>`_ for more **details**.
   * - Prompt-Lookup Decoding
     - Speeds up text generation by using overlapping parts of the input prompt and the generated text, making the process faster without losing quality. Refer `sample script <https://github.com/quic/efficient-transformers/blob/main/examples/performance/speculative_decoding/prompt_lookup.py>`_ for more **details**.
   * - :ref:`PEFT LoRA support <QEffAutoPeftModelForCausalLM>`
     - Enables parameter-efficient fine-tuning using low-rank adaptation techniques, reducing the computational and memory requirements for fine-tuning large models. Refer `sample script <https://github.com/quic/efficient-transformers/blob/main/examples/peft/single_adapter.py>`_ for more **details**.
   * - :ref:`QNN support <id-qnn-compilation-via-python-api>`
     - Enables compilation using QNN SDK, making Qeff adaptable for various backends in the future.
   * - :ref:`Embedding model support <QEFFAutoModel>`
     - Facilitates the generation of vector embeddings for retrieval tasks.
   * - :ref:`Speculative Decoding <id-draft-based-speculative-decoding>`
     - Accelerates text generation by using a draft model to generate preliminary predictions, which are then verified by the target model, reducing latency and improving efficiency. Refer `sample script <https://github.com/quic/efficient-transformers/blob/main/examples/performance/speculative_decoding/draft_based.py>`_ for more **details**.
   * - :ref:`Finite lorax <QEffAutoLoraModelForCausalLM>`
     - Users can activate multiple LoRA adapters and compile them with the base model. At runtime, they can specify which prompt should use which adapter, enabling mixed adapter usage within the same batch. Refer `sample script <https://github.com/quic/efficient-transformers/blob/main/examples/peft/multi_adapter.py>`_ for more **details**.
   * - Python and CPP Inferencing API support
     - Provides flexibility while running inference with Qeff and enabling integration with various applications and improving accessibility for developers. Refer `sample script <https://github.com/quic/efficient-transformers/blob/main/examples/performance/cpp_execution/text_inference_cpp.py>`_ for more **details**.
   * - :ref:`Continuous batching <id-continuous-batching>`
     - Optimizes throughput and latency by dynamically batching requests, ensuring efficient use of computational resources.
   * - AWQ and GPTQ support
     - Supports advanced quantization techniques, improving model efficiency and performance on AI 100.
   * - Support serving successive requests in same session
     - An API that yields tokens as they are generated, facilitating seamless integration with various applications and enhancing accessibility for developers.
   * - Perplexity calculation
     - A script for computing the perplexity of a model, allowing for the evaluation of model performance and comparison across different models and datasets. Refer `sample script <https://github.com/quic/efficient-transformers/blob/main/scripts/perplexity_computation/calculate_perplexity.py>`_ for more **details**.
   * - KV Heads Replication Script
     - A sample script for replicating key-value (KV) heads for the Llama-3-8B-Instruct model, running inference with the original model, replicating KV heads, validating changes, and exporting the modified model to ONNX format. Refer `sample script <https://github.com/quic/efficient-transformers/blob/main/scripts/replicate_kv_head/replicate_kv_heads.py>`_ for more **details**.
   * - Block Attention (in progress)
     - Reduces inference latency and computational cost by dividing context into blocks and reusing key-value states, particularly useful in RAG.
