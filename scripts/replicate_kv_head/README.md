# KV-Head Replication

This example contains a sample script for replicating key-value (KV) heads for the Llama-3-8B-Instruct model. The script performs the following steps:
1. Runs inference with the original model.
2. Replicates the KV heads.
3. Runs inference on the modified model to validate the changes.
4. Exports the modified model to ONNX format.


## Installation

Install efficient-transformers and the required libraries using https://github.com/quic/efficient-transformers#quick-installation


## Usage
You can run the script with different parameters using the command line. Below is an example of how to use the script:

1. **(Optional)** If you are using a gated repository, export the `HF_TOKEN`:
    ```sh
    export HF_TOKEN=<hf_token>
    ```

2. **Run the script** with the desired parameters:
    ```sh
    python script.py --model_name "meta-llama/Meta-Llama-3-8B-Instruct" --prompt "Hello, world!" --repeat 3
    ```

Replace `<hf_token>` with your actual token.

### Arguments
- **--model_name**: Model card name to use (default: “meta-llama/Meta-Llama-3-8B-Instruct”).
- **--prompt**: Prompt to use for the model (default: “My name is”).
- **--repeat**: Factor to repeat key-value heads (default: 2).