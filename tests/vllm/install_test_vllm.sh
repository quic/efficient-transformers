#!/bin/bash

#################### run_fasttest_qaic.sh ##############################
# goal: To quickly run through 3 offline tests to check if system works
######################################################################## 

# Clone the vLLM repo, and apply the patch for qaic backend support
git clone https://github.com/vllm-project/vllm.git

cd vllm

git checkout v0.6.0

git apply /opt/qti-aic/integrations/vllm/qaic_vllm.patch

# Set environment variables and install
export VLLM_TARGET_DEVICE="qaic"

pip install -e .

pip install pytest
#### T1 ####

pytest --disable-warnings -s -v test_qaic_output_consistency.py::test_output_consistency \
--model-name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
--seq-len 128 \
--ctx-len 256 \
--decode-bsz 4 \
--dtype "mxfp6" \
--kv-dtype "mxint8" \
--device-group 1 \


pytest --disable-warnings -s -v test_qaic_output_consistency.py::test_generate \
--model-name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
--seq-len 128 \
--ctx-len 256 \
--decode-bsz 4 \
--dtype "mxfp6" \
--kv-dtype "mxint8" \
--device-group 1 \

pytest --disable-warnings -s -v test_qaic_output_consistency.py::test_generated_tokens \
--model-name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
--seq-len 128 \
--ctx-len 256 \
--decode-bsz 4 \
--dtype "mxfp6" \
--kv-dtype "mxint8" \
--device-group 1 \

