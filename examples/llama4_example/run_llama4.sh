#!/bin/bash

PROMPT="What is happening in the video"
MODEL_NAME=""
HF_TOKEN=""
QPC_VISION=""
QPC_TEXT=""
DEVICE_ID_VISION="24,25,26,27"
DEVICE_ID_TEXT="28,29,30,31"
GENERATION_LEN=256
VIDEO_PATH=""
OUTPUT_DIR=""

# Run the Python script
python ./llama4_multi_image_inference.py \
  --model "${MODEL_NAME}" \
  --hf-token "${HF_TOKEN}" \
  --qpc-vision "${QPC_VISION}" \
  --qpc-text "${QPC_TEXT}" \
  --prompt "${PROMPT}" \
  --device-id-vision ${DEVICE_ID_VISION} \
  --device-id-text ${DEVICE_ID_TEXT} \
  --generation-len ${GENERATION_LEN} \
  --video-path "${VIDEO_PATH}" \
  --output-dir "${OUTPUT_DIR}"
