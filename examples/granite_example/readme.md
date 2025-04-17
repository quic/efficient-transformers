# InternVL Inference
This directory contains an example script of how to run inference on Granite-vision-3.2-2b via QEFFAutoModelForCausalLM class.

Currently for this model we will support dual pcs. No CB support is there for this model.

The model expects the following inputs to be fixed. Please reshape any given image to (w x h) (1610 x 1109) and than pass it to the processor.It accepts a path or a url. Please pass jpg images.

1. Image Size Height =1109
2. Image Size Width =1610
3. Num Patches= 10


To run example script after package installations:
```sh
python granite_vision_inference.py
```

Expected output for given sample inputs in the script:
```sh
The highest scoring model on ChartQA is Granite Vision with a score of 0.87.
```