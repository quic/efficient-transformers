# Granite Vision Inference
This directory contains an example script of how to run inference on Granite-vision-3.2-2b via QEFFAutoModelForCausalLM class.

Currently for this model we will support dual pcs. No CB support is there for this model.

The model expects the following inputs to be fixed. 

1. Image Size Height =1109
2. Image Size Width =1610
3. Num Patches= 10

Please reshape any given image to (w x h) (1610 x 1109) and than pass it to the processor.It accepts a path or a url. Please pass jpg images.

Image used:

http://images.cocodataset.org/val2017/000000039769.jpg


To run example script after package installations:
```sh
python granite_vision_inference.py
```

Expected output for given sample inputs in the script:
```sh
The image depicts two cats lying on a pink blanket that is spread out on a red couch. The cats are positioned in a relaxed manner, with their bodies stretched out and their heads resting on the blanket. 
The cat on the left is a smaller, tabby cat with a mix of black, gray, and white fur. It has a long, slender body and a distinctive tail that is curled up near its tail end. The cat on the right is a larger, 
tabby cat with a mix of gray, black, and brown fur. It has
```