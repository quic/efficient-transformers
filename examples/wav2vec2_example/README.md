# Speech Recognition with Wav2Vec2
This directory contains an example script of how to use the AutoModelForCTC class. (for now, Wav2Vec2 models on audio <30 seconds only has been validated)

## Required packages:
- `librosa==0.10.2`
- `soundfile==0.13.1`

You can install them using pip:
```sh
pip install librosa==0.10.2 soundfile==0.13.1
```

To run example script after package installations:
```sh
python run_wav2vec2_inference.py
```

Expected output for given data sample:
```sh
MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
```