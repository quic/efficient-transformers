# Speech Seq2Seq
This directory contains an example script of how to use the AutoModelForSpeechSeq2Seq class. (for now, Whisper models on audio <30 seconds only has been validated)

## Required packages:
- `librosa==0.10.2`
- `soundfile==0.13.1`

You can install them using pip:
```sh
pip install librosa==0.10.2 soundfile==0.13.1
```

To run example script after package installations:
```sh
python speech_seq2seq_models.py
```

Expected output for given data sample:
```sh
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.<|endoftext|>
```