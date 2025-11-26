import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from QEfficient import QEFFAutoModelForImageTextToText


class SketchDecoder(nn.Module):
    """
    Autoregressive generative model
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.vocab_size = 196042
        self.bos_token_id = 151643
        self.eos_token_id = 196041
        self.pad_token_id = 151643

        config = AutoConfig.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            vocab_size=self.vocab_size,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

        self.transformer = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", config=config, attn_implementation="eager", ignore_mismatched_sizes=True
        )

        self.transformer.resize_token_embeddings(self.vocab_size)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward pass not included in open-source version")


model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
sketch_decoder = SketchDecoder()
weight_path = "/home/dipankar/omnisvg/OmniSVG"
sketch_weight_file = os.path.join(weight_path, "pytorch_model.bin")
if not os.path.exists(sketch_weight_file):
    raise FileNotFoundError(f"pytorch_model.bin not found in {weight_path}")
sketch_decoder.load_state_dict(torch.load(sketch_weight_file))
sketch_decoder.transformer.eval()
qeff_model = QEFFAutoModelForImageTextToText(sketch_decoder.transformer)
qeff_model.export()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", padding_side="left")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", padding_side="left")
path = qeff_model.compile(
    batch_size=1,
    prefill_seq_len=128,
    ctx_len=4096,
    num_cores=16,
    num_devices=8,
    height=354,
    width=536,
    mxfp6_matmul=False,
    aic_enable_depth_first=True,
    skip_vision=True,
    mos=1,
)
