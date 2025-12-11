import functools
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage import (
    QwenDoubleStreamAttnProcessor2_0,
    QwenImageTransformer2DModel,
)
from diffusers.utils.constants import USE_PEFT_BACKEND
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers

logger = logging.getLogger(__name__)


def qeff_apply_rotary_emb_qwen(x, freqs_cos, freqs_sin):
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # [B, S, H, D//2, 2]
    x1 = x_reshaped[..., 0]  # [B, S, H, D//2]
    x2 = x_reshaped[..., 1]  # [B, S, H, D//2]

    # Reshape for broadcasting: [S, D//2] -> [1, S, 1, D//2]
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)

    # Apply rotation
    x_out1 = x1 * freqs_cos - x2 * freqs_sin  # Real part
    x_out2 = x1 * freqs_sin + x2 * freqs_cos  # Imaginary part

    # Stack and reshape back
    x_out = torch.stack([x_out1, x_out2], dim=-1)  # [B, S, H, D//2, 2]
    x_out = x_out.flatten(-2)  # [B, S, H, D]
    return x_out.type_as(x)


class QEffQwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1

        # Store cos and sin separately instead of complex numbers
        pos_freqs_list = [
            self.rope_params(pos_index, self.axes_dim[0], self.theta),
            self.rope_params(pos_index, self.axes_dim[1], self.theta),
            self.rope_params(pos_index, self.axes_dim[2], self.theta),
        ]
        self.pos_freqs_cos = torch.cat([f[0] for f in pos_freqs_list], dim=1)
        self.pos_freqs_sin = torch.cat([f[1] for f in pos_freqs_list], dim=1)

        neg_freqs_list = [
            self.rope_params(neg_index, self.axes_dim[0], self.theta),
            self.rope_params(neg_index, self.axes_dim[1], self.theta),
            self.rope_params(neg_index, self.axes_dim[2], self.theta),
        ]
        self.neg_freqs_cos = torch.cat([f[0] for f in neg_freqs_list], dim=1)
        self.neg_freqs_sin = torch.cat([f[1] for f in neg_freqs_list], dim=1)

        self.rope_cache = {}

    @functools.lru_cache(maxsize=None)
    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        freqs_pos_cos = self.pos_freqs_cos.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_pos_sin = self.pos_freqs_sin.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg_cos = self.neg_freqs_cos.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg_sin = self.neg_freqs_sin.split([x // 2 for x in self.axes_dim], dim=1)

        # Frame dimension
        freqs_frame_cos = freqs_pos_cos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        freqs_frame_sin = freqs_pos_sin[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)

        if self.scale_rope:
            freqs_height_cos = torch.cat(
                [freqs_neg_cos[1][-(height - height // 2) :], freqs_pos_cos[1][: height // 2]], dim=0
            )
            freqs_height_sin = torch.cat(
                [freqs_neg_sin[1][-(height - height // 2) :], freqs_pos_sin[1][: height // 2]], dim=0
            )
            freqs_height_cos = freqs_height_cos.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_height_sin = freqs_height_sin.view(1, height, 1, -1).expand(frame, height, width, -1)

            freqs_width_cos = torch.cat(
                [freqs_neg_cos[2][-(width - width // 2) :], freqs_pos_cos[2][: width // 2]], dim=0
            )
            freqs_width_sin = torch.cat(
                [freqs_neg_sin[2][-(width - width // 2) :], freqs_pos_sin[2][: width // 2]], dim=0
            )
            freqs_width_cos = freqs_width_cos.view(1, 1, width, -1).expand(frame, height, width, -1)
            freqs_width_sin = freqs_width_sin.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height_cos = freqs_pos_cos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_height_sin = freqs_pos_sin[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width_cos = freqs_pos_cos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)
            freqs_width_sin = freqs_pos_sin[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs_cos = torch.cat([freqs_frame_cos, freqs_height_cos, freqs_width_cos], dim=-1).reshape(seq_lens, -1)
        freqs_sin = torch.cat([freqs_frame_sin, freqs_height_sin, freqs_width_sin], dim=-1).reshape(seq_lens, -1)

        return freqs_cos.clone().contiguous(), freqs_sin.clone().contiguous()

    def forward(self, video_fhw, txt_seq_lens, device):
        """
        Args:
            video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video
            txt_length: [bs] a list of 1 integers representing the length of the text
        Returns:
            Tuple of (vid_freqs_cos, vid_freqs_sin, txt_freqs_cos, txt_freqs_sin)
        """
        if self.pos_freqs_cos.device != device:
            self.pos_freqs_cos = self.pos_freqs_cos.to(device)
            self.pos_freqs_sin = self.pos_freqs_sin.to(device)
            self.neg_freqs_cos = self.neg_freqs_cos.to(device)
            self.neg_freqs_sin = self.neg_freqs_sin.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs_cos_list = []
        vid_freqs_sin_list = []
        max_vid_index = 0

        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"
            if not torch.compiler.is_compiling():
                if rope_key not in self.rope_cache:
                    self.rope_cache[rope_key] = self._compute_video_freqs(frame, height, width, idx)
                video_freq_cos, video_freq_sin = self.rope_cache[rope_key]
            else:
                video_freq_cos, video_freq_sin = self._compute_video_freqs(frame, height, width, idx)

            video_freq_cos = video_freq_cos.to(device)
            video_freq_sin = video_freq_sin.to(device)
            vid_freqs_cos_list.append(video_freq_cos)
            vid_freqs_sin_list.append(video_freq_sin)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs_cos = self.pos_freqs_cos[max_vid_index : max_vid_index + max_len, ...]
        txt_freqs_sin = self.pos_freqs_sin[max_vid_index : max_vid_index + max_len, ...]

        vid_freqs_cos = torch.cat(vid_freqs_cos_list, dim=0)
        vid_freqs_sin = torch.cat(vid_freqs_sin_list, dim=0)

        return vid_freqs_cos, vid_freqs_sin, txt_freqs_cos, txt_freqs_sin

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        # Return cos and sin separately instead of complex tensor
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        return freqs_cos, freqs_sin


class QEffQwenImageTransformer2DModel(QwenImageTransformer2DModel):
    def __qeff_init__(self):
        self.pos_embed = QEffQwenEmbedRope(theta=10000, axes_dim=list(self.axes_dims_rope), scale_rope=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        frame: torch.Tensor = None,
        height: torch.Tensor = None,
        width: torch.Tensor = None,
        txt_seq_lens: torch.Tensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`QwenTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
                Mask of the input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # breakpoint()
        # Convert scalar tensors to Python integers and create img_shapes list
        if isinstance(frame, torch.Tensor):
            frame = frame.item() if frame.numel() == 1 else int(frame[0])
        if isinstance(height, torch.Tensor):
            height = height.item() if height.numel() == 1 else int(height[0])
        if isinstance(width, torch.Tensor):
            width = width.item() if width.numel() == 1 else int(width[0])

        if not img_shapes:
            img_shapes = [(frame, height, width)]

        # Convert txt_seq_lens to list if it's a tensor
        if isinstance(txt_seq_lens, torch.Tensor):
            txt_seq_lens = txt_seq_lens.tolist()

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )

        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


class QEffQwenDoubleStreamAttnProcessor2_0(QwenDoubleStreamAttnProcessor2_0):
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            # breakpoint()
            # Unpack the 4 tensors (cos and sin for both img and txt)
            img_freqs_cos, img_freqs_sin, txt_freqs_cos, txt_freqs_sin = image_rotary_emb

            img_query = qeff_apply_rotary_emb_qwen(img_query, img_freqs_cos, img_freqs_sin)
            img_key = qeff_apply_rotary_emb_qwen(img_key, img_freqs_cos, img_freqs_sin)
            txt_query = qeff_apply_rotary_emb_qwen(txt_query, txt_freqs_cos, txt_freqs_sin)
            txt_key = qeff_apply_rotary_emb_qwen(txt_key, txt_freqs_cos, txt_freqs_sin)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Compute joint attention
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output
