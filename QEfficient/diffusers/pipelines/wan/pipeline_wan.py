# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union
from venv import logger

import numpy as np
import torch
import regex as re

from diffusers import WanPipeline
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.video_processor import VideoProcessor
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

from transformers import AutoTokenizer, UMT5EncoderModel

from QEfficient.diffusers.pipelines.pipeline_utils import  QEffTextEncoder, QEffClipTextEncoder, QEffVAE, QEffWanTransformerModel
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import constants

class QEFFWanPipeline(WanPipeline):
    _hf_auto_class = WanPipeline

    r"""
    Pipeline for text-to-video generation using Wan.
    A QEfficient-optimized Wan pipeline, inheriting from `diffusers.WanPipeline`.

    This class integrates QEfficient components (e.g., optimized models for umt5 text encoders,
    wan transformer, and VAE) to enhance performance, particularly for deployment on Qualcomm AI hardware.
    It provides methods for text-to-video generation leveraging these optimized components.
    """

    # model_cpu_offload_seq = "text_encoder->transformer->transformer_2->vae"
    # _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    # _optional_components = ["transformer", "transformer_2"]
            # self,
        # tokenizer: AutoTokenizer,
        # text_encoder: UMT5EncoderModel,
        # vae: AutoencoderKLWan,
        # scheduler: FlowMatchEulerDiscreteScheduler,
        # transformer: Optional[WanTransformer3DModel] = None,
        # transformer_2: Optional[WanTransformer3DModel] = None,
        # boundary_ratio: Optional[float] = None,
        # expand_timesteps: bool = False,  # Wan2.2 ti2v

    def __init__(self, model, *args, **kwargs):
        # Required by diffusers for serialization and device management
        self.model = model
        self.args = args
        self.kwargs = kwargs

        self.text_encoder = model.text_encoder   ##TODO : update with  Qeff umt5
        self.transformer = QEffWanTransformerModel(model.transformer)
        self.transformer_2 = QEffWanTransformerModel(model.transformer_2) # only for wan 14B ##TODO: check for wan5B
        self.vae_decode = model.vae  ##TODO: QEffVAE(model, "decoder")
        self.tokenizer = model.tokenizer
        self.text_encoder.tokenizer = model.tokenizer
        self.scheduler = model.scheduler

        self.register_modules(
            vae=self.vae_decode,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer ,
            scheduler=self.scheduler,
            transformer_2=self.transformer_2,
        )
        # TODO: for wan 5 B 
        # boundary_ratio = self.kwargs.get("boundary_ratio", None)
        # expand_timesteps = self.kwargs.get("expand_timesteps", True) ##TODO :  not used this part of code in onboarding
        ## for wan 14 B
        boundary_ratio = 0.875
        expand_timesteps = self.kwargs.get("expand_timesteps", False)
        self.register_to_config(boundary_ratio=boundary_ratio)
        self.register_to_config(expand_timesteps=expand_timesteps)
        self.vae_scale_factor_temporal = self.vae_decode.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae_decode.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        """
        Instantiate a QEFFWanTransformer3DModel from pretrained Diffusers models.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                The path to the pretrained model or its name.
            **kwargs (additional keyword arguments):
                Additional arguments that can be passed to the underlying `StableDiffusion3Pipeline.from_pretrained`
                method.
        """
        model = cls._hf_auto_class.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float32,
            **kwargs,
        )
        model.to("cpu")
        return cls(model, pretrained_model_name_or_path)

    @property
    def components(self):
        return {
            "text_encoder": self.text_encoder,
            "transformer": self.transformer,
            "transformer_2":self.transformer_2, #TODO: for wan 5B getattr(self, "transformer_2", None),
            "vae": self.vae_decode,
            "tokenizer": self.tokenizer,
            "scheduler": self.scheduler,
        }

    def export(self, export_dir: Optional[str] = None) -> str:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.

        ``Optional`` Args:
           :export_dir (str, optional): The directory path to store ONNX-graph.

        Returns:
            :str: Path of the generated ``ONNX`` graph.
        """

        # text_encoder - umt5 ##TODO: update once umt5 modeling is available
        # example_inputs_text_encoder, dynamic_axes_text_encoder, output_names_text_encoder = (
        #     self.text_encoder.get_onnx_config(seq_len = self.tokenizer.model_max_length)
        # )
        # self.text_encoder.export(
        #     inputs=example_inputs_text_encoder,
        #     output_names=output_names_text_encoder,
        #     dynamic_axes=dynamic_axes_text_encoder,
        #     export_dir=export_dir,
        # )

        # transformers
        example_inputs_transformer, dynamic_axes_transformer, output_names_transformer = (
            self.transformer.get_onnx_config(batch_size=1, seq_length=512, cl=3840, latent_height=24, latent_width=40)
        )
        self.transformer.export(
            inputs=example_inputs_transformer,
            output_names=output_names_transformer,
            dynamic_axes=dynamic_axes_transformer,
            export_dir=export_dir,
        )
        # transformers #TODO only for wan 14B
        example_inputs_transformer, dynamic_axes_transformer, output_names_transformer = (
            self.transformer_2.get_onnx_config(batch_size=1, seq_length=512, cl=3840, latent_height=24, latent_width=40)
        )
        self.transformer_2.export(
            inputs=example_inputs_transformer,
            output_names=output_names_transformer,
            dynamic_axes=dynamic_axes_transformer,
            export_dir=export_dir,
        )

        # vae
        # example_inputs_vae, dynamic_axes_vae, output_names_vae = self.vae_decode.get_onnx_config()
        # self.vae_decoder_onnx_path = self.vae_decode.export(
        #     example_inputs_vae,
        #     output_names_vae,
        #     dynamic_axes_vae,
        #     export_dir=export_dir,
        # )
        return str(self.transformer.onnx_path),str(self.transformer_2.onnx_path)


    def compile(
        self,
        onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        *,
        seq_len: Union[int, List[int]] = 512,
        batch_size: int = 1,
        num_devices_text_encoder: int = 1,
        num_devices_transformer: int = 16,
        num_devices_transformer_2: int = 16,
        num_devices_vae_decoder: int = 1,
        num_cores: int = 16,  # FIXME: Make this mandatory arg
        mxfp6_matmul: bool = False,
        **compiler_options,
    ) -> str:
        """
        Compiles the ONNX graphs of the different model components for deployment on Qualcomm AI hardware.
        """
        if any(
            path is None
            for path in [
                # self.text_encoder.onnx_path,
                self.transformer.onnx_path,
                self.transformer_2.onnx_path,
                # self.vae_decode.onnx_path,
            ]
        ):
            self.export()
        # text_encoder - umt5
        # specializations_text_encoder = self.text_encoder.get_specializations(
        #     batch_size, self.tokenizer.model_max_length
        # )

        # self.text_encoder_compile_path = self.text_encoder._compile(
        #     onnx_path,
        #     compile_dir,
        #     compile_only=True,
        #     specializations=specializations_text_encoder,
        #     convert_to_fp16=True,
        #     mxfp6_matmul=mxfp6_matmul,
        #     mdp_ts_num_devices=num_devices_text_encoder,
        #     aic_num_cores=num_cores,
        #     **compiler_options,
        # )


        # transformer
        # import pdb; pdb.set_trace()
        specializations_transformer = self.transformer.get_specializations(batch_size, seq_len, latent_height=24, latent_width=40, cl=3840)
        compiler_options = {"mos": 1, "mdts-mos":1}
        self.trasformer_compile_path = self.transformer._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            specializations=specializations_transformer,
            convert_to_fp16=True,
            mxfp6_matmul=True,
            mdp_ts_num_devices=num_devices_transformer,
            aic_num_cores=num_cores,
            **compiler_options,
        )
        self.trasformer_2_compile_path = self.transformer_2._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            specializations=specializations_transformer,
            convert_to_fp16=True,
            mxfp6_matmul=True,
            mdp_ts_num_devices=num_devices_transformer_2,
            aic_num_cores=num_cores,
            **compiler_options,
        )
        # vae
        # specializations_vae = self.vae_decode.get_specializations(batch_size)
        # self.vae_decoder_compile_path = self.vae_decode._compile(
        #     onnx_path,
        #     compile_dir,
        #     compile_only=True,
        #     specializations=specializations_vae,
        #     convert_to_fp16=True,
        #     mdp_ts_num_devices=num_devices_vae_decoder,
        # )

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 61, #81
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""The call function to the pipeline for generation. """

        # if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        #     callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )
        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        # print(f"[DEBUG]>>>>>>>>>>>>> self._guidance_scale ; {self._guidance_scale}; self._guidance_scale_2 : {self._guidance_scale_2}, self.do_classifier_free_guidance ; {self.do_classifier_free_guidance}")
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.model.dtype  # if self.transformer is not None else self.transformer_2.dtype update it to self.transformer_2.model.dtype for 14 B
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = (
            self.transformer.model.config.in_channels
            # if self.transformer is not None
            # else self.transformer_2.model.config.in_channels
        )

        # import pdb; pdb.set_trace()
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        # 6. Denoising loop
        ###### AIC related changes of transformers ######
        if self.transformer.qpc_session is None:
            self.transformer.qpc_session = QAICInferenceSession(str(self.trasformer_compile_path)) #, device_ids=device_ids_transformer)
        if self.transformer_2.qpc_session is None:
            self.transformer_2.qpc_session = QAICInferenceSession(str(self.trasformer_2_compile_path)) #, device_ids=device_ids_transformer)

        if height == 480:
            cl = 24960 # for wan 14 B 480 P  #TODO: calculate on fly CL based on H, W
            # print('>>>>>>>>>>>>>>>>>>> 480P')
        else:
            cl = 3840

        output_buffer = {
            "output": np.random.rand(
                batch_size, cl, 64
                ##TODO: check for wan 5B : 6240, 192 #self.transformer.model.config.joint_attention_dim , self.transformer.model.config.in_channels
            ).astype(np.int32),
        }
        self.transformer.qpc_session.set_buffers(output_buffer)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    # wan2.1 or high-noise stage in wan2.2
                    current_model = self.transformer.model
                    current_model_qpc = self.transformer.qpc_session
                    current_guidance_scale = guidance_scale
                else:
                    ## TODO : not available for wan 5B
                    # low-noise stage in wan2.2
                    current_model = self.transformer_2.model
                    current_model_qpc = self.transformer_2.qpc_session
                    current_guidance_scale = guidance_scale_2
                    print(f">>>>>>>>>>> {i}: low noise")

                latent_model_input = latents.to(transformer_dtype)
                print(f"latent_model_input to DIT : {latent_model_input}")
                if self.config.expand_timesteps:
                    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                batch_size, num_channels, num_frames, height, width = latents.shape #  modeling
                p_t, p_h, p_w = self.transformer.model.config.patch_size
                post_patch_num_frames = num_frames // p_t
                post_patch_height = height // p_h
                post_patch_width = width // p_w

                # patch_states = self.transformer.patch_embedding(latent_model_input)
                # import pdb; pdb.set_trace()
                rotary_emb = self.transformer.model.rope(latent_model_input)
                rotary_emb = torch.cat(rotary_emb, dim=0)
                ts_seq_len = None
                # ts_seq_len = timestep.shape[1]
                timestep = timestep.flatten()

                temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.transformer.model.condition_embedder(
                    timestep, prompt_embeds, encoder_hidden_states_image=None, timestep_seq_len=ts_seq_len
                )
                if self.do_classifier_free_guidance:
                    temb, timestep_proj, encoder_hidden_states_neg, encoder_hidden_states_image = self.transformer.model.condition_embedder(
                        timestep, negative_prompt_embeds, encoder_hidden_states_image=None, timestep_seq_len=ts_seq_len
                    )
                # timestep_proj = timestep_proj.unflatten(2, (6, -1)) # for 5 B new_app.py ##TODO: cross check once
                timestep_proj = timestep_proj.unflatten(1, (6, -1))
                # import pdb; pdb.set_trace()
                inputs_aic = {
                    "hidden_states": latents.detach().numpy(),
                    "encoder_hidden_states": encoder_hidden_states.detach().numpy(),
                    "rotary_emb": rotary_emb.detach().numpy(),
                    "temb": temb.detach().numpy(),
                    "timestep_proj": timestep_proj.detach().numpy()
                }
                if self.do_classifier_free_guidance:
                    inputs_aic2 = {
                        "hidden_states": latents.detach().numpy(),
                        "encoder_hidden_states": encoder_hidden_states_neg.detach().numpy(),
                        "rotary_emb": rotary_emb.detach().numpy(),
                        "temb": temb.detach().numpy(),
                        "timestep_proj": timestep_proj.detach().numpy()
                    }

                # import pdb; pdb.set_trace()

                with current_model.cache_context("cond"):
                    ########### pytorch
                    # noise_pred_torch = current_model(
                    #     hidden_states=latent_model_input,
                    #     # timestep=timestep,
                    #     encoder_hidden_states=encoder_hidden_states,
                    #     rotary_emb=rotary_emb,
                    #     temb=temb,
                    #     timestep_proj=timestep_proj,
                    #     attention_kwargs=attention_kwargs,
                    #     return_dict=False,
                    # )[0]

                    start_time = time.time()
                    outputs = current_model_qpc.run(inputs_aic)
                    end_time = time.time()
                    print(f"Time : {end_time - start_time:.2f} seconds")

                    # noise_pred = torch.from_numpy(outputs["output"])
                    hidden_states = torch.tensor(outputs["output"])

                    hidden_states = hidden_states.reshape(
                        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
                    )

                    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
                    noise_pred = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


                if self.do_classifier_free_guidance: # for Wan lighting CFG is False
                    with current_model.cache_context("uncond"):
                        ############ pytorch
                        # noise_uncond_pytorch = current_model(
                        #     hidden_states=latent_model_input,
                        #     timestep=timestep,
                        #     encoder_hidden_states=negative_prompt_embeds,
                        #     attention_kwargs=attention_kwargs,
                        #     return_dict=False,
                        # )[0]
                        start_time = time.time()
                        outputs = current_model_qpc.run(inputs_aic2)
                        end_time = time.time()
                        print(f"Time : {end_time - start_time:.2f} seconds")

                        # noise_uncond = torch.from_numpy(outputs["output"])
                        hidden_states = torch.tensor(outputs["output"])

                        hidden_states = hidden_states.reshape(
                            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
                        )

                        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
                        noise_uncond = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # if callback_on_step_end is not None:
                #     callback_kwargs = {}
                #     for k in callback_on_step_end_tensor_inputs:
                #         callback_kwargs[k] = locals()[k]
                #     callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                #     latents = callback_outputs.pop("latents", latents)
                #     prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                #     negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                # if XLA_AVAILABLE:
                #     xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae_decode.model.dtype)
            latents_mean = (
                torch.tensor(self.vae_decode.model.config.latents_mean)
                .view(1, self.vae_decode.model.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae_decode.model.config.latents_std).view(1, self.vae_decode.model.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            # import pdb; pdb.set_trace()
            video = self.model.vae.decode(latents, return_dict=False)[0] #TODO: to enable aic with qpc self.vae_decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video.detach())
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)