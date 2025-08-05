import os
from typing import List, Optional, Union

import numpy as np
import torch

from diffusers import StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from QEfficient.diffusers.pipelines.pipeline_utils import QEffSafetyChecker, QEffTextEncoder, QEffUNet, QEffVAE
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import constants


class QEFFStableDiffusionPipeline(StableDiffusionPipeline):
    _hf_auto_class = StableDiffusionPipeline

    def __init__(self, model, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.tokenizer = model.tokenizer
        self.scheduler = model.scheduler
        self.feature_extractor = model.feature_extractor

        self.text_encoder = QEffTextEncoder(model)
        self.unet = QEffUNet(model)

        # VAE Encoder
        self.vae_encoder = QEffVAE(model, "encoder")
        self.vae_encoder.model.forward = lambda sample, return_dict: self.vae_encoder.model.encode(sample, return_dict)

        # VAE Decoder
        self.vae_decoder = QEffVAE(model, "decoder")
        self.vae_decoder.model.forward = lambda latent_sample, return_dict: self.vae_decoder.model.decode(
            latent_sample, return_dict
        )

        # Saftey Checker
        self.safety_checker = QEffSafetyChecker(model)
        self.safety_checker.model.forward = model.safety_checker.forward_onnx

        self.pretrained_model_name_or_path = kwargs.get("pretrained_model_name_or_path", None)

        self.vae_scale_factor = (
            2 ** (len(self.vae.model.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        kwargs.update({"attn_implementation": "eager"})
        model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float32, **kwargs)
        model.to("cpu")
        return cls(model, pretrained_model_name_or_path)

    def export(self, export_dir: Optional[str] = None) -> str:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.

        ``Optional`` Args:
           :export_dir (str, optional): The directory path to store ONNX-graph.

        Returns:
            :str: Path of the generated ``ONNX`` graph.
        """

        # Text encoder export

        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len = self.tokenizer.model_max_length

        example_inputs = {
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int32),
            # "attention_mask": torch.ones((bs, seq_len), dtype=bool),
        }

        dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}, "attention_mask": {0: "batch_size", 1: "seq_len"}}

        output_names = ["last_hidden_state", "pooler_output"]

        # self.text_encoder.model.set_attn_processor(AttnProcessor())

        # config = self.text_encoder.model.text_model.config
        # for layer in self.text_encoder.model.text_model.encoder.layers:
        #     layer.self_attn = CLIPAttention(config)

        self.text_encoder_onnx_path = self.text_encoder.export(
            example_inputs,
            output_names,
            dynamic_axes,
            export_dir=export_dir,
        )

        # UNET Export

        print("######################  Text Encoder Exported #####################")

        unet_example_input = {
            "sample": torch.randn(
                bs, self.unet.model.in_channels, self.unet.model.config.sample_size, self.unet.model.config.sample_size
            ),
            "timestep": torch.tensor([1]),
            "encoder_hidden_states": torch.randn(bs, seq_len, self.unet.model.config.cross_attention_dim),
            "return_dict": False,
        }

        output_names = ["out_sample"]

        dynamic_axes = {
            "sample": {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
            "timestep": {0: "batch_size"},
            "encoder_hidden_states": {0: "batch_size", 1: "seq_len"},
        }
        # self.unet.model.set_attn_processor(AttnProcessor())

        self.unet_onnx_path = self.unet.export(
            unet_example_input,
            output_names,
            dynamic_axes,
            export_dir=export_dir,
        )

        print("######################  UNet Exported #####################")

        vae_encoder_input = {
            "sample": torch.randn(bs, 3, 512, 512),
            "return_dict": False,
        }

        output_names = ["latent_sample"]

        dynamic_axes = {
            "sample": {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
        }

        # self.vae_encoder.model.set_attn_processor(AttnProcessor())

        self.vae_encoder_onnx_path = self.vae_encoder.export(
            vae_encoder_input,
            output_names,
            dynamic_axes,
            export_dir=None,
        )

        print("######################  VAE Encoder Exported #####################")

        vae_decoder_input = {
            "latent_sample": torch.randn(bs, 4, 64, 64),
            "return_dict": False,
        }

        output_names = ["sample"]

        dynamic_axes = {
            "latent_sample": {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
        }

        # self.vae_decoder.model.set_attn_processor(AttnProcessor())

        self.vae_decoder_onnx_path = self.vae_decoder.export(
            vae_decoder_input,
            output_names,
            dynamic_axes,
            export_dir=None,
        )

        print("######################  VAE Decoder Exported #####################")

        saftey_checker_input = {"clip_input": torch.randn(bs, 3, 224, 224), "images": torch.randn(bs, 3, 512, 512)}
        output_names = ["out_images", "has_nsfw_concepts"]

        dynamic_axes = {
            "clip_input": {0: "batch_size", 1: "channels", 2: "clip_height", 3: "clip_width"},
            "images": {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
        }

        # self.safety_checker.model.set_attn_processor(AttnProcessor())

        # for layer in self.safety_checker.model.vision_model.vision_model.encoder.layers:
        #     config = self.safety_checker.model.config.vision_config
        #     layer.self_attn = CLIPAttention(config)
        # Replace with eager version

        self.safety_checker_onnx_path = self.safety_checker.export(
            saftey_checker_input,
            output_names,
            dynamic_axes,
            export_dir=None,
        )

        print("######################  Safety Checker Exported #####################")

    def compile(
        self,
        onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        *,
        seq_len: Union[int, List[int]] = 32,
        batch_size: int = 1,
        num_devices: int = 1,
        num_cores: int = 16,  # FIXME: Make this mandatory arg
        mxfp6_matmul: bool = False,
        **compiler_options,
    ) -> str:
        # Compile text_encoder

        # Make specilization

        seq_len = self.tokenizer.model_max_length

        specializations = [
            {"batch_size": batch_size, "seq_len": seq_len},
        ]

        self.text_encoder_compile_path = self.text_encoder._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            specializations=specializations,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            **compiler_options,
        )

        print("######################  Text Encoder Compiled #####################")

        # Compile unet

        specializations = [
            {
                "batch_size": batch_size,
                "channels": 4,
                "height": self.unet.model.config.sample_size,
                "width": self.unet.model.config.sample_size,
                "seq_len": seq_len,
            }
        ]

        self.compiled_unet_path = self.unet._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            specializations=specializations,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            **compiler_options,
        )

        print("######################  Unet Compiled #####################")

        # Compile vae_encoder

        encoder_specializations = [
            {
                "batch_size": batch_size,
                "channels": self.vae_encoder.model.config.in_channels,
                "height": self.vae_encoder.model.config.sample_size,
                "width": self.vae_encoder.model.config.sample_size,
            }
        ]

        # self.vae_encoder_compile_path=self.vae_encoder._compile(
        #     onnx_path,
        #     compile_dir,
        #     compile_only=True,
        #     specializations=encoder_specializations,
        #     convert_to_fp16=True,
        # )

        print("######################  VAE Encoder Compiled #####################")

        # compile vae decoder

        decoder_sepcializations = [
            {
                "batch_size": batch_size,
                "channels": 4,
                "height": self.vae_decoder.model.config.sample_size,
                "width": self.vae_decoder.model.config.sample_size,
            }
        ]

        # self.vae_decoder_compile_path=self.vae_decoder._compile(
        #     onnx_path,
        #     compile_dir,
        #     compile_only=True,
        #     specializations=decoder_sepcializations,
        #     convert_to_fp16=True,
        # )

        # TODO: Add support of comilation for now it will run on host

        print("######################  VAE Decoder Compiled #####################")

        # compile safety check

        safety_check_specializations = [
            {
                "batch_size": batch_size,
                "channels": 3,
                "height": 512,
                "width": 512,
                "clip_height": 224,
                "clip_width": 224,
            }
        ]

        self.compiled_safety_checker_path = self.safety_checker._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            specializations=safety_check_specializations,
            convert_to_fp16=True,
        )

        print("######################  Safety Checker Compiled #####################")

    # def generate()

    @property
    def model_name(self) -> str:
        pass

    @property
    def model_hash(self) -> str:
        pass

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        device_ids: List[int] = [0],
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        **kwargs,
    ):
        # # Get output for text_encoder
        if self.text_encoder.qpc_session is None:
            self.text_encoder.qpc_session = QAICInferenceSession(str(self.text_encoder_compile_path), device_ids)

        # Dynamic switching to closest seq_Len based on input_ids_len

        # find the inputs/attention mask shape for which qpc compiled
        bs, compield_inputs_shape = self.text_encoder.qpc_session.bindings[0].dims

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )
        text_encoder_output = {
            "last_hidden_state": np.random.rand(bs, 77, 768).astype(np.float32),
            "pooler_output": np.random.rand(bs, 768).astype(np.float32),
        }
        self.text_encoder.qpc_session.set_buffers(text_encoder_output)
        ##  Testing with the ORT output ##

        import onnxruntime as ort

        ort_session = ort.InferenceSession(str(self.text_encoder.onnx_path))

        onnx_inputs = {k: v for k, v in text_inputs.items() if k in [i.name for i in ort_session.get_inputs()]}

        onnx_inputs["input_ids"] = onnx_inputs["input_ids"].astype(np.int32)

        ort_outputs = ort_session.run(None, onnx_inputs)
        text_inputs_pt = {k: torch.from_numpy(v) for k, v in onnx_inputs.items()}

        pt_output = self.text_encoder.model(**text_inputs_pt)
        mad = torch.mean(torch.abs(pt_output[0] - torch.tensor(ort_outputs[0])))
        print("CLIP: MAD onnx vs pytorch", mad)

        self.text_encoder.qpc_session.set_buffers(text_encoder_output)
        ai100_output = self.text_encoder.qpc_session.run(onnx_inputs)
        mad_ai100_onnnx = np.mean(np.abs(ai100_output["last_hidden_state"] - ort_outputs[0]))

        print("CLIP: MAD ai100 vs onnx", mad_ai100_onnnx)

        ai100_output = ai100_output["last_hidden_state"]

        ## CLIP done here
        # 4. Prepare timesteps

        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, timesteps, sigmas)
        timesteps = timesteps.numpy()
        # 5. Prepare latent variables
        # 0. Default height and width to unet
        # timesteps = timesteps.astype(np.float32)

        width = height = self.unet.model.config.sample_size
        height, width = height * self.vae_scale_factor, width * self.vae_scale_factor

        num_channels_latents = self.unet.model.config.in_channels
        latents = self.prepare_latents(
            bs,
            num_channels_latents,
            height,
            width,
            torch.float32,
            generator,
            latents,
        )

        # Load qpc
        self.unet_qpc_session = QAICInferenceSession(str(self.compiled_unet_path), [1])

        unet_output = {"out_sample": np.random.rand(bs, 4, 64, 64).astype(np.float32)}
        self.unet_qpc_session.set_buffers(unet_output)

        # 3. Denoising loop
        for t in timesteps:
            latent_input = latents
            latent_input = self.scheduler.scale_model_input(latent_input, t)
            noise_pred = self.unet_qpc_session.run(
                {"encoder_hidden_states": ai100_output, "timestep": t, "sample": latent_input.numpy()}
            )
            latents = self.scheduler.step(noise_pred["out_sample"], t, latents).prev_sample

        # VAE decode step
        # TODO: Add QPC for VAE decode
        image = self.vae_decoder.model(latents / self.vae_decoder.model.config.scaling_factor, return_dict=False)[0]

        # Saftey check

        if torch.is_tensor(image):
            feature_extractor_input = self.image_processor.postprocess(image.detach(), output_type="pil")
        else:
            feature_extractor_input = self.image_processor.numpy_to_pil(image)

        safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt")

        self.safety_checker_session = QAICInferenceSession(str(self.compiled_safety_checker_path), [2])

        safety_checker_output = {
            "out_images": np.random.rand(1, 3, 512, 512).astype(np.float32),
            "has_nsfw_concepts": np.bool_(1),
        }
        self.safety_checker_session.set_buffers(safety_checker_output)

        checker_output = self.safety_checker_session.run(
            {"clip_input": safety_checker_input["pixel_values"].numpy(), "images": image.detach().numpy()}
        )

        has_nsfw_concept = checker_output["has_nsfw_concepts"].astype("bool")

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(image.detach(), output_type=output_type, do_denormalize=do_denormalize)

        # self.maybe_free_model_hooks()

        from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
