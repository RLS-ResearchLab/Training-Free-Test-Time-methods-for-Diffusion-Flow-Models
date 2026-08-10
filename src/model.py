import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL

class PretrainedT2IModel:
    """Wraps a pre-trained Text-to-Image model for granular step-level manipulation."""
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        
        # Component loading
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(self.device)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    @torch.no_grad()
    def encode_prompt(self, prompt: str, negative_prompt: str = ""):
        """Helper to get conditioned and unconditioned text embeddings."""
        cond_tokens = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids.to(self.device)
        uncond_tokens = self.tokenizer(
            negative_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids.to(self.device)

        cond_embeds = self.text_encoder(cond_tokens)[0]
        uncond_embeds = self.text_encoder(uncond_tokens)[0]
        return torch.cat([uncond_embeds, cond_embeds])

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor):
        """Decodes latent representations back into PIL Images."""
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image
import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, AutoencoderKL
from transformers import CLIPTextModelWithProjection, T5EncoderModel

class SD35ModelWrapper:
    """Modular wrapper for Stable Diffusion 3.5 providing a unified model interface."""

    def __init__(self, model_id: str = "stabilityai/stable-diffusion-3.5-medium", device: str = "cuda", torch_dtype=torch.bfloat16):
        self.device = device
        self.dtype = torch_dtype

        print(f"Loading SD3.5 components from {model_id}...")
        
        # Load pipeline to extract components cleanly
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, 
            torch_dtype=self.dtype
        ).to(self.device)

        # 1. Store individual components
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.text_encoder_3 = pipe.text_encoder_3
        
        self.transformer: SD3Transformer2DModel = pipe.transformer
        self.vae: AutoencoderKL = pipe.vae
        self.scheduler = pipe.scheduler

        # Reference to pipeline encode_prompt helper for SD3 multi-encoder handling
        self._encode_prompt_fn = pipe.encode_prompt

    @torch.no_grad()
    def encode_prompt(self, prompt: str, negative_prompt: str = ""):
        """Encodes prompts into conditioned and unconditioned sequence & pooled embeddings."""
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._encode_prompt_fn(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=self.device,
            do_classifier_free_guidance=True,
        )

        return {
            "cond": {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds},
            "uncond": {"prompt_embeds": negative_prompt_embeds, "pooled_prompt_embeds": negative_pooled_prompt_embeds},
        }

    @torch.no_grad()
    def predict_noise(self, latent: torch.Tensor, timestep: torch.Tensor, prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor):
        """Runs a single forward pass through the DiT backbone."""
        return self.transformer(
            hidden_states=latent,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor):
        """Decodes latents to RGB PIL images."""
        # Scale latents according to VAE configuration
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        
        # Denormalize [-1, 1] to [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image