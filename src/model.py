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
from torch import nn
from diffusers import StableDiffusion3Pipeline

# Helper function as seen in Dhia's repository
def freeze(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False

class SD35Wrapper(nn.Module):
    """SD 3.5 Model Wrapper following Dhia's architectural pattern."""

    def __init__(
        self, 
        model_id: str = "stabilityai/stable-diffusion-3.5-medium",
        decode_steps: int = 28,
        guidance_scale: float = 4.5,
        image_size: int = 512,
        torch_dtype=torch.bfloat16
    ):
        super().__init__()
        self.decode_steps = decode_steps
        self.guidance_scale = guidance_scale
        self.image_size = image_size
        self.dtype = torch_dtype

        self._load_model(model_id)

    def _load_model(self, model_id: str):
        print(f"Loading SD 3.5 pipeline from {model_id}...")
        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=self.dtype)

        self.vae = pipe.vae.cuda()
        self.transformer = pipe.transformer.cuda()
        
        # Text encoders & tokenizers
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.text_encoder_3 = pipe.text_encoder_3
        
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        
        self.scheduler = pipe.scheduler
        self._encode_prompt_fn = pipe.encode_prompt

        # Freeze all modules to save memory
        for module in [self.vae, self.transformer, self.text_encoder, self.text_encoder_2, self.text_encoder_3]:
            module.eval()
            freeze(module)

        del pipe
        torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_prompts(self, prompts: list[str]):
        """Encodes prompts into dual prompt embeddings and manages GPU memory dynamically."""
        # Ensure encoders are on GPU for execution
        self.text_encoder.cuda()
        self.text_encoder_2.cuda()
        if self.text_encoder_3:
            self.text_encoder_3.cuda()

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._encode_prompt_fn(
            prompt=prompts,
            device="cuda",
            do_classifier_free_guidance=True,
        )

        # Move encoders back to CPU to free VRAM for Transformer & VAE execution
        self.text_encoder.cpu()
        self.text_encoder_2.cpu()
        if self.text_encoder_3:
            self.text_encoder_3.cpu()

        torch.cuda.empty_cache()

        return {
            "cond": {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds},
            "uncond": {"prompt_embeds": negative_prompt_embeds, "pooled_prompt_embeds": negative_pooled_prompt_embeds},
        }

    def init_latents(self, batch_size: int = 1, seed: int = 42):
        """Initializes Gaussian noise latents."""
        h = w = self.image_size // 8
        torch.manual_seed(seed)
        z = torch.randn(batch_size, 16, h, w, device="cuda", dtype=self.dtype)
        return z

    @torch.no_grad()
    def decode(self, latents: torch.Tensor):
        """Decodes latents to RGB image tensor."""
        latents_scaled = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            images = self.vae.decode(latents_scaled, return_dict=False)[0]

        images = (images.float().clamp(-1.0, 1.0) + 1.0) / 2.0
        return images