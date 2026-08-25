import torch
from torch import nn
from diffusers import StableDiffusion3Pipeline

class SD35Wrapper(nn.Module):
    

    def __init__(
        self, 
        model_id: str = "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.bfloat16
    ):
        super().__init__()
        self.dtype = torch_dtype

        print(f"Loading pre-trained pipeline: {model_id}...")
        
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe.to(device)

        
        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler

    @torch.no_grad()
    def encode_prompts(self, prompt: str, negative_prompt: str = ""):

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt,
            device="cuda" if torch.cuda.is_available() else "cpu",
            do_classifier_free_guidance=True
        )

        return {
            "cond": {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds
            },
            "uncond": {
                "prompt_embeds": negative_prompt_embeds,
                "pooled_prompt_embeds": negative_pooled_prompt_embeds
            }
        }
    @torch.no_grad()
    def init_latents(self, batch_size: int = 1, height: int = 1024, width: int = 1024, seed: int = 42):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

        num_channels_latents = self.transformer.config.in_channels
        vae_scale_factor = 8  # SD3 VAE downsamples by 8

        latents = torch.randn(
            (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor),
            generator=generator,
            device=device,
            dtype=self.dtype,
        )
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor):
        
        latents_sc = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            image = self.vae.decode(latents_sc, return_dict=False)[0]
        return image.float()