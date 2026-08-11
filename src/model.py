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

        
        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler

    @torch.no_grad()
    def encode_prompts(self, prompt: str, negative_prompt: str = ""):
        
        return self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device="cuda" if torch.cuda.is_available() else "cpu",
            do_classifier_free_guidance=True
        )

    @torch.no_grad()
    def decode(self, latents: torch.Tensor):
        
        latents_sc = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            image = self.vae.decode(latents_sc, return_dict=False)[0]
        return image.float()