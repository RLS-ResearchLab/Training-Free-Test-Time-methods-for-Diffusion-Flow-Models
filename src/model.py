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