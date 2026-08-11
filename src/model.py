import torch
from torch import nn
from diffusers import StableDiffusion3Pipeline
from .utils import freeze

class SD35Wrapper(nn.Module):
    """Wrapper SD 3.5 aligné sur le style et l'interface de FluxWrapper."""

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
        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=self.dtype)
        
        # 1. Extraction explicite des composants
        self.vae = pipe.vae.cuda()
        self.transformer = pipe.transformer.cuda()
        
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.text_encoder_3 = pipe.text_encoder_3

        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        
        self.scheduler = pipe.scheduler
        self._encode_prompt_fn = pipe.encode_prompt

        # 2. Gel des modules (Freeze)
        for module in [self.vae, self.transformer, self.text_encoder, self.text_encoder_2]:
            module.eval()
            freeze(module)

        if self.text_encoder_3 is not None:
            self.text_encoder_3.eval()
            freeze(self.text_encoder_3)

        del pipe
        torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_prompts(self, prompts: list[str]):
        """Encode les prompts en passant temporairement les encodeurs sur GPU puis CPU."""
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

        # Offload vers le CPU pour libérer la VRAM pour le DiT
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
        """Initialise le bruit gaussien."""
        h = w = self.image_size // 8
        torch.manual_seed(seed)
        return torch.randn(batch_size, 16, h, w, device="cuda", dtype=self.dtype)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor):
        """Décode les latents en image RGB."""
        latents_sc = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            images = self.vae.decode(latents_sc, return_dict=False)[0]
        return images.float()