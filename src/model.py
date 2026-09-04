import torch
from torch import nn
from diffusers import StableDiffusion3Pipeline

try:
    from diffusers import FluxPipeline
    _HAS_FLUX = True
except ImportError:
    FluxPipeline = None
    _HAS_FLUX = False


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class BaseFlowModelWrapper(nn.Module):
    """
    Common interface every model wrapper implements, so src/sampler.py and
    the samplers in src/samplers/ don't need to know which model family
    they're driving. Concrete wrappers (SD35Wrapper, FluxWrapper) fill in
    self.transformer / self.vae / self.scheduler and encode_prompts(); this
    base class provides predict(), the single place a raw forward pass goes
    through, which is also what src/flops.py hooks into for FLOP counting.
    """
    family = "unknown"
    model_size_b = None  # approx param count in billions, filled in by load_model()

    def predict(self, hidden_states, timestep, encoder_hidden_states, pooled_projections):
        """One transformer forward pass -> predicted velocity/noise. Every
        sampler should call this instead of self.model.transformer(...)
        directly, so FLOP tracking and any future model-specific quirks
        (extra kwargs some families need) live in one place per wrapper."""
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            return_dict=False,
        )[0]

    def encode_prompts(self, prompt, negative_prompt: str = ""):
        raise NotImplementedError

    def init_latents(self, batch_size: int = 1, height: int = 1024, width: int = 1024, seed: int = 42):
        raise NotImplementedError

    def decode(self, latents: torch.Tensor):
        raise NotImplementedError


class SD35Wrapper(BaseFlowModelWrapper):
    family = "sd3"

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.bfloat16
    ):
        super().__init__()
        self.model_id = model_id
        self.dtype = torch_dtype

        print(f"Loading pre-trained pipeline: {model_id}...")

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype
        )
        self.pipe.enable_model_cpu_offload()

        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.model_size_b = sum(p.numel() for p in self.transformer.parameters()) / 1e9

    @torch.no_grad()
    def encode_prompts(self, prompt, negative_prompt: str = ""):
        # `prompt` may be a str or a list[str] (batched calls) — diffusers'
        # encode_prompt accepts either and returns batch-sized embeddings.
        neg = negative_prompt if isinstance(prompt, str) else [negative_prompt] * len(prompt)
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt=neg,
            negative_prompt_2=neg,
            negative_prompt_3=neg,
            device=_device(),
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
        device = _device()
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
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=self.dtype):
            image = self.vae.decode(latents_sc, return_dict=False)[0]
        return image.float()


class FluxWrapper(BaseFlowModelWrapper):
    """
    Flux (dev/schnell) wrapper, same interface as SD35Wrapper. Flux's
    transformer takes a couple of extra required kwargs diffusers' SD3
    pipeline doesn't (img_ids/txt_ids position ids, guidance for the
    distilled dev variant) so predict() is overridden rather than reusing
    the base implementation. NOT yet run end-to-end against real weights —
    written from the diffusers FluxPipeline reference implementation, needs
    a smoke test on GPU before trusting its output.
    """
    family = "flux"

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    ):
        super().__init__()
        if not _HAS_FLUX:
            raise ImportError(
                "FluxWrapper requires a diffusers version with FluxPipeline "
                "(diffusers>=0.30). `pip install -U diffusers`."
            )
        self.model_id = model_id
        self.dtype = torch_dtype
        self.is_schnell = "schnell" in model_id.lower()

        print(f"Loading pre-trained pipeline: {model_id}...")
        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=self.dtype)
        # Flux's transformer alone (~12B params, ~24GB in bf16) is close to or
        # over a 24GB GPU's full capacity, leaving no room for activations —
        # enable_model_cpu_offload() (block-at-a-time) still OOMs here, unlike
        # SD3.5 where it fits fine. enable_sequential_cpu_offload() moves
        # layer-by-layer instead of block-by-block: much slower, but the peak
        # memory footprint drops enough to actually fit.
        self.pipe.enable_sequential_cpu_offload()

        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.model_size_b = sum(p.numel() for p in self.transformer.parameters()) / 1e9

    @torch.no_grad()
    def encode_prompts(self, prompt, negative_prompt: str = ""):
        prompts = prompt if isinstance(prompt, list) else [prompt]
        encoded = self.pipe.encode_prompt(
            prompt=prompts, prompt_2=prompts, device=_device(), num_images_per_prompt=1,
        )
        # Different diffusers versions have returned either
        # (prompt_embeds, pooled_prompt_embeds, text_ids) or just
        # (prompt_embeds, pooled_prompt_embeds) for Flux's encode_prompt —
        # don't assume which; unpack defensively and build text_ids
        # ourselves either way. text_ids is just a zeros placeholder of
        # shape (seq_len, 3) (fixed positional ids, independent of prompt
        # content) per diffusers' own FluxPipeline.__call__, so recomputing
        # it locally is exactly as correct as whatever encode_prompt returns
        # — this was the source of the 'NoneType' object has no attribute
        # 'ndim' crash when the installed diffusers version returned/omitted
        # it differently than expected.
        if len(encoded) == 3:
            prompt_embeds, pooled_prompt_embeds, _ = encoded
        else:
            prompt_embeds, pooled_prompt_embeds = encoded
        self._text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=_device(), dtype=self.dtype)
        out = {
            "cond": {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds},
        }
        if self.is_schnell:
            # schnell is guidance-distilled -> no CFG, no uncond branch needed
            out["uncond"] = out["cond"]
        else:
            neg = [negative_prompt] * len(prompts)
            neg_encoded = self.pipe.encode_prompt(
                prompt=neg, prompt_2=neg, device=_device(), num_images_per_prompt=1,
            )
            neg_embeds, neg_pooled = neg_encoded[0], neg_encoded[1]
            out["uncond"] = {"prompt_embeds": neg_embeds, "pooled_prompt_embeds": neg_pooled}
        return out

    @torch.no_grad()
    def init_latents(self, batch_size: int = 1, height: int = 1024, width: int = 1024, seed: int = 42):
        device = _device()
        generator = torch.Generator(device=device).manual_seed(seed)
        num_channels_latents = self.transformer.config.in_channels // 4
        vae_scale_factor = 8
        latents = torch.randn(
            (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor),
            generator=generator, device=device, dtype=self.dtype,
        )
        # Flux's transformer wants packed (patchified) latents + matching img_ids;
        # self.pipe exposes the same packing helper the official pipeline uses.
        latents = self.pipe._pack_latents(
            latents, batch_size, num_channels_latents,
            height // vae_scale_factor, width // vae_scale_factor,
        )
        self._img_ids = self.pipe._prepare_latent_image_ids(
            batch_size, height // vae_scale_factor // 2, width // vae_scale_factor // 2, device, self.dtype,
        )
        self._packed_hw = (height // vae_scale_factor, width // vae_scale_factor)
        return latents

    def predict(self, hidden_states, timestep, encoder_hidden_states, pooled_projections):
        guidance = None
        if not self.is_schnell:
            guidance = torch.full((hidden_states.shape[0],), 3.5, device=hidden_states.device, dtype=self.dtype)
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep / 1000 if timestep.max() > 1.5 else timestep,
            guidance=guidance,
            pooled_projections=pooled_projections,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=self._text_ids,
            img_ids=self._img_ids,
            return_dict=False,
        )[0]

    @torch.no_grad()
    def decode(self, latents: torch.Tensor):
        h, w = self._packed_hw
        latents = self.pipe._unpack_latents(latents, h * 8, w * 8, self.pipe.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=self.dtype):
            image = self.vae.decode(latents, return_dict=False)[0]
        return image.float()


_FAMILY_MAP = {
    "flux": FluxWrapper,
    "sd3": SD35Wrapper,
}


def _detect_family(model_id: str) -> str:
    mid = model_id.lower()
    if "flux" in mid:
        return "flux"
    if "stable-diffusion-3" in mid or "sd3" in mid:
        return "sd3"
    raise ValueError(
        f"Can't infer model family from model_id={model_id!r}; pass family='sd3'|'flux' explicitly. "
        f"To add a new family, subclass BaseFlowModelWrapper and register it in _FAMILY_MAP."
    )


def load_model(model_id: str, family: str = None, torch_dtype=torch.bfloat16) -> BaseFlowModelWrapper:
    """
    Single entry point for every script (main.py, eval_large_scale.py,
    eval_fid_clip.py, run_multi_prompt.py) that needs a model wrapper.
    Auto-detects family from model_id if not given explicitly, so sweeping
    across models (different sizes within SD3.5, or across SD3.5/Flux) is
    just a matter of changing model_id — no per-script branching needed.
    """
    family = family or _detect_family(model_id)
    if family not in _FAMILY_MAP:
        raise ValueError(f"Unknown family={family!r}; known families: {list(_FAMILY_MAP)}")
    return _FAMILY_MAP[family](model_id=model_id, torch_dtype=torch_dtype)