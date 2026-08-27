import os
import random
import torch
from PIL import Image

from src.utils import ComputeTracker


def find_max_batch_size(sampler, prompt_template, config, ceiling=256, start=1,
                         methods_override=None):
    """
    Auto-detects the largest batch_size that fits in currently-free GPU
    memory by doubling until OOM, then backing off one step for safety
    margin (fragmentation, other processes on a shared GPU can still eat
    into what was measured as "free" a moment ago).

    methods_override: IMPORTANT — probe with the most expensive method
    combination you intend to actually run at this batch_size. A batch_size
    that's safe for bare "cfg" is NOT necessarily safe for "auto_guidance"
    (extra forward pass per step) or specifically the weight_noise variant
    (temporarily clones weight tensors on top of that). Defaults to
    ["cfg", "auto_guidance"] with auto_guidance_variant="weight_noise"
    forced into a copy of `config` for the probe, since that's the priciest
    combination this repo runs — probing with anything cheaper risks
    exactly the false-safe result that caused an OOM mid-run before.

    ceiling=256: if you ever run on hardware where 256 genuinely fits
    (e.g. a dedicated 80GB+ GPU, low resolution, few inference steps),
    this will find it. On a shared ~24GB GPU at 512-1024px it will not —
    that's expected, the function reports the real number, it doesn't
    force one.
    """
    if not torch.cuda.is_available():
        print("[find_max_batch_size] No CUDA device — defaulting to batch_size=1")
        return 1

    if methods_override is None:
        methods_override = ["cfg", "auto_guidance"]
    probe_config = dict(config)
    if "auto_guidance" in methods_override:
        probe_config["auto_guidance_variant"] = "weight_noise"  # priciest variant

    device = torch.cuda.current_device()
    last_good = None
    bs = start
    while bs <= ceiling:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            test_prompts = [prompt_template] * bs
            sampler.sample(
                prompt=test_prompts, config=probe_config, exp_name="_batch_probe",
                save_name="probe.png", save_to_disk=False,
                methods_override=methods_override,
            )
            peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
            free_gb, total_gb = [x / 1e9 for x in torch.cuda.mem_get_info(device)]
            print(f"[find_max_batch_size] batch_size={bs} OK "
                  f"(peak={peak_gb:.2f} GiB, free={free_gb:.2f}/{total_gb:.2f} GiB)")
            last_good = bs
            bs *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[find_max_batch_size] batch_size={bs} OOM — stopping search")
            break

    if last_good is None:
        print("[find_max_batch_size] Even batch_size=1 OOM'd — GPU is too full "
              "right now (check nvidia-smi for competing processes) or "
              "resolution/steps are too high for available VRAM.")
        return 1

    # back off one notch from the largest that worked, for headroom against
    # fragmentation and any other process's memory growing after this probe
    safe_bs = max(1, last_good // 2) if last_good == bs // 2 else last_good
    # (last_good is already the biggest that worked since we broke on OOM
    # at the next size up; still take one more step back for margin)
    safe_bs = max(1, last_good // 2) if last_good > start else last_good
    print(f"[find_max_batch_size] Largest that fit: {last_good} — using {safe_bs} for headroom")
    return safe_bs


class ModularSampler:
    def __init__(self, model_wrapper, output_dir="samples"):
        self.model = model_wrapper
        self.output_dir = output_dir

    # ------------------------------------------------------------------ #
    # Auto-Guidance "weak model" strategies
    #
    # Auto-Guidance (Karras et al.) works by contrasting the real model's
    # prediction against a deliberately *weaker* prediction, then pushing
    # away from that weak prediction — similar in spirit to CFG, but the
    # "bad" signal comes from a degraded version of the same model instead
    # of an unconditional pass. Three ways to make the prediction weaker
    # are implemented below, selected via config["auto_guidance_variant"]:
    #
    #   - "latent_scale"    : shrink the input latents before the forward
    #                         pass (original placeholder in this repo).
    #   - "weight_noise"    : temporarily inject Gaussian noise into one
    #                         (or a few) of the transformer's weight
    #                         tensors, run the forward pass, then restore
    #                         the original weights.
    #   - "fewer_timesteps" : feed the weak branch a coarser timestep
    #                         schedule (round the current step down to the
    #                         nearest multiple of a stride), simulating a
    #                         model that only ever saw a subset of noise
    #                         levels.
    # ------------------------------------------------------------------ #

    def _get_weaker_prediction(self, latents, t, cond_embeds, pooled, config, step_idx, total_steps):
        variant = config.get("auto_guidance_variant", "latent_scale")
        if variant == "weight_noise":
            return self._weak_prediction_weight_noise(latents, t, cond_embeds, pooled, config)
        elif variant == "fewer_timesteps":
            return self._weak_prediction_fewer_timesteps(
                latents, cond_embeds, pooled, config, step_idx, total_steps
            )
        elif variant == "latent_scale":
            return self._weak_prediction_latent_scale(latents, t, cond_embeds, pooled, config)
        else:
            raise ValueError(f"Unknown auto_guidance_variant: {variant!r}")

    def _weak_prediction_latent_scale(self, latents, t, cond_embeds, pooled, config):
        """Original placeholder: shrink the latents before the forward pass."""
        scale = config.get("auto_guidance_latent_scale", 0.95)
        perturbed_latents = latents * scale
        return self.model.transformer(
            hidden_states=perturbed_latents,
            timestep=t.unsqueeze(0) if t.dim() == 0 else t,
            encoder_hidden_states=cond_embeds,
            pooled_projections=pooled,
            return_dict=False,
        )[0]

    def _weak_prediction_weight_noise(self, latents, t, cond_embeds, pooled, config):
        """
        Method 1: perturb a fixed subset of the transformer's weight matrices
        with additive Gaussian noise, run the forward pass through the
        perturbed network, then restore the original weights.

        Only tensors with ndim >= 2 are eligible (real weight matrices —
        this skips biases and norm scale/shift params, which are far more
        sensitive to perturbation and tend to just break the forward pass
        rather than "weaken" it). The subset is chosen once via a seeded
        RNG so the *same* weight(s) are perturbed at every timestep of a
        given run, matching how Auto-Guidance perturbs a fixed part of the
        network rather than re-randomizing every step.
        """
        std_scale = config.get("auto_guidance_weight_noise_std", 0.05)
        num_weights = config.get("auto_guidance_num_weights_perturbed", 1)
        seed = config.get("seed", 42)

        candidate_params = [
            p for p in self.model.transformer.parameters() if p.ndim >= 2
        ]
        rng = random.Random(seed)
        chosen = rng.sample(candidate_params, k=min(num_weights, len(candidate_params)))

        originals = []
        for p in chosen:
            originals.append((p, p.data.clone()))
            noise = torch.randn_like(p.data) * p.data.std() * std_scale
            p.data.add_(noise)

        try:
            noise_weak = self.model.transformer(
                hidden_states=latents,
                timestep=t.unsqueeze(0) if t.dim() == 0 else t,
                encoder_hidden_states=cond_embeds,
                pooled_projections=pooled,
                return_dict=False,
            )[0]
        finally:
            # Always restore, even if the forward pass raises.
            for p, orig in originals:
                p.data.copy_(orig)

        return noise_weak

    def _weak_prediction_fewer_timesteps(self, latents, cond_embeds, pooled, config, step_idx, total_steps):
        """
        Method 2: give the weak branch a coarser view of the schedule by
        rounding the current step down to the nearest multiple of
        `auto_guidance_timestep_stride`. E.g. with stride=4 and 30 steps,
        the weak branch only ever sees timesteps at indices 0, 4, 8, ...
        instead of every step — effectively predicting with fewer distinct
        timesteps than the real (strong) branch.
        """
        stride = max(1, config.get("auto_guidance_timestep_stride", 4))
        coarse_idx = min((step_idx // stride) * stride, total_steps - 1)
        t_weak = self.model.scheduler.timesteps[coarse_idx]
        return self.model.transformer(
            hidden_states=latents,
            timestep=t_weak.unsqueeze(0) if t_weak.dim() == 0 else t_weak,
            encoder_hidden_states=cond_embeds,
            pooled_projections=pooled,
            return_dict=False,
        )[0]

    @torch.no_grad()
    def sample(
        self,
        prompt,
        config: dict,
        exp_name: str = "exp_run",
        methods_override: list = None,
        save_name: str = "result.png",
        save_to_disk: bool = True,
    ):
        """
        prompt: a single string (original behavior, batch_size=1) OR a list of
            strings (batched: all prompts denoised together in one forward pass
            per step). When a list is passed, the returned `image` tensor has
            shape [len(prompt), C, H, W] and save_to_disk is ignored (batched
            calls are for large-scale sweeps where individual PNGs aren't written).
        methods_override: if provided, replaces config["methods"] for this call only
            (the config dict itself is never mutated). This is what lets main.py run
            the exact same config twice — once with a method included, once with it
            stripped out — to produce a clean "with X" / "without X" pair.
        save_name: filename written inside samples/<exp_name>/, so the "with" and
            "without" runs of the same experiment don't overwrite one another.
        save_to_disk: set False for large-N sweeps (e.g. scripts/eval_large_scale.py
            generating 10k-50k samples per technique) where writing every generated
            image to samples/ would flood disk for no benefit — the caller only needs
            the returned tensor. Defaults to True so every other caller (main.py,
            run_multi_prompt.py, eval_fid_clip.py) keeps its existing behavior.
        """
        save_folder = os.path.join(self.output_dir, exp_name)
        active_methods = methods_override if methods_override is not None else config.get("methods", ["cfg"])

        cfg_scale = config.get("cfg_scale", 4.5)
        ag_scale = config.get("auto_guidance_scale", 1.0)

        prompts = prompt if isinstance(prompt, list) else [prompt]
        batch_size = len(prompts)

        use_amp = config.get("use_amp", False)
        amp_dtype = getattr(self.model, "dtype", torch.bfloat16)
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        with ComputeTracker() as tracker:
            # 1. Encode prompts (conditioned & unconditioned) — encode_prompts
            #    already accepts a list and returns batch-sized embeddings, so
            #    no manual .repeat() is needed for the batched case.
            embeds = self.model.encode_prompts(prompts)
            cond_embeds, cond_pooled = embeds["cond"]["prompt_embeds"], embeds["cond"]["pooled_prompt_embeds"]
            uncond_embeds, uncond_pooled = embeds["uncond"]["prompt_embeds"], embeds["uncond"]["pooled_prompt_embeds"]

            # 2. Init Latents & Scheduler
            latents = self.model.init_latents(
                batch_size=batch_size,
                height=config.get("height", 1024),
                width=config.get("width", 1024),
                seed=config.get("seed", 42),
            )
            self.model.scheduler.set_timesteps(
                config.get("num_inference_steps", 28),
                device=device_type,
            )
            total_steps = len(self.model.scheduler.timesteps)

            # 3. Modular Denoising Loop
            forward_pass_count = 0
            for step_idx, t in enumerate(self.model.scheduler.timesteps):
                t_batch = t.repeat(batch_size) if t.dim() == 0 else t

                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    # Base Forward Pass (Conditional)
                    noise_cond = self.model.transformer(
                        hidden_states=latents,
                        timestep=t_batch,
                        encoder_hidden_states=cond_embeds,
                        pooled_projections=cond_pooled,
                        return_dict=False,
                    )[0]
                forward_pass_count += batch_size

                guided_noise = noise_cond.clone()

                # --- Method A: Classifier-Free Guidance (CFG) ---
                if "cfg" in active_methods and cfg_scale > 1.0:
                    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                        noise_uncond = self.model.transformer(
                            hidden_states=latents,
                            timestep=t_batch,
                            encoder_hidden_states=uncond_embeds,
                            pooled_projections=uncond_pooled,
                            return_dict=False,
                        )[0]
                    forward_pass_count += batch_size
                    guided_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

                # --- Method B: Auto-Guidance (variant-dispatched weak prediction) ---
                if "auto_guidance" in active_methods and ag_scale > 0.0:
                    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                        noise_weaker = self._get_weaker_prediction(
                            latents, t_batch, cond_embeds, cond_pooled, config, step_idx, total_steps
                        )
                    forward_pass_count += batch_size
                    guided_noise = guided_noise + ag_scale * (noise_cond - noise_weaker)

                # Update latent step (t -> t-1)
                latents = self.model.scheduler.step(guided_noise, t, latents, return_dict=False)[0]

            # 4. Decode
            image = self.model.decode(latents)

        if save_to_disk and batch_size == 1:
            self._save_tensor_as_image(image, os.path.join(save_folder, save_name))
        elif save_to_disk and batch_size > 1:
            print(f"[ModularSampler] save_to_disk=True ignored for batched call (batch_size={batch_size}); "
                  f"save individual images from the returned tensor yourself if needed.")
        metrics = {
            "total_forward_passes": forward_pass_count,
            "active_methods": list(active_methods),
            "auto_guidance_variant": config.get("auto_guidance_variant", "latent_scale"),
            "time_sec": tracker.elapsed_sec,
            "peak_memory_mb": tracker.peak_memory_mb,
        }
        return image, metrics

    def _save_tensor_as_image(self, tensor: torch.Tensor, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tensor = (tensor.squeeze(0).clamp(-1.0, 1.0) + 1.0) / 2.0
        tensor = (tensor.cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        Image.fromarray(tensor).save(save_path)
        print(f"Saved result: {save_path}")