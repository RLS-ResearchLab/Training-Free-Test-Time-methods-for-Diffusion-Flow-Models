def apply_time_shift(scheduler, num_inference_steps: int, device, config: dict):
    """
    Calls scheduler.set_timesteps(...) with a shifted sigma schedule when
    config["time_shift"] is set, otherwise falls back to the scheduler's
    normal (unshifted) set_timesteps. Mutates `scheduler` in place, same as
    a normal set_timesteps() call — nothing is returned.

    config["time_shift"]:
      - unset / None            -> no shift, normal schedule
      - "resolution"            -> mu derived from config["height"]/["width"]
      - a float                 -> used directly as mu
    """
    shift_cfg = config.get("time_shift", None)
    if shift_cfg is None:
        if getattr(scheduler.config, "use_dynamic_shifting", False):
            mu = getattr(scheduler, "mu", None) or getattr(scheduler.config, "mu", 1.0)
            scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device)
        return

    if not hasattr(scheduler, "sigmas") or not hasattr(scheduler, "set_timesteps"):
        raise ValueError(
            "time_shift requested but this scheduler doesn't look like a "
            "flow-matching scheduler (no .sigmas). Only FlowMatchEuler-style "
            "schedulers (SD3, Flux) support time shifting here."
        )

    if shift_cfg == "resolution":
        mu = resolution_mu(config.get("height", 1024), config.get("width", 1024))
    else:
        mu = float(shift_cfg)

    # Most diffusers FlowMatch schedulers accept `mu` directly when the
    # scheduler was constructed with use_dynamic_shifting=True (Flux's
    # default). Try that path first since it's the "supported" one; fall
    # back to manually shifting the sigma schedule ourselves for schedulers
    # that don't expose it (e.g. SD3's default static-shift scheduler).
    try:
        scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
        return
    except TypeError:
        pass

    scheduler.set_timesteps(num_inference_steps, device=device)
    shifted = _shift_sigmas(scheduler.sigmas.tolist(), mu)
    scheduler.sigmas = scheduler.sigmas.new_tensor(shifted)
    scheduler.timesteps = scheduler.sigmas[:-1] * scheduler.config.num_train_timesteps \
        if hasattr(scheduler.config, "num_train_timesteps") else scheduler.timesteps