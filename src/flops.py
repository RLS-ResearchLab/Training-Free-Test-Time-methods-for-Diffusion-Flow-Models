"""
FLOP accounting for the denoising loop.

"Average forward passes" / NFEs treats every forward pass as equally
expensive, which is wrong as soon as you compare across model sizes or
families (a Flux-dev pass and an SD3.5-medium pass are not the same cost),
or across techniques that change the effective sequence length / resolution
mid-run. This module measures real FLOPs per forward pass with
torch.utils.flop_counter.FlopCounterMode (torch>=2.1) and caches the result
per (module id, input-shape signature), since the transformer's cost only
depends on shapes, not on the actual tensor values — so we pay the counting
overhead once per distinct shape instead of once per denoising step.

Usage:
    tracker = FlopTracker(model_wrapper)
    ...
    flops = tracker.count(hidden_states=latents, timestep=t_batch, ...)
    total_flops += flops
"""
import torch

try:
    from torch.utils.flop_counter import FlopCounterMode
    _HAS_FLOP_COUNTER = True
except ImportError:  # torch < 2.1
    FlopCounterMode = None
    _HAS_FLOP_COUNTER = False


def _shape_signature(kwargs) -> tuple:
    """Hashable signature of a call's tensor shapes/dtypes — FLOPs for a
    given module are a pure function of these, not of the tensor values."""
    sig = []
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if torch.is_tensor(v):
            sig.append((k, tuple(v.shape), str(v.dtype)))
        else:
            sig.append((k, repr(v)))
    return tuple(sig)


class FlopTracker:
    """
    Wraps a model wrapper (src/model.py's BaseFlowModelWrapper subclasses)
    and counts FLOPs for one predict() call, caching by input-shape
    signature so a whole denoising run only actually invokes
    FlopCounterMode once per distinct shape (usually once, ever, since
    every step uses the same latent/embed shapes) instead of once per step.

    Deliberately goes through model_wrapper.predict(), NOT the raw
    transformer module directly: predict() is where each family's
    wrapper adds whatever extra kwargs its transformer needs (Flux's
    img_ids/txt_ids/guidance, the timestep/1000 rescale, etc). Calling
    the transformer directly here bypassed all of that and sent it a
    None txt_ids for Flux — it happened to work for SD3.5 only because
    SD3.5's predict() doesn't add anything beyond the 4 generic kwargs.
    """

    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        self._cache = {}
        self._warned_no_counter = False

    def count(self, **kwargs) -> float:
        """Returns FLOPs for one predict() call with these kwargs. Runs the
        real forward pass either way (the caller needs the output); this
        only adds the counting overhead, and only on cache misses."""
        if not _HAS_FLOP_COUNTER:
            if not self._warned_no_counter:
                print("[flops] torch.utils.flop_counter unavailable (torch<2.1) — "
                      "falling back to a 0 FLOP count; compute-vs-quality plots "
                      "will be uninformative. Upgrade torch to fix this.")
                self._warned_no_counter = True
            return 0.0

        sig = _shape_signature(kwargs)
        if sig in self._cache:
            return self._cache[sig]

        with FlopCounterMode(display=False) as fcm:
            self.model_wrapper.predict(**kwargs)
        flops = float(fcm.get_total_flops())
        self._cache[sig] = flops
        return flops


def estimate_dit_flops_per_forward(num_params: int, seq_len: int) -> float:
    """
    Cheap fallback estimate (2 * N_params * seq_len, the standard transformer
    FLOPs-per-token approximation) for contexts where running a real
    FlopCounterMode pass isn't convenient (e.g. quick back-of-envelope
    comparisons before a full run). Prefer FlopTracker.count() for anything
    that ends up in a reported figure.
    """
    return 2.0 * num_params * seq_len