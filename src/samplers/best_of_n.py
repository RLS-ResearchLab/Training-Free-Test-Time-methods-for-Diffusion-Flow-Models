"""
Task 3 — Best-of-N sampling.

Generates N images from N different noise seeds for the same prompt,
scores each candidate against the prompt with CLIP, and returns the
best-scoring one — with full visibility into every candidate's score,
not just the winner.

This wraps whatever base method(s) you already use (passed as
`base_methods`, e.g. ["cfg"] or ["cfg", "auto_guidance"]) — best-of-N
is a technique that composes on top of the existing methods, not a
replacement for them.

Compute cost note: best-of-N's real cost is N full denoising runs, not
N cheap reward evals. `total_forward_passes` / `time_sec` in the
returned metrics are SUMS across all N candidates, so cost comparisons
against single-shot methods stay honest.
"""
import copy
import time

from src.sampler import ModularSampler
from src.rewards.clip_reward import CLIPPromptReward
from src.utils import clip_align_score


class BestOfNSampler:
    def __init__(self, model_wrapper, base_methods=("cfg",), n=5, reward_fn=None):
        self.sampler = ModularSampler(model_wrapper)
        self.base_methods = list(base_methods)
        self.n = n
        self.reward_fn = reward_fn or CLIPPromptReward()

    def sample(self, prompt, config, exp_name, save_name="best_of_n.png"):
        candidates = []
        t_start = time.time()
        total_forward_passes = 0
        base_seed = config.get("seed", 0)

        for i in range(self.n):
            cand_config = copy.deepcopy(config)
            cand_config["seed"] = base_seed + i * 1000 + 7  # distinct noise per candidate

            img, metrics = self.sampler.sample(
                prompt=prompt,
                config=cand_config,
                exp_name=exp_name,
                methods_override=self.base_methods,
                save_name=f"candidate_{i}_{save_name}",
            )
            score = clip_align_score(self.reward_fn, img, prompt)
            total_forward_passes += metrics.get("total_forward_passes", 0)

            candidates.append({
                "candidate_idx": i,
                "seed": cand_config["seed"],
                "clip_score": round(score, 4),
                "image": img,
                "metrics": metrics,
            })

        best = max(candidates, key=lambda c: c["clip_score"])

        aggregate_metrics = {
            "total_forward_passes": total_forward_passes,
            "time_sec": time.time() - t_start,
            "peak_memory_mb": max(c["metrics"].get("peak_memory_mb", 0) for c in candidates),
            "n": self.n,
            "best_candidate_idx": best["candidate_idx"],
            "best_seed": best["seed"],
            "all_candidate_scores": [
                {"candidate_idx": c["candidate_idx"], "seed": c["seed"], "clip_score": c["clip_score"]}
                for c in candidates
            ],
        }

        print(f"[best_of_n] prompt={prompt!r}")
        for c in candidates:
            marker = "  <- BEST" if c["candidate_idx"] == best["candidate_idx"] else ""
            print(f"    candidate {c['candidate_idx']} (seed={c['seed']}): clip_score={c['clip_score']:.4f}{marker}")

        return best["image"], aggregate_metrics