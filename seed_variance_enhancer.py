import nodes
import torch
import node_helpers

#Released under the terms of the MIT No Attribution License
#Version 1.0

class SeedVarianceEnhancer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "randomize_percent": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1, "tooltip": "The percentage of embedding values to which random noise is added."}),  # Probability of modifying a value
                "strength": ("FLOAT", {"default": 20, "min": 0.0, "max": 1000000, "step": 0.001, "tooltip": "The scale of the random noise to add to the selected embedding values."}),
                "noise_insert": (["noise on beginning steps", "noise on ending steps", "noise on all steps"],),
                "steps_switchover_percent": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1, "tooltip": "What percentage of steps to process before switching from noisy to original embedding, or from original to noisy embedding."}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFF, "step": 1})
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "randomize_conditioning"
    CATEGORY = "conditioning"

    def randomize_conditioning(self, conditioning, strength, seed, randomize_percent, noise_insert, steps_switchover_percent):
        if randomize_percent == 0 or strength == 0:
            return (conditioning,)

        randomize_percent = randomize_percent / 100;
        steps_switchover_percent = steps_switchover_percent / 100;

        torch.manual_seed(seed)

        noisy_embedding = []
        for t in conditioning:
            if isinstance(t[0], torch.Tensor):
                noise = torch.rand_like(t[0]) * 2 * strength - strength
                mask = torch.bernoulli(torch.ones_like(t[0]) * randomize_percent).bool() # Randomly select a percentage of values.
                modified_noise = noise * mask  # Only apply noise to the selected values.
                noisy_embedding.append([t[0] + modified_noise, t[1]])
            else:
                logging.warning("SeedVarianceEnhancer doesn't know how to work with this conditioning. Passing it through untouched.")
                return (conditioning,)
            break # we will only use the first conditioning

        if noise_insert == "noise on beginning steps":
            new_conditioning = node_helpers.conditioning_set_values(noisy_embedding, {"start_percent": 0.0, "end_percent": steps_switchover_percent})
            new_conditioning = new_conditioning + node_helpers.conditioning_set_values(conditioning, {"start_percent": steps_switchover_percent, "end_percent": 1.0})
        elif noise_insert == "noise on ending steps":
            new_conditioning = node_helpers.conditioning_set_values(conditioning, {"start_percent": 0.0, "end_percent": steps_switchover_percent})
            new_conditioning = new_conditioning + node_helpers.conditioning_set_values(noisy_embedding, {"start_percent": steps_switchover_percent, "end_percent": 1.0})
        else:
            return (noisy_embedding,)

        return (new_conditioning,)


NODE_CLASS_MAPPINGS = {
    "SeedVarianceEnhancer": SeedVarianceEnhancer
}
