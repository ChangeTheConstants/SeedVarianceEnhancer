import nodes
import torch
import node_helpers
import logging

#Released under the terms of the MIT No Attribution License
#Version 2.0

class SeedVarianceEnhancer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "randomize_percent": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1, "tooltip": "The percentage of embedding values to which random noise is added."}),
                "strength": ("FLOAT", {"default": 20, "min": -0xFFFFFFFF, "max": 0xFFFFFFFF, "step": 0.00001, "tooltip": "The scale of the random noise."}),
                "noise_insert": (["noise on beginning steps", "noise on ending steps", "noise on all steps", "disabled"], {"tooltip": "Specifies on which steps in the generation process the noisy text embedding is used."}),
                "steps_switchover_percent": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1, "tooltip": "At which point in the generation process the switch between noisy and original embeddings occurs. Use this formula: (100/TOTALSTEPS) * STEPS - 1"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True, "tooltip": "The random seed used for embedding value selection and noise generation."}),
                "mask_starts_at": (["beginning", "end"], {"tooltip": "Which part of the prompt will be protected from noise."}),
                "mask_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1, "tooltip": "The percentage of the prompt that will be protected from noise."}),
                "log_to_console": ("BOOLEAN", {"default": False, "tooltip": "Print out useful information to the ComfyUI console, including suggested strength values. You are using version 2.0."})
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "randomize_conditioning"
    CATEGORY = "advanced/conditioning"

    def log_tensor_statistics(self, tensor):

        # Calculate statistics
        mean = torch.mean(tensor).item()
        std = torch.std(tensor).item()
        min_val = torch.min(tensor).item()
        max_val = torch.max(tensor).item()

        # Log statistics to console
        logging.info(f"Embedding Tensor Statistics:  (from SeedVarianceEnhancer)")
        logging.info(f"  Dimensions: {', '.join(map(str, tensor.shape))}   Min: {min_val:.6f}   Max: {max_val:.6f}   Mean: {mean:.6f}   Standard Deviation: {std:.6f}   Maybe try strength in range {std/2:.6f} - {std*2:.6f}")

    def randomize_conditioning(self, conditioning, randomize_percent, strength, noise_insert, steps_switchover_percent, seed, mask_starts_at, mask_percent, log_to_console):

        # Validate and scale input
        steps_switchover_percent = max(0, min(100, steps_switchover_percent)) / 100
        randomize_percent = max(0, min(100, randomize_percent)) / 100
        mask_percent = max(0, min(100, mask_percent)) / 100

        noisy_embedding = []
        t = conditioning[0]

        # Check for early return conditions
        if randomize_percent <= 0 or strength == 0 or mask_percent == 1:
            if log_to_console:
                warning_msg = ""
                if randomize_percent <= 0:
                    warning_msg = "randomize_percent is set to zero"
                elif strength == 0:
                    warning_msg = "strength is set to zero"
                elif mask_percent == 1.0:
                    warning_msg = "mask_percent is set to 100"
                logging.warning(f"SeedVarianceEnhancer is disabled. {warning_msg}. Passing conditioning through unchanged.")
                if isinstance(t[0], torch.Tensor):
                    self.log_tensor_statistics(t[0])
            return (conditioning,)
        if noise_insert == "disabled":
            if log_to_console:
                logging.warning("SeedVarianceEnhancer is disabled. Passing conditioning through unchanged.")
                if isinstance(t[0], torch.Tensor):
                    self.log_tensor_statistics(t[0])
            return (conditioning,)
        if len(conditioning) < 1:
            if log_to_console:
                logging.warning("SeedVarianceEnhancer received a zero length conditioning. Passing it through unchanged.")
            return (conditioning,)

        if len(conditioning) > 1 and log_to_console:
            logging.warning("SeedVarianceEnhancer will only use the first embedding from this conditioning.")


        if isinstance(t[0], torch.Tensor):
            torch.manual_seed(seed)

            if log_to_console:
                self.log_tensor_statistics(t[0]) # print statistical analysis of tensor to console

            noise = torch.rand_like(t[0]) * 2 * strength - strength
            noise_mask = torch.bernoulli(torch.ones_like(t[0]) * randomize_percent).bool() # Randomly select a percentage of values.

            if mask_percent > 0:
                seq_len = t[0].size(1)

                if mask_starts_at == "end":
                    mask_start = seq_len - int(seq_len * mask_percent)
                    mask_end = seq_len
                else:
                    mask_start = 0
                    mask_end = int(seq_len * mask_percent)

                # Create the mask
                middle_mask = torch.arange(seq_len, device=t[0].device).view(1, -1, 1).expand(t[0].size(0), -1, t[0].size(2))
                middle_mask = (middle_mask >= mask_start) & (middle_mask < mask_end)

                # Combine with existing mask
                noise_mask = noise_mask & (~middle_mask)  # Zeros noise_mask within the mask range

            modified_noise = noise * noise_mask  # Only apply noise to the selected values.
            noisy_embedding.append([t[0] + modified_noise, t[1]])
        else:
            if log_to_console:
                logging.warning("SeedVarianceEnhancer received a conditioning with no Tensor. Passing it through untouched.")
            return (conditioning,)

        if noise_insert == "noise on beginning steps":
            new_conditioning = node_helpers.conditioning_set_values(noisy_embedding, {"start_percent": 0.0, "end_percent": steps_switchover_percent})
            new_conditioning += node_helpers.conditioning_set_values(conditioning, {"start_percent": steps_switchover_percent, "end_percent": 1.0})
        elif noise_insert == "noise on ending steps":
            new_conditioning = node_helpers.conditioning_set_values(conditioning, {"start_percent": 0.0, "end_percent": steps_switchover_percent})
            new_conditioning += node_helpers.conditioning_set_values(noisy_embedding, {"start_percent": steps_switchover_percent, "end_percent": 1.0})
        else:
            return (noisy_embedding,)

        return (new_conditioning,)


NODE_CLASS_MAPPINGS = {
    "SeedVarianceEnhancer": SeedVarianceEnhancer
}
