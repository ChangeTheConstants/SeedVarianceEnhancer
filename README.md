
# SeedVarianceEnhancer v2.0

SeedVarianceEnhancer is a ComfyUI custom node designed to add diversity to the outputs of Z-Image Turbo. It compensates for low seed variance, which is when generated images look very similar despite being generated with different seeds. It works by adding random noise to the embedding for the early generation steps.


## Installation:

Extract the contents of the zip file and place the SeedVarianceEnhancer directory in your ComfyUI/custom_nodes directory. Launch or restart ComfyUI. You will find the SeedVarianceEnhancer inside the advanced/conditioning node group.


## Usage:

The node should be placed next in line after the positive prompt "CLIP Text Encode (Prompt)" node. The conditioning output should then connect to the sampler's positive prompt input. The default settings should work well with Z-Image Turbo. The node has been tested to work with the workflow from [https://github.com/comfyanonymous/ComfyUI_examples/tree/master/z_image](https://github.com/comfyanonymous/ComfyUI_examples/tree/master/z_image).

The randomize_percent setting determines what percentage of embedding values will be randomly selected for modification with noise. Strength adjusts the scale of the noise. The seed determines which values to modify and with what noise. Noise_insert is set to "noise on beginning steps" by default. By removing the noise before the end of the generation, the model will have time to pivot back towards prompt adherence. "Noise on all steps" will reduce prompt adherence and text rendering significantly. Steps_switchover_percent specifies what percentage of steps are processed before switching between noisy and original embeddings.

Strength values in the range of 15 to 30 and a randomize_percent of 50 are reasonable starting points for use with Z-Image Turbo. The steps_switchover_percent defaults to 20%. Use this formula to calculate which percentage setting to use: (100/TOTALSTEPS) * STEPS - 1. Some prompts will benefit from higher or lower settings. Longer more detailed prompts provide more embedding values for SeedVarianceEnhancer to manipulate. Somewhat counterintuitively, use with more detailed prompts may result in more diverse outputs.


## Advanced usage:

SeedVarianceEnhancer includes an embedding masking feature, which excludes portions of the embedding from exposure to noise. Masking can protect portions of the prompt. The mask_starts_at setting controls whether the masks extends from the beginning or end. Mask_percent controls how far the mask extends. Setting mask_starts_at to beginning and mask_percent to 50 will mask the first half of the prompt. If the prompt is structured so that critical details are in the first half and less important details are in the second, the image generations should more consistently adhere to the critical details.

SeedVarianceEnhancer nodes can be chained together. The noise from each node will add up. The node only uses the first embedding of a conditioning, so chaining to a node set to "noise at end steps" will not work.

Strength values can be set to negative values which will invert the noise added to the embedding.


## Use with other models:

SeedVarianceEnhancer should work with other models, but the strength value may need to be adjusted. Which text encoder is being used is a major factor. The node includes a feature that outputs statistics about the embedding to the console when log_to_console is set to true. The standard deviation is usually within an order of magnitude of the optimal strength setting.


## Results:

When used with Z-Image Turbo, the default settings will add a moderate amount of diversity at the cost of a some prompt adherence. Higher "randomize_percent" and "strength," as well as the use of "noise on all steps," will all substantially increasing deversity while worsening prompt adherence. The reduced prompt adherence may be desirable to users interested in generating highly diverse, strange, or unexpected images. Using very high settings offers the potential to discover unknown capabilities and knowledge within the model.


## Limitations:

SeedVarianceEnhancer does not properly handle all conditioning inputs, such as those containing multiple embeddings, so using it alongside other conditioning nodes may result in unexpected behavior.


## Change log:

v1.0 -> v2.0

* cleaned up code structure
* added input validity checks
* widened strength value limits, and increased precision
* revised tooltips text
* added logging to console toggle control
* added statistical analysis of embedding tensor feature
* added "disabled" setting to noise_insert control.
* changed category from conditioning to advanced/conditioning
* implemented the masking feature


## License

SeedVarianceEnhancer is released under the MIT No Attribution license.
