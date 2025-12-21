
![header](../../wiki/images/SeedVarianceEnhancer_GitHub_header.webp)

https://github.com/user-attachments/assets/bdbbfa83-f8ec-4cab-81cf-36a6820a51e5


# SeedVarianceEnhancer v2.2

SeedVarianceEnhancer is a ComfyUI custom node designed to add diversity to the outputs of Z-Image Turbo. It compensates for low seed variance, which is when generated images look similar despite being generated with different seeds. It works by adding random noise to the embedding for the early generation steps.


## Installation:

Extract the contents of the zip file and place the SeedVarianceEnhancer directory in your ComfyUI/custom_nodes directory. Launch or restart ComfyUI. You will find SeedVarianceEnhancer inside the advanced/conditioning node group.


## Usage:

The node should be placed next in line after the positive prompt "CLIP Text Encode (Prompt)" node. The conditioning output should connect to the sampler's positive prompt input. The default settings should work well with Z-Image Turbo. The node has been tested to work with the workflow from [https://github.com/comfyanonymous/ComfyUI_examples/tree/master/z_image](https://github.com/comfyanonymous/ComfyUI_examples/tree/master/z_image).

The randomize_percent setting determines what percentage of embedding values will be randomly selected for modification with noise. Strength adjusts the scale of the noise. The seed determines which values to modify and with what noise. Noise_insert is set to "noise on beginning steps" by default. By removing the noise before the end of the generation, the model will have time to pivot back towards prompt adherence. "Noise on all steps" will reduce prompt adherence and text rendering significantly. Steps_switchover_percent specifies what percentage of steps are processed before switching between noisy and original embeddings.

Strength values in the range of 15 to 30 and a randomize_percent of 50 are reasonable starting points for use with Z-Image Turbo. The steps_switchover_percent defaults to 20%. Optimal settings vary depending on the prompt. Some prompts will be adversely affected when using the default settings. Experimentation may be required to achieve good results. An often effective alternate configuration is to decrease steps_switchover_percent to 10 and increase strength to 40.

To ensure a specific number of steps are used before the switch, use this formula: (100/TOTALSTEPS) * STEPS - 1.

Longer more detailed prompts provide more embedding values for SeedVarianceEnhancer to manipulate. It is somewhat counterintuitive, but more detailed prompts may result in more diverse outputs.


## Advanced usage:

SeedVarianceEnhancer includes an embedding masking feature, which excludes portions of the embedding from exposure to noise. Masking can protect portions of the prompt. The mask_starts_at setting controls whether the mask extends from the beginning or the end. Mask_percent controls how far the mask extends. Setting mask_starts_at to beginning and mask_percent to 50 will mask the first half of the prompt. If the prompt is structured so that critical details are in the first half, and less important details are in the second half, the image generations should more consistently adhere to the critical details.

An image can be reproduced exactly if an old workflow is reused. The prompt can be revised while maintaining a consistent output composition provided guidelines are followed. The embedding values most maintain alignment with the noise in order to produce a similar image. Appending text to the end of the prompt maintains alignment. Making single word substitutions, such as changing a color, may work but if the resulting embedding values become offset, the output image will no longer maintain consistency.

SeedVarianceEnhancer nodes can be chained together. The noise from each node will add up. Chaining does not produce more than one switchover event. The switchover schedule will follow the last node's setting.


Strength values can be set to negative values, which will invert the noise added to the embedding.


## Use with other models:

SeedVarianceEnhancer should work with other models, but the strength value may need to be adjusted. The text encoder is a major factor. The node includes a feature that outputs statistics about the embedding to the console when log_to_console is set to true. The standard deviation is usually within an order of magnitude of the optimal strength setting. The node will detect embeddings that are padded with null byte sequences and automatically mask that region from noise.


## Results:

When used with Z-Image Turbo, the default settings will add a moderate amount of diversity at the cost of some prompt adherence. Higher "randomize_percent" and "strength," as well as the use of "noise on all steps," will all substantially increase diversity while worsening prompt adherence. The reduced prompt adherence may be desirable to users interested in generating highly diverse, strange, or unexpected images. Using very high settings offers the potential to discover unknown capabilities and knowledge within the model.


## Limitations:

The node's effects are inconsistent. No single set of settings have proven to be universally beneficial. The node is not a "set and forget" solution and it should not be used unless needed.

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

v2.0 -> v2.1

* more tooltip revisions
* new limits prevent settings of:
    - randomize_percent to 0
    - mask_percent to 100
    - steps_switchover_percent to 0 or 100

* embedding statistics now includes information about null byte sequences
* added automatic masking of null sequences at the end of embeddings
    - Some text encoders, such as umt5-xxl (used by Wan), pad the end of embeddings with zeros.
    - Adding noise to those regions causes adverse effects. Doing so is now avoided.

v2.1 -> v2.2

The focus of v2.2 is to facilitate making variations upon variations.

* The random number generator is now reset with seed+1 after noise generation and before value selection.
    - This makes the noise alignment independent of embedding length.
    - The output will be more consistent when regenerating with an extended prompt.
    - The new seed behavior breaks compatibility with old generations.
    - Adding 1 billion to strength will revert to the old seed behavior and allow regeneration of old images.
* improved functionality when chaining SeedVarianceEnhancer nodes
    - Chaining to a node set to "noise on ending steps" is now supported.
    - Chaining a "noise on beginning steps" to another "noise on beginning steps" will retain the original embedding for the end steps
    - Chaining a "noise on ending steps" to a "noise on beginning steps" results in both the beginning and ending steps being exposed to separately controlled noise.


## Links

The wiki contains images demonstrating usage and the effects

[https://github.com/ChangeTheConstants/SeedVarianceEnhancer/wiki](https://github.com/ChangeTheConstants/SeedVarianceEnhancer/wiki)

Report issues here:

[https://github.com/ChangeTheConstants/SeedVarianceEnhancer/issues](https://github.com/ChangeTheConstants/SeedVarianceEnhancer/wiki)

Here users are encouraged to share their experiences, including the settings and workflows they found effective:

[https://github.com/ChangeTheConstants/SeedVarianceEnhancer/discussions](https://github.com/ChangeTheConstants/SeedVarianceEnhancer/discussions)


## License

SeedVarianceEnhancer is released under the MIT No Attribution license.

