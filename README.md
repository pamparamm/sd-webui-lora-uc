# Disable LoRA for UC
Based on [Composable LoRA](https://github.com/opparco/stable-diffusion-webui-composable-lora). Compatible with `lora_apply_weights` from current implementation of LoRA in [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

## Features

### Eliminate the impact on negative prompts
With the built-in LoRA, negative prompts are always affected by LoRA. This often has a negative impact on the output.
So this extension offers option to eliminate the negative effects.

## How to use
### Disable Lora for UC
When checked, script is preventing LoRA weights from being applied to both UC text model encoder and UC denoiser.

## Compatibilities
### WebUI
- `--always-batch-cond-uncond` must be enabled while using `--medvram` or `--lowvram`
- `Pad prompt/negative prompt to be same length` in Optimization settings must be enabled

### LyCORIS
If you are using LyCORIS, move all your LoRAs into LyCORIS folder and use them as lycos.