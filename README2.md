# Pixel Art LoRA Model

## What's Included:
- `pxstyle_lora_model.zip` - Your trained LoRA model
- `test_images.zip` - All generated test images
- `training_dataset.zip` - The dataset used for training
- `usage_code.py` - Code to use the model

## How to Use the Model:

### Installation:
```bash
pip install diffusers transformers accelerate safetensors torch
```

### Basic Usage:
```python
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from torch import Generator

# Load base model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

# Load your LoRA (unzip pxstyle_lora_model.zip first)
pipe.load_lora_weights("./pxstyle_lora", adapter_name="pxstyle")
pipe.set_adapters("pxstyle", 0.85)

# Better scheduler
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config
)

# Generate
prompt = "single pixel art warrior, idle pose, centered, <pxstyle>"
negative = "multiple characters, group, realistic, photo, 3d"

image = pipe(
    prompt=prompt,
    negative_prompt=negative,
    num_inference_steps=28,
    guidance_scale=7.5,
    generator=Generator("cuda").manual_seed(42),
).images[0]

image.save("output.png")
```

## Tips:
- Always use `<pxstyle>` trigger word in prompts
- Start prompts with "single" or "solo" for one character
- Adjust LoRA strength: 0.5-1.0 (default 0.85)
- Guidance scale: 7.5-9.0 for cleaner results
- Use negative prompt: "multiple characters, group, sprite sheet"

## Checkpoints:
The model includes checkpoints at 500, 1000, 1500, 2000 steps.
If final model is overfitted, try checkpoint-1000 or checkpoint-500.

Trained on: Pixel Art 2D Game Character Sprites (Idle poses)
Base Model: Stable Diffusion v1.5
Training Steps: 2000
