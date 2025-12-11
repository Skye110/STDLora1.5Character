"""
Pixel Art LoRA - Usage Example
"""

import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from torch import Generator

def load_model(lora_path="./pxstyle_lora", lora_strength=0.85):
    """Load the model with LoRA"""
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    
    pipe.load_lora_weights(lora_path, adapter_name="pxstyle")
    pipe.set_adapters("pxstyle", lora_strength)
    
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config
    )
    pipe.enable_attention_slicing()
    
    return pipe

def generate_character(
    pipe,
    character_type="warrior",
    seed=42,
    steps=28,
    guidance=7.5
):
    """Generate a pixel art character"""
    
    prompt = f"single pixel art {character_type}, idle pose, centered, isolated, <pxstyle>"
    negative = "multiple characters, two people, group, sprite sheet, realistic, photo, 3d"
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=Generator("cuda").manual_seed(seed),
        height=512,
        width=512,
    ).images[0]
    
    return image

if __name__ == "__main__":
    # Load model
    print("Loading model...")
    pipe = load_model()
    
    # Generate characters
    characters = ["knight", "mage", "rogue", "archer", "paladin"]
    
    for i, char in enumerate(characters):
        print(f"Generating {char}...")
        img = generate_character(pipe, char, seed=42+i)
        img.save(f"{char}.png")
        print(f"Saved {char}.png")
    
    print("Done!")
