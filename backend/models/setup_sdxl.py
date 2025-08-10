from diffusers import StableDiffusionXLPipeline


import torch

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, variant="fp16" if torch.cuda.is_available() else None)
pipe.save_pretrained("sdxl-base-1.0", max_shard_size="5GB")
