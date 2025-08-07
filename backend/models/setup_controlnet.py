import torch
from diffusers import ControlNetModel

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float32
)

controlnet.save_pretrained('controlnet-canny-sdxl-1.0')
