import torch
from diffusers import ControlNetModel
from transformers import CLIPVisionModelWithProjection

# controlnet = ControlNetModel.from_pretrained(
#     "diffusers/controlnet-canny-sdxl-1.0",
#     torch_dtype=torch.float32
# )

# controlnet.save_pretrained('controlnet-canny-sdxl-1.0')

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

image_encoder.save_pretrained("IP-Adapter-FaceID/image_encoder")
