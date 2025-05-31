import torch
from torch import Tensor, ScriptModule
import torch.nn as nn
import open_clip
from open_clip import CLIP
from open_clip.transformer import VisionTransformer
from typing import cast, Tuple

model_config = open_clip.get_model_config("ViT-B-32")

# print(model_config)

# model = torch.jit.load("ViT-B-32-scripted.pt")
# print(model)
# print(model.conv1.weight.dtype)


model =  open_clip.create_model(model_name="ViT-B-32", precision="fp32", pretrained="openai", jit=False, force_quick_gelu=True
)
model = cast(CLIP, model)
visual = model.visual
vision_encoder = cast(VisionTransformer, visual.eval())

vision_encoder.conv1 = nn.Conv2d(
in_channels=3,
out_channels=vision_encoder.conv1.out_channels,
kernel_size=cast(Tuple[int, int], vision_encoder.conv1.kernel_size),
stride=cast(Tuple[int, int],vision_encoder.conv1.stride),
padding=cast(Tuple[int, int],vision_encoder.conv1.padding),
bias=vision_encoder.conv1.bias is not None
)

visual = model.visual
visual = cast(VisionTransformer, visual)

# with torch.no_grad():
#     weight = visual.conv1.weight
#     weight: Tensor = cast(Tensor, weight)
#     vision_encoder.conv1.weight.copy_(weight.float())

#     if vision_encoder.conv1.bias is not None and visual.conv1.bias is not None:
#         bias: Tensor = cast(Tensor, visual.conv1.bias)
#         vision_encoder.conv1.bias.copy_(bias.float())

#     for param in vision_encoder.parameters():
#         param.data = param.data.float()
#     for buffer_name, buffer in vision_encoder.named_buffers():
#         setattr(vision_encoder, buffer_name, buffer.float())

weight = cast(Tensor, vision_encoder.conv1.weight)
print(f"conv1 dtype: {weight.dtype}")

scripted = torch.jit.script(vision_encoder)
scripted.save("ViT-B-32-scripted.pt")

# example_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
# traced: ScriptModule = cast(ScriptModule, torch.jit.trace(vision_encoder, example_input))
# traced.save(traced, "ViT-B-32-scripted.pt")


print("TorchScript model saved.")
