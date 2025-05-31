from numpy._core.numeric import ndarray
import torch
import open_clip
from PIL import Image
from torchvision import transforms
import time
from typing import cast
import numpy as np
from zendolib import PyClipModel
from sympy.physics.tests.test_qho_1d import nu

image_path = "../backend/python/app/uploads/6e0fbd28-e451-4dc8-90b5-6fc25b25c106.webp"

preprocess = transforms.Compose([
transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(
    mean=[0.48145, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711]
    )
])
img = Image.open(image_path)
img_tensor = cast(torch.Tensor, preprocess(img))
img_tensor = img_tensor.unsqueeze(0)

img = img.copy()

model_eager, _, _ = open_clip.create_model_and_transforms(
"ViT-B-32", pretrained="openai"
)
model_eager.eval()

start = time.perf_counter()
for _ in range(10):
    with torch.no_grad():
        _ = model_eager.encode_image(img_tensor)
end = time.perf_counter()

print(f"PyTorch (eager): {(end - start) / 10:.4f}s per image")

scripted = torch.jit.load("../backend/models/openclip/ViT-B-32-scripted.pt")
scripted.eval()

start = time.perf_counter()
for _ in range(10):
    with torch.no_grad():
        _ = scripted(img_tensor)
end = time.perf_counter()
print(f"TorchScript (Python): {(end - start) / 10:.4f}s per image")

transform = transforms.Compose([
transforms.Resize(224),
transforms.CenterCrop(224),
transforms.ToTensor(),

])

N = 32
dummy_np = np.random.rand(N, 3, 224, 224).astype(np.float32)
# dummy_np = dummy_tensor.numpy()

assert dummy_np.flags['C_CONTIGUOUS']
assert dummy_np.dtype == np.float32

model = PyClipModel("../backend/models/openclip/ViT-B-32-scripted.pt")

_ = model.predict(dummy_np)


start = time.perf_counter()
for _ in range(10):
    with torch.no_grad():
        _ = model.predict(img_tensor.numpy())
end = time.perf_counter()
print(f"PyClip (Rust): {(end - start) / 10:.4f}s per image")
