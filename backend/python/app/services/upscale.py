import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import os
import base64
from io import BytesIO
from PIL import Image, ImageOps
from PIL.Image import Resampling
import cv2
import numpy as np
from typing import List, Any, cast
from pydantic import BaseModel

def extract_base64_data(data_url: str) -> str:

    b64 = data_url.strip().replace('\n', '').replace(' ', '')
    pad = len(b64) % 4
    if pad:
        b64 += '=' * (4 - pad)
    return b64


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def upscale(image_data: str):
    # model = torch.load(os.path.join(os.getcwd(), "../models/UltraSharp/4x-UltraSharp.pth"), weights_only=True).to(device, dtype=dtype)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    upsampler = RealESRGANer(
            scale=4,
            model_path=os.path.join(os.getcwd(), "../models/Real-ESRGAN/RealESRGAN_x4plus.pth"),
            device=device,
            model=model,
            dni_weight=[0.5, 0.5]
    )

    b64 = extract_base64_data(image_data)
    bytes = base64.b64decode(b64)
    image = Image.open(BytesIO(bytes)).convert("RGB")
    print(f"image: {image}")
    image.save(os.path.join(os.getcwd(), "app/test_images/upscaler_input.png"))
    # image = image.resize((int(image.width * 0.25), int(image.height * 0.25)), resample=Resampling.BOX)
    # print(f"image: {image}")
    # image.save(os.path.join(os.getcwd(), "app/test_images/upscaler_scaled_down.png"))
    # image = image.resize((int(image.width * 4), int(image.height * 4)), resample=Resampling.BOX)

    # image = image.resize((int(image.width * 0.0625), int(image.height * 0.0625)), resample=Resampling.BICUBIC)

    # image.save(os.path.join(os.getcwd(),"app/test_images/upscaler_scaled_up.png"))
    # print(f"image: {image}")
    # image = image.resize((int(image.width * 0.25), int(image.height * 0.25)), resample=Resampling.BOX)


    buf = BytesIO()
    image.save(buf, format="jpeg")
    print(f"image: {image}")
    buf.seek(0)
    print(f"buf: {buf}")
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)

    print(f"arr.shape: {arr.shape} arr.size: {arr.size}")
    print(f"arr.shape: {arr.shape}")

    im = cv2.imdecode(arr, cv2.IMREAD_COLOR_RGB)

    print(f"im.shape: {im.shape}")

    output, _ = upsampler.enhance(im, outscale=4)

    print(f"output: {output}")

    # output = arr

    image = Image.fromarray(output).convert("RGBA")
    image = image.reduce(4)

    image.save(os.path.join(os.getcwd(), "app/test_images/upscaler_output.png"))

    buf = BytesIO()
    image.save(buf, format="png")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')

    return b64


class Layer(BaseModel):
    selected: bool
    label: str
    visible: bool
    opacity: float
    currentLayerHistoryIndex: int
    history: List[Any]

class UpscaleRequest(BaseModel):
    layers: List[Layer]

def composite_layers(layers: List[Layer]):

    root_layer = cast(Layer, [l if l.label == "root" else None for l in layers][0])
    root_history_index = root_layer.currentLayerHistoryIndex
    x1, y1, x2, y2 = root_layer.history[root_history_index]["bbox"]

    width, height = x2 - x1, y2 - y1
    size = max(width, height)
    mx, my = (size - width) // 2, (size - height) // 2

    image = Image.new("RGB", (width, height))

    for layer in layers:

        history_index = layer.currentLayerHistoryIndex
        history = layer.history
        image_data = history[history_index]["imageData"]
        print(f"image_data: {image_data[0:100]}")
        bbox = history[history_index]["bbox"]

        bbox = [int(n) for n in bbox]
        print(f"bbox: {bbox}, bbox width: {bbox[2] - bbox[0]} bbox height: {bbox[3] - bbox[1]}")

        b64 = extract_base64_data(image_data)
        image_bytes = base64.b64decode(b64)
        bbox = [bbox[i] - my if i % 2 else bbox[i] - mx for i in range(len(bbox))]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        scale_x = 1024 / w
        scale_y = 1024 / h
        scale = min(scale_x, scale_y)
        # bbox = [int(n * scale) for n in bbox]

        layer_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        print(f"layer_image width: {layer_image.width} layer_image height: {layer_image.height}")

        image.paste(layer_image, box=bbox)

    image = image.convert("RGBA")
    scale_x = 1024 / width
    scale_y = 1024 / height
    scale = min(scale_x, scale_y)


    image = image.resize((int(width * scale), int(height * scale)))
    print(f"image: {image}")
    mx, my = (1024 - image.width) // 2, (1024 - image.height) // 2
    bbox = [int(n * scale) for n in bbox]
    bbox = [ bbox[i] + my if i % 2 else bbox[i] + mx for i in range(len(bbox))]

    image.save(os.path.join(os.getcwd(), "app/test_images/composite_layers_output.png"))

    buf = BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    image_data = base64.b64encode(buf.read()).decode('utf-8')

    return image_data, bbox
