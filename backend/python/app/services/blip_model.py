#!/usr/bin/env python3
"""
app/services/blip_model.py — robust BLIP loader + inference for CUDA/MPS/CPU.

- Exports:
    - CroppedImageCaptionRequest (Pydantic model)
    - async caption(image: PIL.Image) -> dict
    - cropped_image_caption(image_data: str, crop_box: tuple[int,int,int,int]) -> str

- Loads either:
    • A PEFT/LoRA adapter dir (merges when possible), or
    • A full fine-tuned BLIP folder

- Prefers processor from the FT folder to honor tokenizer/generation_config.

- Env overrides to steer verbosity:
    BLIP_NUM_BEAMS        (default 5)
    BLIP_MIN_NEW_TOKENS   (default 20)
    BLIP_MAX_NEW_TOKENS   (default 60)
    BLIP_NO_REPEAT_NGRAM  (default 3)
    BLIP_LENGTH_PENALTY   (default 0.8)  # <1 usually lengthens outputs
    BLIP_DO_SAMPLE        (default false)
    BLIP_TEMPERATURE      (default 0.9)
    BLIP_TOP_P            (default 0.95)
    BLIP_FT_DIR           (path to fine-tuned dir; default ../models/blip-output)
"""

import os
import base64
from io import BytesIO
from typing import Tuple, List

import torch
from PIL import Image, ImageOps
from pydantic import BaseModel
from transformers import BlipProcessor, BlipForConditionalGeneration

# PEFT (optional): only needed if your FT dir is a LoRA adapter.
try:
    from peft import PeftModel, PeftConfig  # type: ignore
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


# ---------------------------
# Public request model
# ---------------------------
class CroppedImageCaptionRequest(BaseModel):
    image_data: str
    crop_box: List[int]


# ---------------------------
# Device selection
# ---------------------------
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
IS_CUDA = DEVICE.type == "cuda"
IS_MPS = DEVICE.type == "mps"


# ---------------------------
# FT directory resolution
# ---------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_FT_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "models", "blip-output"))
FT_DIR = os.environ.get("BLIP_FT_DIR", _DEFAULT_FT_DIR)


# ---------------------------
# Loader
# ---------------------------
def _load_blip(ft_dir: str = FT_DIR):
    """
    Load either:
      - a PEFT (LoRA) adapter dir: apply to base and merge if supported
      - or a full fine-tuned BLIP model directory.

    Returns: (model.eval().to(DEVICE), processor, info_dict)
    """
    is_peft = False
    merged = False
    base_name = None

    if not os.path.isdir(ft_dir):
        raise FileNotFoundError(
            f"[BLIP] Fine-tune directory not found: {ft_dir}\n"
            "Set BLIP_FT_DIR or ensure the path exists."
        )

    # Detect a PEFT adapter if present
    if _HAS_PEFT:
        try:
            peft_cfg = PeftConfig.from_pretrained(ft_dir)
            is_peft = True
            base_name = peft_cfg.base_model_name_or_path
        except Exception:
            is_peft = False

    if is_peft:
        base = BlipForConditionalGeneration.from_pretrained(base_name)
        if hasattr(base, "safety_checker"):
            # Disable CLIP safety checker if present (not used for captioning)
            base.safety_checker = None
        model = PeftModel.from_pretrained(base, ft_dir)
        if hasattr(model, "safety_checker"):
            model.safety_checker = None
        try:
            model = model.merge_and_unload()
            merged = True
        except Exception:
            merged = False
    else:
        model = BlipForConditionalGeneration.from_pretrained(ft_dir)
        if hasattr(model, "safety_checker"):
            model.safety_checker = None
        base_name = getattr(model.config, "_name_or_path", "unknown-base")

    model.to(DEVICE).eval()

    # Prefer processor from FT dir; fallback to base
    try:
        processor = BlipProcessor.from_pretrained(ft_dir)
        if hasattr(processor, 'safety_checker'):
            processor.safety_checker = None
    except Exception:
        processor = BlipProcessor.from_pretrained(base_name)
        if hasattr(processor, 'safety_checker'):
            processor.safety_checker = None

    # Sanity logs
    print(f"[BLIP] ckpt_dir: {ft_dir}")
    print(f"[BLIP] device:   {DEVICE}")
    print(f"[BLIP] peft:     {is_peft} | merged: {merged}")
    print(f"[BLIP] base:     {base_name}")
    try:
        has_lora = any("lora" in n for n, _ in model.named_modules())
        print(f"[BLIP] lora layers present: {has_lora}")
    except Exception:
        pass

    info = {
        "peft": is_peft,
        "merged": merged,
        "base": base_name,
        "ft_dir": ft_dir,
        "device": str(DEVICE),
    }
    return model, processor, info


# Global model & processor (load once)
MODEL, PROCESSOR, _INFO = _load_blip(os.path.join(os.getcwd(), '../models/blip-output'))


# ---------------------------
# Generation helper
# ---------------------------
def _generate_caption(image: Image.Image) -> str:
    """
    Run BLIP generate() with params that encourage detailed captions.
    Tunable via env vars (see module docstring).
    """
    num_beams = int(os.getenv("BLIP_NUM_BEAMS", "5"))
    min_new = int(os.getenv("BLIP_MIN_NEW_TOKENS", "20"))
    max_new = int(os.getenv("BLIP_MAX_NEW_TOKENS", "77"))
    no_rep_ngram = int(os.getenv("BLIP_NO_REPEAT_NGRAM", "3"))
    length_pen = float(os.getenv("BLIP_LENGTH_PENALTY", "0.8"))  # <1.0 tends to lengthen outputs
    do_sample = os.getenv("BLIP_DO_SAMPLE", "false").lower() in ("1", "true", "yes")
    temperature = float(os.getenv("BLIP_TEMPERATURE", "0.9"))
    top_p = float(os.getenv("BLIP_TOP_P", "0.95"))

    with torch.inference_mode():
        inputs = PROCESSOR(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE, non_blocking=IS_CUDA)

        gen_kwargs = dict(
            pixel_values=pixel_values,
            min_new_tokens=min_new,
            max_new_tokens=max_new,
            length_penalty=length_pen,
        )
        if no_rep_ngram > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_rep_ngram

        if do_sample:
            gen_kwargs.update(dict(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_beams=1,  # prefer sampling OR beams, not both
            ))
        else:
            gen_kwargs.update(dict(
                do_sample=False,
                num_beams=num_beams,
            ))

        out = MODEL.generate(**gen_kwargs)
        text = PROCESSOR.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        return text


# ---------------------------
# Public API
# ---------------------------
async def caption(image: Image.Image) -> dict:
    """
    Generate a caption for a PIL image.
    Returns: {"caption": "..."}
    """
    text = _generate_caption(image.convert("RGB"))
    return {"caption": text}


def cropped_image_caption(image_data: str, crop_box: Tuple[int, int, int, int]) -> str:
    """
    image_data: base64-encoded image (data URL prefix allowed or not)
    crop_box  : (left, top, right, bottom)
    Returns: caption string
    """
    # Strip potential data URL prefix
    if "," in image_data and image_data.lstrip().startswith(("data:image/", "data:")):
        image_data = image_data.split(",", 1)[1]

    image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
    image = image.crop(tuple(map(int, crop_box)))
    image = ImageOps.pad(image, (1024, 1024), centering=(0.5, 0.5), color=0)

    return _generate_caption(image)


# ---------------------------
# Optional local smoke test
# ---------------------------
if __name__ == "__main__":
    test_path = os.getenv("BLIP_SMOKE_IMAGE", "").strip()
    if test_path and os.path.isfile(test_path):
        im = Image.open(test_path).convert("RGB")
        print("[BLIP] Smoke caption:", _generate_caption(im))
    else:
        print("[BLIP] Ready. Set BLIP_SMOKE_IMAGE to a path to test locally.")