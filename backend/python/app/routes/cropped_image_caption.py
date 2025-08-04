from fastapi import APIRouter
from app.services.blip_model import cropped_image_caption, CroppedImageCaptionRequest

router = APIRouter()

@router.post("/cropped-image-caption")
async def get_cropped_image_caption(request: CroppedImageCaptionRequest):
    image_data, crop_box = request.image_data, request.crop_box
    caption = cropped_image_caption(image_data, crop_box)
    print(f"Generated caption: {caption}")
    return {"caption": caption}
