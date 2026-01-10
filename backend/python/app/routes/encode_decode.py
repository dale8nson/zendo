from fastapi import APIRouter
from typing import Union, List, Dict
from pydantic import BaseModel
from ..services.SDXL import Layer, encode_decode as encdec

class EncodeDecodeRequest(BaseModel):
    image_data: list | str

router = APIRouter()

@router.post('/encode_decode')
async def encode_decode(data: EncodeDecodeRequest) -> dict:

    image_data =  await encdec(data.image_data)

    return {'image_data': image_data}
