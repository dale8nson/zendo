from fastapi import APIRouter
from app.services.training import DatasetPostRequest, create_set

router = APIRouter()

@router.post("/dataset")
async def post_dataset(request: DatasetPostRequest):
   print(f"request: {request}")
   result = await create_set(request.image_data, request.masks, request.collection, request.token, request.caption)

   return result
