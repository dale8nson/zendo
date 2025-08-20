from fastapi import APIRouter
import os
import json

router = APIRouter()

@router.get("/tokens")
async def tokens() -> dict:
    path = os.path.join(os.getcwd(), '../models/user')
    dir_paths = [os.path.join(path, d) for d in filter(lambda f: os.path.isdir(os.path.join(path, f)), os.listdir(path))]

    tokens = []
    for d in dir_paths:
        file = os.path.join(d, 'captions.json')
        if os.path.exists(file):
            with open(file) as f:
                data = json.load(f)
                tokens.extend(list(set([data[k]['token'] for k in data.keys()])))

    print(f'tokens: {tokens}')
    return  {'tokens': tokens}
