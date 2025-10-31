from fastapi import FastAPI, HTTPException
from typing import List
import io

app = FastAPI()

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/compose')
async def compose(image_paths: List[str]):
    # stub logic – to be implemented
    return {'message': 'stub – received image paths', 'paths': image_paths}
