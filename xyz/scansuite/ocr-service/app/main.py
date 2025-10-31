from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI()

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/ocr')
async def ocr(file: UploadFile = File(...)):
    # stub logic – to be implemented
    return {'message': 'stub – image received for OCR', 'filename': file.filename}
