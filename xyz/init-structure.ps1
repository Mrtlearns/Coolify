# -------------------------------------------------------------
# Create and populate the scansuite folder from within xyz
# -------------------------------------------------------------

# Ensure current directory is `xyz`
$cwd = Get-Location
Write-Host "Running from folder: $cwd"

# Create the scansuite folder and service sub-folders
New-Item -ItemType Directory -Path "scansuite\preprocess-service", "scansuite\compose-service", "scansuite\ocr-service", "scansuite\volumes" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path "scansuite\preprocess-service\app", "scansuite\compose-service\app", "scansuite\ocr-service\app" -ErrorAction SilentlyContinue

# Create top-level stub files inside scansuite folder
New-Item -ItemType File -Path "scansuite\.env.example" -Force
New-Item -ItemType File -Path "scansuite\README.md" -Force
New-Item -ItemType File -Path "scansuite\docker-compose.yml" -Force

# Create each service’s stub files
# Preprocess service
New-Item -ItemType File -Path "scansuite\preprocess-service\Dockerfile" -Force
New-Item -ItemType File -Path "scansuite\preprocess-service\requirements.txt" -Force
New-Item -ItemType File -Path "scansuite\preprocess-service\app\main.py" -Force

# Compose service
New-Item -ItemType File -Path "scansuite\compose-service\Dockerfile" -Force
New-Item -ItemType File -Path "scansuite\compose-service\requirements.txt" -Force
New-Item -ItemType File -Path "scansuite\compose-service\app\main.py" -Force

# OCR service
New-Item -ItemType File -Path "scansuite\ocr-service\Dockerfile" -Force
New-Item -ItemType File -Path "scansuite\ocr-service\requirements.txt" -Force
New-Item -ItemType File -Path "scansuite\ocr-service\app\main.py" -Force

# Create volumes README placeholder
New-Item -ItemType File -Path "scansuite\volumes\README-VOL.md" -Force

# Define the content for each stub file
$dockerComposeContent = @"
version: "3.8"

services:
  preprocess-service:
    build:
      context: ./preprocess-service
      dockerfile: Dockerfile
    image: scansuite/preprocess-service:latest
    restart: unless-stopped
    environment:
      - PREPROCESS_DPI=300
      - PREPROCESS_GRAY=true
    volumes:
      - receipts_data:/data/receipts
    networks:
      - scansuite_net
    ports:
      - "8001:8000"

  ocr-service:
    build:
      context: ./ocr-service
      dockerfile: Dockerfile
    image: scansuite/ocr-service:latest
    restart: unless-stopped
    environment:
      - OCR_LANG=eng
    volumes:
      - receipts_data:/data/receipts
    networks:
      - scansuite_net
    ports:
      - "8002:8000"

  compose-service:
    build:
      context: ./compose-service
      dockerfile: Dockerfile
    image: scansuite/compose-service:latest
    restart: unless-stopped
    environment:
      - COMPOSE_PAGE_DPI=300
      - COMPOSE_MARGIN_MM=10
    volumes:
      - receipts_data:/data/receipts
      - output_pdfs:/data/output
    networks:
      - scansuite_net
    ports:
      - "8003:8000"

volumes:
  receipts_data:
  output_pdfs:

networks:
  scansuite_net:
    driver: bridge
"@
Set-Content -Path "scansuite\docker-compose.yml" -Value $dockerComposeContent

$envExampleContent = @"
# .env.example – environment variables for ScanSuite stack

# Preprocess service
PREPROCESS_DPI=300
PREPROCESS_GRAY=true

# OCR service
OCR_LANG=eng

# Compose service
COMPOSE_PAGE_DPI=300
COMPOSE_MARGIN_MM=10
"@
Set-Content -Path "scansuite\.env.example" -Value $envExampleContent

$readmeContent = @"
# ScanSuite: Receipt & Document Pipeline  
*Version 1.0 – Master-plan pipeline for receipt scanning, processing, layout, PDF export*

## Overview  
This repository contains a multi-service Docker Compose stack for automated receipt processing:
- **preprocess-service**: converts uploaded image files (JPEG/HEIC) to standardized format; auto-crop, deskew and rotate receipts.
- **ocr-service**: performs OCR on processed images and exports extracted text/metadata, ready for further workflow integration.
- **compose-service**: accepts one or more processed receipt images and generates one or more A4-sized PDF pages (optimized for emailing) with layout and compression.

The stack is designed for orchestration via n8n and is self-hosted using Coolify for deployment.

## Repository Structure  


scansuite/
├── docker-compose.yml
├── .env.example
├── README.md
├── preprocess-service/
│   ├── Dockerfile
│   ├── app/
│   └── requirements.txt
├── ocr-service/
│   ├── Dockerfile
│   ├── app/
│   └── requirements.txt
├── compose-service/
│   ├── Dockerfile
│   ├── app/
│   └── requirements.txt
└── volumes/
└── README-VOL.md



## Deployment via Coolify  
1. In Coolify, create new application → Git repo URL: `https://github.com/Mrtlearns/Coolify`  
2. Base Directory: `/xyz/scansuite`  
3. Build Pack: **Docker Compose**  
4. Docker-Compose file: `docker-compose.yml`  
5. Set environment variables, deploy, verify endpoints.

## Usage  
- **preprocess-service**: `POST /preprocess` file upload → returns processed image + metadata.  
- **ocr-service**: `POST /ocr` processed image → returns JSON metadata.  
- **compose-service**: `POST /compose` list of image paths → returns optimized A4 PDF(s).

## Next Steps  
- Implement logic inside each service (image processing, layout).  
- Build n8n workflow: ingestion → preprocess → OCR → store → compose → send.  
- Later integrate archival system (Paperless-ngx or Receipt Wrangler) for UI/search.

## Environment Variables  
See `.env.example`:  


PREPROCESS_DPI=300
PREPROCESS_GRAY=true
OCR_LANG=eng
COMPOSE_PAGE_DPI=300
COMPOSE_MARGIN_MM=10



## License & Contribution  
Open-source - MIT License.
"@
Set-Content -Path "scansuite\README.md" -Value $readmeContent

# Populate service stub files
$ppDockerfile = @"
# Version 1.0 – stub for preprocess service
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
"@
Set-Content -Path "scansuite\preprocess-service\Dockerfile" -Value $ppDockerfile

$ppRequirements = @"
fastapi
uvicorn[standard]
opencv-python-headless
numpy
pillow
python-multipart
"@
Set-Content -Path "scansuite\preprocess-service\requirements.txt" -Value $ppRequirements

$ppMain = @"
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
from PIL import Image
import io
import base64

app = FastAPI()

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/preprocess')
async def preprocess(file: UploadFile = File(...)):
    # stub logic – to be implemented
    data = await file.read()
    return {'message': 'stub – image received', 'filename': file.filename}
"@
Set-Content -Path "scansuite\preprocess-service\app\main.py" -Value $ppMain

$csDockerfile = @"
# Version 1.0 – stub for compose service
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
"@
Set-Content -Path "scansuite\compose-service\Dockerfile" -Value $csDockerfile

$csRequirements = @"
fastapi
uvicorn[standard]
reportlab
Pillow
python-multipart
"@
Set-Content -Path "scansuite\compose-service\requirements.txt" -Value $csRequirements

$csMain = @"
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
"@
Set-Content -Path "scansuite\compose-service\app\main.py" -Value $csMain

$ocrDockerfile = @"
# Version 1.0 – stub for OCR service
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
"@
Set-Content -Path "scansuite\ocr-service\Dockerfile" -Value $ocrDockerfile

$ocrRequirements = @"
fastapi
uvicorn[standard]
pytesseract
Pillow
python-multipart
"@
Set-Content -Path "scansuite\ocr-service\requirements.txt" -Value $ocrRequirements

$ocrMain = @"
from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI()

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/ocr')
async def ocr(file: UploadFile = File(...)):
    # stub logic – to be implemented
    return {'message': 'stub – image received for OCR', 'filename': file.filename}
"@
Set-Content -Path "scansuite\ocr-service\app\main.py" -Value $ocrMain

Write-Host "All stub files created with boilerplate content in scansuite folder. Edit each service to implement logic."

