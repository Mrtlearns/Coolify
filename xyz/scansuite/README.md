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
1. In Coolify, create new application → Git repo URL: https://github.com/Mrtlearns/Coolify  
2. Base Directory: /xyz/scansuite  
3. Build Pack: **Docker Compose**  
4. Docker-Compose file: docker-compose.yml  
5. Set environment variables, deploy, verify endpoints.

## Usage  
- **preprocess-service**: POST /preprocess file upload → returns processed image + metadata.  
- **ocr-service**: POST /ocr processed image → returns JSON metadata.  
- **compose-service**: POST /compose list of image paths → returns optimized A4 PDF(s).

## Next Steps  
- Implement logic inside each service (image processing, layout).  
- Build n8n workflow: ingestion → preprocess → OCR → store → compose → send.  
- Later integrate archival system (Paperless-ngx or Receipt Wrangler) for UI/search.

## Environment Variables  
See .env.example:  


PREPROCESS_DPI=300
PREPROCESS_GRAY=true
OCR_LANG=eng
COMPOSE_PAGE_DPI=300
COMPOSE_MARGIN_MM=10



## License & Contribution  
Open-source - MIT License.
