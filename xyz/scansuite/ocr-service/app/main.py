from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pytesseract
from PIL import Image
import io
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR Service", version="1.0")

@app.get('/health')
def health():
    """Health check endpoint."""
    return {'status': 'ok', 'service': 'ocr', 'version': '1.0'}

@app.post('/ocr')
async def ocr(file: UploadFile = File(...)):
    """
    Extract text from image using Tesseract OCR.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON with extracted text, confidence, and metadata
    """
    try:
        # Read image file
        contents = await file.read()
        
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        # Open image with PIL
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
        # Convert to RGB if necessary (handle RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Processing image: {file.filename}, Mode: {image.mode}, Size: {image.size}")
        
        # Extract text using Tesseract
        try:
            # Extract text
            extracted_text = pytesseract.image_to_string(image, lang='eng')
            
            # Get confidence data
            try:
                data = pytesseract.image_to_data(image, lang='eng', output_type=pytesseract.Output.DICT)
                # Calculate confidence (average of detected text confidence)
                confidences = [int(conf) for conf in data['confidence'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            except Exception as conf_e:
                logger.warning(f"Could not calculate confidence: {str(conf_e)}")
                avg_confidence = 0
            
            # Clean up extracted text
            extracted_text = extracted_text.strip()
            
            logger.info(f"✅ OCR Success: {len(extracted_text)} characters extracted, Confidence: {avg_confidence:.1f}%")
            
            return {
                'status': 'success',
                'filename': file.filename,
                'text': extracted_text,
                'confidence': avg_confidence,
                'char_count': len(extracted_text),
                'word_count': len(extracted_text.split()),
                'message': 'Text successfully extracted from image'
            }
            
        except Exception as e:
            logger.error(f"❌ Tesseract Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

