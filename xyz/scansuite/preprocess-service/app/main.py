import os
import io
import base64
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

app = FastAPI()

RECEIPT_DIR = "/data/receipts"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/preprocess")
async def preprocess(
    file: UploadFile = File(...),
    gray: bool = True,
    crop: bool = True,
    deskew: bool = True,
    threshold: bool = True,
    dpi: int = 300,
    rotation_threshold: float = 2.0,
    padding: int = 10,
    quality: int = 90
):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")

        # Validate parameters
        if rotation_threshold < 0:
            raise HTTPException(status_code=400, detail="rotation_threshold must be >= 0")
        if padding < 0:
            raise HTTPException(status_code=400, detail="padding must be >= 0")
        if not (1 <= quality <= 100):
            raise HTTPException(status_code=400, detail="quality must be between 1 and 100")
        if dpi <= 0:
            raise HTTPException(status_code=400, detail="dpi must be > 0")

        contents = await file.read()
        try:
            pil_image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

        # Convert to OpenCV BGR
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Convert to grayscale if requested
        if gray:
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            cv_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        else:
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Resize large image for performance
        height, width = cv_image.shape[:2]
        if height > 1000:
            ratio = height / 1000.0
            new_h = 1000
            new_w = int(width / ratio)
            cv_image = cv2.resize(cv_image, (new_w, new_h))
            gray_image = cv2.resize(gray_image, (new_w, new_h))

        # 1) Detect and correct rotation (deskew) if requested
        if deskew:
            rotated_image, rotation_angle = detect_and_correct_rotation(cv_image, rotation_threshold)
        else:
            rotated_image, rotation_angle = cv_image.copy(), 0
        
        # 2) Edge detection and contour finding
        gray_img = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_img, (5,5), 0)
        edged = cv2.Canny(blur, 75, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        num_contours_examined = len(contours)
        screenCnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        if crop and screenCnt is None:
            # Unable to find rectangle contour, apply smart crop to detect edges
            warped = smart_crop(rotated_image, padding)
            if warped is None or warped.size == 0:
                warped = rotated_image.copy()
            did_crop = True
        elif crop and screenCnt is not None:
            pts = screenCnt.reshape(4,2)
            rect = order_points(pts)
            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = max(int(widthA), int(widthB))
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = max(int(heightA), int(heightB))
            
            # Ensure dimensions are valid
            if maxWidth <= 0 or maxHeight <= 0:
                warped = rotated_image.copy()
                did_crop = False
            else:
                dst = np.array([
                    [0, 0],
                    [maxWidth-1, 0],
                    [maxWidth-1, maxHeight-1],
                    [0, maxHeight-1]
                ], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(rotated_image, M, (maxWidth, maxHeight))
                did_crop = True
        else:
            warped = rotated_image.copy()
            did_crop = False

        # 3) Threshold / cleanup
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        if threshold:
            T = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            processed_image = cv2.cvtColor(T, cv2.COLOR_GRAY2BGR)
        else:
            processed_image = warped.copy()

        # Save processed image
        try:
            os.makedirs(RECEIPT_DIR, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {RECEIPT_DIR}: {e}")
            RECEIPT_DIR_ALT = "/tmp/receipts"
            os.makedirs(RECEIPT_DIR_ALT, exist_ok=True)
            processed_path = os.path.join(RECEIPT_DIR_ALT, f"processed_{file.filename}")
        else:
            processed_path = os.path.join(RECEIPT_DIR, f"processed_{file.filename}")
        
        pil_out = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        pil_out.save(processed_path, format="JPEG", quality=quality)

        width_final, height_final = pil_out.size
        processed_filename = os.path.basename(processed_path)
        
        # Read the processed image as base64
        with open(processed_path, "rb") as f:
            img_bytes = f.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        # Additional flags (for logging/diagnostics)
        return {
            "message": "image processed",
            "original_filename": file.filename,
            "processed_filename": processed_filename,
            "file_path": processed_path,
            "width": width_final,
            "height": height_final,
            "num_contours_examined": num_contours_examined,
            "did_crop": did_crop,
            "rotation_angle": float(rotation_angle),
            "processing_flags": {
                "gray": gray,
                "crop": crop,
                "deskew": deskew,
                "threshold": threshold
            },
            "processing_params": {
                "rotation_threshold": rotation_threshold,
                "padding": padding,
                "quality": quality,
                "dpi": dpi
            },
            "image": {
                "base64": img_base64,
                "mime_type": "image/jpeg"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = f"Error processing image: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # This will appear in container logs
        raise HTTPException(status_code=500, detail=str(e))

def detect_and_correct_rotation(image, rotation_threshold=2.0):
    """
    Detect and correct rotation of document in image using edge detection
    and line angle analysis.
    
    Args:
        image: Input image
        rotation_threshold: Minimum angle (in degrees) to trigger rotation. Default: 2.0
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)
    
    # Use Hough line transform to detect lines and their angles
    lines = cv2.HoughLines(edged, 1, np.pi / 180, 100)
    
    if lines is None or len(lines) == 0:
        return image, 0
    
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta)
        # Convert angle to range [-90, 90] for easier processing
        if angle > 90:
            angle = angle - 180
        angles.append(angle)
    
    # Get median angle (more robust than mean)
    median_angle = np.median(angles)
    
    # Only rotate if angle is significant (> rotation_threshold)
    if abs(median_angle) < rotation_threshold:
        return image, 0
    
    # Rotate image to correct the angle
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    
    # Calculate new dimensions after rotation
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for the new dimensions
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    return rotated_image, median_angle


def smart_crop(image, padding=10):
    """
    Smart crop to remove white/empty borders around document.
    Detects the actual content area and crops to it.
    
    Args:
        image: Input image
        padding: Padding around the detected content in pixels. Default: 10
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert and threshold
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Get bounding box of largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    # Crop the image
    cropped = image[y:y+h, x:x+w]
    
    return cropped


def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
