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

        # STEP 1: First crop dark borders from original image (rough crop)
        # This removes the large black areas before rotation
        if crop:
            rough_crop = smart_crop(cv_image, padding=20)
            if rough_crop is None or rough_crop.size == 0:
                rough_crop = cv_image.copy()
        else:
            rough_crop = cv_image.copy()
        
        # STEP 2: Detect and correct rotation on the roughly-cropped image
        # Now rotation happens on an image without large black borders
        if deskew:
            rotated_image, rotation_angle = detect_and_correct_rotation(rough_crop, rotation_threshold)
        else:
            rotated_image, rotation_angle = rough_crop.copy(), 0
        
        # STEP 3: Apply final precise crop to remove any white borders from rotation
        if crop:
            cropped_image = smart_crop(rotated_image, padding)
            if cropped_image is None or cropped_image.size == 0:
                cropped_image = rotated_image.copy()
                did_crop = False
            else:
                did_crop = True
        else:
            cropped_image = rotated_image.copy()
            did_crop = False
        
        warped = cropped_image
        
        # Set num_contours_examined for compatibility
        num_contours_examined = 1

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

def detect_upside_down(gray_image):
    """
    Detect if receipt image is upside down by analyzing text/content density.
    Receipts typically have more content at top (store name, address, items)
    and sparser content at bottom (totals, barcode).
    
    Uses multiple checks for robustness:
    1. Text density comparison (top vs bottom)
    2. Minimum density threshold to avoid false positives on sparse receipts
    3. Ratio threshold to ensure clear difference
    
    Args:
        gray_image: Grayscale image
        
    Returns:
        True if image appears to be upside down, False otherwise
    """
    h, w = gray_image.shape
    
    # Safety check: image must be tall enough to analyze
    if h < 300:
        print(f"DEBUG Upside-down: Image too short ({h}px), skipping detection")
        return False
    
    # Use top and bottom quarters (not thirds) for more focused analysis
    # This avoids the middle section which might be cluttered with items
    quarter_h = h // 4
    top_quarter = gray_image[0:quarter_h, :]
    bottom_quarter = gray_image[h-quarter_h:h, :]
    
    # Apply threshold to detect text/content (dark text on light background)
    # Using Otsu for automatic threshold detection
    _, top_binary = cv2.threshold(top_quarter, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, bottom_binary = cv2.threshold(bottom_quarter, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Count text pixels (dark pixels after inversion)
    top_content = np.sum(top_binary > 0)
    bottom_content = np.sum(bottom_binary > 0)
    
    # Calculate density (percentage of pixels with content)
    quarter_area = quarter_h * w
    top_density = top_content / quarter_area
    bottom_density = bottom_content / quarter_area
    
    print(f"DEBUG Upside-down: Top density: {top_density:.4f}, Bottom density: {bottom_density:.4f}")
    
    # Safety check: both sections must have minimum content to make a decision
    # This prevents false positives on very sparse or blank receipts
    min_density = 0.02  # At least 2% content required
    if top_density < min_density and bottom_density < min_density:
        print(f"DEBUG Upside-down: Both sections too sparse, skipping detection")
        return False
    
    # If one section is nearly empty but the other has content, that's a strong signal
    if top_density < min_density and bottom_density > min_density * 2:
        print(f"DEBUG Upside-down: Top nearly empty, bottom has content - upside down!")
        return True
    
    if bottom_density < min_density and top_density > min_density * 2:
        print(f"DEBUG Upside-down: Bottom nearly empty, top has content - correct orientation")
        return False
    
    # If bottom has significantly more content than top, likely upside down
    # Use 1.3x threshold (30% more) for good balance between detection and false positives
    if bottom_density > top_density * 1.3:
        ratio = bottom_density / top_density if top_density > 0 else 0
        print(f"DEBUG Upside-down: Bottom has {ratio:.2f}x more content - upside down!")
        return True
    
    print(f"DEBUG Upside-down: Normal orientation detected")
    return False

def detect_and_correct_rotation(image, rotation_threshold=2.0):
    """
    Enhanced rotation detection specifically for receipts.
    First checks for 90-degree rotation (portrait vs landscape),
    then checks if upside down (180° rotation needed),
    then detects and corrects small angle skew to make receipt upright.
    
    Args:
        image: Input image (should be cropped receipt)
        rotation_threshold: Minimum angle (in degrees) to trigger rotation. Default: 2.0
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # STEP 1: Check if receipt needs 90-degree rotation (sideways)
    # Receipts are typically portrait (taller than wide) or close to square
    aspect_ratio = w / h
    
    print(f"DEBUG Rotation: Image dims: {w}x{h}, aspect ratio: {aspect_ratio:.2f}")
    
    # If image is significantly wider than tall (landscape), it's probably sideways
    if aspect_ratio > 1.5:
        print(f"DEBUG Rotation: Image is landscape (ratio {aspect_ratio:.2f}), rotating 90° CCW")
        # Rotate 90 degrees counter-clockwise to make it portrait
        rotated_90 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Now continue with fine angle detection on the rotated image
        image = rotated_90
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        base_rotation = 90
    elif aspect_ratio < 0.67:  # Very tall and narrow - might be upside down landscape
        print(f"DEBUG Rotation: Image is very tall (ratio {aspect_ratio:.2f}), rotating 90° CW")
        rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = rotated_90
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        base_rotation = -90
    else:
        base_rotation = 0
    
    # STEP 2: Check if image is upside down (180° rotation needed)
    # Strategy: Receipts usually have more text/content at the top (store name, address)
    # and less at the bottom (totals, footer). Check text density in top vs bottom thirds.
    is_upside_down = detect_upside_down(gray)
    if is_upside_down:
        print(f"DEBUG Rotation: Detected upside down, rotating 180°")
        image = cv2.rotate(image, cv2.ROTATE_180)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        base_rotation += 180
    
    # STEP 3: Now detect small angle corrections
    
    # Enhanced edge detection for text lines
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Use Canny for strong edges (works well for printed text)
    edged = cv2.Canny(blur, 30, 100)
    
    # Dilate edges horizontally to connect text in lines
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    dilated = cv2.dilate(edged, kernel_horizontal, iterations=1)
    
    # Use HoughLinesP (probabilistic) for better line detection
    lines = cv2.HoughLinesP(dilated, 1, np.pi / 180, threshold=50, 
                            minLineLength=w * 0.3, maxLineGap=20)
    
    if lines is None or len(lines) == 0:
        # Fallback: try with regular Hough transform
        lines = cv2.HoughLines(edged, 1, np.pi / 180, 80)
        if lines is None or len(lines) == 0:
            return image, 0
        
        # Process regular Hough lines
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta)
            # Convert to range [-90, 90]
            if angle > 90:
                angle = angle - 180
            # Focus on near-horizontal lines (text lines in receipts)
            if abs(angle) < 45:
                angles.append(angle)
    else:
        # Process probabilistic Hough lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle of line
            if x2 - x1 != 0:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Focus on near-horizontal lines
                if abs(angle) < 45:
                    angles.append(angle)
    
    if not angles or len(angles) < 3:
        print(f"DEBUG Rotation: Not enough line angles detected, returning with base rotation: {base_rotation}°")
        return image, base_rotation
    
    # Get median angle (more robust than mean)
    median_angle = np.median(angles)
    
    print(f"DEBUG Rotation: Detected skew angle: {median_angle:.2f}°")
    
    # Round to nearest degree for small adjustments
    if abs(median_angle) < 0.5:
        median_angle = 0
    
    # Only rotate if angle is significant
    if abs(median_angle) < rotation_threshold:
        print(f"DEBUG Rotation: Skew angle too small, total rotation: {base_rotation}°")
        return image, base_rotation
    
    # Rotate image to correct the angle
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    
    # Calculate new dimensions to prevent cropping
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for the new dimensions
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Apply rotation with white background
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    total_rotation = base_rotation + median_angle
    print(f"DEBUG Rotation: Total rotation applied: {total_rotation:.2f}° (base: {base_rotation}°, skew: {median_angle:.2f}°)")
    
    return rotated_image, total_rotation


def smart_crop(image, padding=10):
    """
    Smart crop by finding where dark background ends and receipt begins.
    Works for multiple scenarios:
    - Black/dark backgrounds (before rotation)
    - White borders (after rotation)
    - Colored backgrounds
    - Low contrast receipts
    
    Args:
        image: Input image
        padding: Padding around the detected content in pixels. Default: 10
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Calculate average brightness to determine background type
    avg_brightness = np.mean(gray)
    print(f"DEBUG Crop: Original: {w}x{h}, Avg brightness: {avg_brightness:.1f}")
    
    # Adaptive thresholding based on image characteristics
    if avg_brightness > 200:
        # Very bright image (white borders) - look for any content
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        threshold_ratio = 0.10  # Just need 10% non-white pixels
        print(f"DEBUG Crop: Bright image detected, using white-border mode")
    else:
        # Dark/medium brightness - use fixed high threshold for bright receipt
        # Threshold of 180 works well for most receipts (white/cream paper)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        threshold_ratio = 0.30  # Need 30% bright pixels
        print(f"DEBUG Crop: Using bright-receipt detection mode (threshold: 180)")
    
    # Count bright/content pixels in each row and column
    row_sums = np.sum(binary, axis=1) // 255
    col_sums = np.sum(binary, axis=0) // 255
    
    # Calculate thresholds based on image dimensions
    row_threshold = w * threshold_ratio
    col_threshold = h * threshold_ratio
    
    content_rows = np.where(row_sums > row_threshold)[0]
    content_cols = np.where(col_sums > col_threshold)[0]
    
    print(f"DEBUG Crop: Rows with content: {len(content_rows)}/{h}, Cols with content: {len(content_cols)}/{w}")
    
    if len(content_rows) == 0 or len(content_cols) == 0:
        print(f"DEBUG Crop: No substantial content found, returning original")
        return image
    
    # Get bounding box
    y1 = content_rows[0]
    y2 = content_rows[-1]
    x1 = content_cols[0]
    x2 = content_cols[-1]
    
    print(f"DEBUG Crop: Content bounds: x={x1}-{x2}, y={y1}-{y2}")
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w - 1, x2 + padding)
    y2 = min(h - 1, y2 + padding)
    
    # Validate crop dimensions
    crop_w = x2 - x1
    crop_h = y2 - y1
    
    # Minimum size check - receipt must be at least 50x50 pixels
    if crop_w < 50 or crop_h < 50:
        print(f"DEBUG Crop: Crop too small ({crop_w}x{crop_h}), returning original")
        return image
    
    # Maximum size check - if crop is >90% of original, probably didn't find anything useful
    crop_area = crop_w * crop_h
    original_area = w * h
    crop_ratio = crop_area / original_area
    
    print(f"DEBUG Crop: Crop dimensions: {crop_w}x{crop_h}, ratio: {crop_ratio:.2%}")
    
    if crop_ratio > 0.90:
        print(f"DEBUG Crop: Crop too large ({crop_ratio:.2%}), minimal cropping benefit - returning original")
        return image
    
    # Aspect ratio validation - receipts shouldn't be extremely thin or wide
    crop_aspect = crop_w / crop_h if crop_h > 0 else 1
    if crop_aspect > 5 or crop_aspect < 0.1:
        print(f"DEBUG Crop: Unusual aspect ratio ({crop_aspect:.2f}), may not be a receipt - returning original")
        return image
    
    # Crop the image
    cropped = image[y1:y2+1, x1:x2+1]
    
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
