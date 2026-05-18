import cv2
import numpy as np
from PIL import Image
import os

SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.jfif', '.bmp', '.tiff', '.tif', '.webp']

def preprocess_image(image_path):
    """
    Takes an image path, cleans and sharpens it
    so Tesseract can read text more accurately.
    Handles small text, low resolution, and noisy images.
    """

    # Step 0: Check format
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        print(f"❌ Unsupported format: {ext}")
        return None

    # Step 1: Open with PIL (handles all formats reliably)
    pil_img = Image.open(image_path).convert('RGB')
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Smart upscaling based on image size
    # Small images need more aggressive upscaling
    height, width = gray.shape
    if width < 600:
        scale = 4.0       # very small image — 4x
    elif width < 1000:
        scale = 3.0       # small image — 3x
    elif width < 1800:
        scale = 2.0       # medium image — 2x
    else:
        scale = 1.5       # large image — 1.5x

    gray = cv2.resize(
        gray, None,
        fx=scale, fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    # Step 4: Sharpen to make text edges crisp
    sharpen_kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

    # Step 5: Denoise — less aggressive to preserve small text
    denoised = cv2.fastNlMeansDenoising(sharpened, h=15)

    # Step 6: CLAHE for contrast — helps small text stand out
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(denoised)

    # Step 7: Adaptive threshold — much better than Otsu for small text
    # Otsu uses one global threshold — bad for documents with mixed regions
    # Adaptive uses local thresholds — handles small text much better
    thresh = cv2.adaptiveThreshold(
        contrasted,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10
    )

    # Step 8: Remove small noise dots that appear after thresholding
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Step 9: Convert to PIL for pytesseract
    final_image = Image.fromarray(cleaned)

    return final_image