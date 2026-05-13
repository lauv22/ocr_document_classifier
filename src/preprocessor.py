import cv2
import numpy as np
from PIL import Image

SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.jfif', '.bmp', '.tiff', '.webp']

def preprocess_image(image_path):
    """
    Takes an image path, cleans and sharpens it
    so Tesseract can read text more accurately.
    Supports: jpg, jpeg, png, jfif, bmp, tiff, webp
    """

    # Step 0: Check if file format is supported
    import os
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        print(f"❌ Unsupported format: {ext}")
        print(f"   Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return None

    # Step 1: Use PIL to open image first (handles jfif, webp, etc better than cv2)
    pil_img = Image.open(image_path)

    # Convert to RGB first to handle any mode (RGBA, P, etc)
    pil_img = pil_img.convert('RGB')

    # Convert PIL image to OpenCV format
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Upscale aggressively — card images need to be bigger
    height, width = gray.shape
    scale = 2.5
    gray = cv2.resize(gray, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_CUBIC)

    # Step 4: Sharpen the image so text edges are crisp
    sharpen_kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

    # Step 5: Denoise
    denoised = cv2.fastNlMeansDenoising(sharpened, h=20)

    # Step 6: Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(denoised)

    # Step 7: Threshold — make text pure black, background pure white
    _, thresh = cv2.threshold(contrasted, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 8: Convert back to PIL Image
    final_image = Image.fromarray(thresh)

    return final_image