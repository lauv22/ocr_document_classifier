import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path):
    """
    Takes an image path, cleans and sharpens it
    so Tesseract can read text more accurately.
    """

    # Step 1: Read the image using OpenCV
    img = cv2.imread(image_path)

    # Step 2: Convert to grayscale (black & white)
    # Tesseract works better on grayscale images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Resize image if it's too small
    # Small images = blurry text = bad OCR
    height, width = gray.shape
    if width < 1000:
        scale = 1000 / width
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

    # Step 4: Remove noise from image
    denoised = cv2.fastNlMeansDenoising(gray, h=30)

    # Step 5: Apply threshold (make text pure black, background pure white)
    _, thresh = cv2.threshold(denoised, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 6: Convert back to PIL Image (pytesseract needs PIL format)
    final_image = Image.fromarray(thresh)

    return final_image