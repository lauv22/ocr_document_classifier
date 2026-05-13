import pytesseract

# Tell pytesseract exactly where tesseract.exe is installed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(preprocessed_image):
    """
    Takes a preprocessed PIL image and extracts
    all text from it using Tesseract OCR.
    """

    # config options explained:
    # --oem 3  → use the best OCR engine available (LSTM neural net)
    # --psm 6  → treat image as a single block of text (good for documents)
    custom_config = r'--oem 3 --psm 6'

    # Extract text from image
    text = pytesseract.image_to_string(preprocessed_image, config=custom_config)

    # Clean up the text (remove extra whitespace and blank lines)
    cleaned_text = ' '.join(text.split())

    return cleaned_text