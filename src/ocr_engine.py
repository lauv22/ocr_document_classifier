import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(preprocessed_image):
    """
    Extracts text using both English and Nepali language models.
    """

    # Try English + Nepali combined first
    try:
        custom_config = r'--oem 3 --psm 3 -l eng+nep'
        text = pytesseract.image_to_string(
            preprocessed_image,
            config=custom_config
        )
        if len(text.strip()) > 20:
            cleaned_text = ' '.join(text.split())
            return cleaned_text
    except:
        pass

    # Fallback 1: English only psm 3
    try:
        custom_config = r'--oem 3 --psm 3'
        text = pytesseract.image_to_string(
            preprocessed_image,
            config=custom_config
        )
        if len(text.strip()) > 20:
            cleaned_text = ' '.join(text.split())
            return cleaned_text
    except:
        pass

    # Fallback 2: English only psm 6
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(
        preprocessed_image,
        config=custom_config
    )
    cleaned_text = ' '.join(text.split())
    return cleaned_text