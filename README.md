# OCR Document Classifier

Automatically extracts text from document images and identifies the document type.
Built with Python, Tesseract OCR, and a Streamlit web interface.

## What It Does

Upload a document image and the system will:
- Extract all text from the image using OCR
- Classify the document type automatically
- Display extracted fields like Name, Date of Birth, Date of Issue, etc.
- Show a confidence score for the classification

Supports: Passport, Citizenship Certificate, PAN Card, National ID

---

## Requirements

**1. Tesseract OCR** (required)
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
Keep the default install path: C:\Program Files\Tesseract-OCR

**2. Python 3.x** (required)
Download from: https://www.python.org/downloads/

---

## Setup

Step 1 - Clone the project

    git clone https://github.com/lauv22/ocr_document_classifier.git
    cd ocr_document_classifier

Step 2 - Install Python libraries

    pip install pytesseract Pillow opencv-python streamlit

That's it! You're ready to run.

---

## Run

**Option 1 - Web UI (recommended)**

    streamlit run app.py

Opens in your browser at http://localhost:8501

**Option 2 - Command Line**

    python main.py sample_images/your_image.jpg

---

## How to Use the Web UI

1. Open the app in your browser
2. Upload a document image using the file uploader
3. Click "Classify Document"
4. View the results page showing:
   - Document type with confidence score
   - Extracted fields (Name, Date of Birth, Date of Issue, etc.)
   - Keyword match scores
   - Raw OCR text (expandable)
5. Click "Classify Another Document" to go back

---

## Tips for Best Results
- Use clear, flat, well-lit images
- Higher resolution = better accuracy
- Supported formats: jpg, jpeg, png, jfif, bmp, tiff, webp

---

## Project Structure

    ocr_document_classifier/
    ├── sample_images/      -> place your document images here
    ├── src/
    │   ├── preprocessor.py -> cleans and sharpens the image
    │   ├── ocr_engine.py   -> extracts text using Tesseract OCR
    │   ├── classifier.py   -> identifies document type and extracts fields
    │   └── pipeline.py     -> connects all steps together
    ├── app.py              -> Streamlit web interface
    └── main.py             -> command line entry point

---

## Built With
- Tesseract OCR - text extraction engine
- OpenCV - image preprocessing
- pytesseract + Pillow - Python OCR interface
- Streamlit - web interface

---

Built for AI/ML Internship - OCR + Classification Pipeline