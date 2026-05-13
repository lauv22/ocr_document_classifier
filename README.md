OCR Document Classifier
Automatically extracts text from document images and identifies the document type.

What It Does
Upload an image of a document and get the document type back.

Supports: Passport, Citizenship, PAN Card, National ID

Requirements
Before running this project, install the following:

1. Tesseract OCR (required) Download and install from: https://github.com/UB-Mannheim/tesseract/wiki Keep the default install path: C:\Program Files\Tesseract-OCR

2. Python 3.x (required) Download from: https://www.python.org/downloads/

Setup
Step 1 — Clone the project

git clone <your-repo-url>
cd ocr_document_classifier
Step 2 — Install Python libraries

pip install pytesseract Pillow opencv-python
That's it! You're ready to run.

Run
python main.py sample_images/your_image.jpg
Example:

python main.py sample_images/passport.jpg
Tips for Best Results
Use clear, flat, well-lit images
Higher resolution = better accuracy
Supported formats: jpg, jpeg, png, jfif, bmp, tiff, webp
Project Structure
ocr_document_classifier/
├── sample_images/      → place your document images here
├── src/
│   ├── preprocessor.py → cleans the image
│   ├── ocr_engine.py   → reads text from image
│   ├── classifier.py   → identifies document type
│   └── pipeline.py     → connects all steps
└── main.py             → run this file
Built With
Tesseract OCR — text extraction
OpenCV — image preprocessing
pytesseract + Pillow — Python OCR interface
Built for AI/ML Internship — OCR + Classification Pipeline
