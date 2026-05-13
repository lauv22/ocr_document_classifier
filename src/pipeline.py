import os
from src.preprocessor import preprocess_image
from src.ocr_engine import extract_text
from src.classifier import classify_document

def run_pipeline(image_path):
    """
    Master function that connects all steps together.
    Give it an image path → it returns the document type and extracted text.
    """

    # Step 0: Check if the image file actually exists
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found → {image_path}")
        return None

    # Check supported formats
    supported = ['.jpg', '.jpeg', '.png', '.jfif', '.bmp', '.tiff', '.webp']
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in supported:
        print(f"❌ Unsupported format: {ext}")
        print(f"   Supported: {', '.join(supported)}")
        return None

    print(f"\n📄 Processing: {image_path}")
    print("-" * 50)

    # Step 1: Preprocess the image
    print("🔧 Step 1: Preprocessing image...")
    preprocessed = preprocess_image(image_path)
    print("   ✅ Image cleaned and sharpened")

    # Step 2: Extract text using OCR
    print("🔍 Step 2: Extracting text with OCR...")
    extracted_text = extract_text(preprocessed)
    print(f"   ✅ Text extracted ({len(extracted_text)} characters)")

    # Step 3: Classify the document
    print("🏷️  Step 3: Classifying document...")
    doc_type, scores = classify_document(extracted_text)
    print(f"   ✅ Classification done")

    # Step 4: Show results
    print("\n" + "=" * 50)
    print("📊 RESULTS")
    print("=" * 50)
    print(f"📁 Document Type : {doc_type}")
    print(f"\n📝 Extracted Text:\n{extracted_text}")
    print(f"\n📈 Keyword Scores:")
    for doc, score in scores.items():
        print(f"   {doc}: {score} keyword(s) matched")
    print("=" * 50)

    return {
        'document_type': doc_type,
        'extracted_text': extracted_text,
        'scores': scores
    }