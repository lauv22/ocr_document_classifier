import sys
from src.pipeline import run_pipeline

def main():
    """
    Entry point of the program.
    Run this file to classify a document image.
    """

    print("=" * 50)
    print("   OCR Document Classifier")
    print("=" * 50)

    # Check if user provided an image path as argument
    # Example: python main.py sample_images/passport.jpg
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    else:
        # If no argument given, ask the user to type the path
        print("\nNo image path provided.")
        image_path = input("Enter image path: ").strip()

    # Run the full pipeline
    result = run_pipeline(image_path)

    if result:
        print(f"\n Done! Document classified as: {result['document_type']}")

if __name__ == "__main__":
    main()