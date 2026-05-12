# ─────────────────────────────────────────────────────────────
# STEP 3 — Classify a new document image
#
# Run this after step2_finetune.py
# Usage:
#   python step3_predict.py path/to/your/document.jpg
# ─────────────────────────────────────────────────────────────

import sys
import json
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

MODEL_DIR = "./donut-finetuned"

with open("label_names.json") as f:
    LABEL_NAMES = json.load(f)

# Build token → label mapping
TOKEN_TO_LABEL = {
    f"<{label.replace(' ', '_')}>": label
    for label in LABEL_NAMES
}

# ── Load fine-tuned model ─────────────────────────────────────
print("Loading fine-tuned model...")
processor = DonutProcessor.from_pretrained(MODEL_DIR)
model     = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Running on: {device}\n")


def classify_document(image_path: str) -> dict:
    """
    Classify a document image using the fine-tuned Donut model.
    Returns a dict with: predicted_label, token, confidence
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            max_length=32,
            num_beams=4,           # beam search for better accuracy
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Decode output token
    generated_token = processor.tokenizer.decode(
        outputs.sequences[0],
        skip_special_tokens=False
    ).strip()

    # Map token back to label
    predicted_label = "unknown"
    for token, label in TOKEN_TO_LABEL.items():
        if token in generated_token:
            predicted_label = label
            break

    # Rough confidence from beam scores
    if outputs.scores:
        import torch.nn.functional as F
        probs = F.softmax(outputs.scores[0][0], dim=-1)
        confidence = float(probs.max().item())
    else:
        confidence = 0.0

    return {
        "predicted_label": predicted_label,
        "raw_token":       generated_token,
        "confidence":      round(confidence * 100, 1),
    }


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python step3_predict.py <image_path>")
        print("Example: python step3_predict.py invoice.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Classifying: {image_path}")
    print("─" * 40)

    result = classify_document(image_path)

    print(f"Document Type : {result['predicted_label'].upper()}")
    print(f"Confidence    : {result['confidence']}%")
    print(f"Raw token     : {result['raw_token']}")
    print("─" * 40)
