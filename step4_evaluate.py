# ─────────────────────────────────────────────────────────────
# STEP 4 — Evaluate model accuracy on test set
# Prints accuracy, per-class results, confusion matrix
# ─────────────────────────────────────────────────────────────

import json
import torch
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

MODEL_DIR    = "./donut-finetuned"
TEST_SAMPLES = 100   # increase for more accurate evaluation

with open("label_names.json") as f:
    LABEL_NAMES = json.load(f)

TOKEN_TO_LABEL = {
    f"<{label.replace(' ', '_')}>": label
    for label in LABEL_NAMES
}

# ── Load model ────────────────────────────────────────────────
print("Loading fine-tuned model...")
processor = DonutProcessor.from_pretrained(MODEL_DIR)
model     = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ── Load test data ────────────────────────────────────────────
print(f"Loading {TEST_SAMPLES} test samples...")
dataset = load_dataset("rvl_cdip", split=f"test[:{TEST_SAMPLES}]")

# ── Run predictions ───────────────────────────────────────────
y_true, y_pred = [], []
print("Running predictions...")

for i, sample in enumerate(dataset):
    image = sample["image"].convert("RGB")
    true_label = LABEL_NAMES[sample["label"]]

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model.generate(pixel_values, max_length=32)

    token = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)

    pred_label = "unknown"
    for tok, lbl in TOKEN_TO_LABEL.items():
        if tok in token:
            pred_label = lbl
            break

    y_true.append(true_label)
    y_pred.append(pred_label)

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{TEST_SAMPLES} done...")

# ── Results ───────────────────────────────────────────────────
correct = sum(t == p for t, p in zip(y_true, y_pred))
accuracy = correct / len(y_true) * 100

print("\n" + "═" * 50)
print(f"  OVERALL ACCURACY: {accuracy:.1f}%  ({correct}/{len(y_true)})")
print("═" * 50)

print("\nPer-class results:")
print(classification_report(y_true, y_pred, zero_division=0))

print("\nMost common predictions:")
for label, count in Counter(y_pred).most_common(10):
    print(f"  {label:30s} {count}")
