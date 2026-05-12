# ─────────────────────────────────────────────────────────────
# STEP 1 — Download dataset and prepare it
# RVL-CDIP has 400,000 document images in 16 classes:
#   letter, form, email, handwritten, advertisement,
#   scientific report, scientific publication, specification,
#   file folder, news article, budget, invoice,
#   presentation, questionnaire, resume, memo
# ─────────────────────────────────────────────────────────────

from datasets import load_dataset

print("Downloading RVL-CDIP dataset from HuggingFace...")
print("This may take a while on first run (large dataset).")
print("For quick testing we load only the test split (40,000 images).")

# Load just the test split to keep things fast for demo
# For full training use: split="train" (320,000 images)
dataset = load_dataset("rvl_cdip", split="test[:500]")  # first 500 images

print(f"\nDataset loaded: {len(dataset)} samples")
print(f"Features: {dataset.features}")
print(f"\nLabel names:")

label_names = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific report", "scientific publication", "specification",
    "file folder", "news article", "budget", "invoice",
    "presentation", "questionnaire", "resume", "memo"
]

for i, name in enumerate(label_names):
    print(f"  {i:2d} → {name}")

# Save label names for use in later scripts
import json
with open("label_names.json", "w") as f:
    json.dump(label_names, f)

print("\nSaved label_names.json")
print("Done! Run step2_finetune.py next.")
