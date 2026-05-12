# ─────────────────────────────────────────────────────────────
# STEP 2 — Fine-tune Donut on RVL-CDIP
#
# Donut (Document Understanding Transformer) by Naver Clova
# - No OCR needed — reads image directly
# - Encoder: Swin Transformer (vision)
# - Decoder: BART (language model)
# - Pretrained on IIT-CDIP and other document datasets
# ─────────────────────────────────────────────────────────────

import json
import torch
from datasets import load_dataset
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from PIL import Image

# ── Config ──────────────────────────────────────────────────
MODEL_NAME   = "naver-clova-ix/donut-base"
OUTPUT_DIR   = "./donut-finetuned"
MAX_LENGTH   = 32       # max tokens to generate
IMAGE_SIZE   = [1280, 960]
BATCH_SIZE   = 2        # lower if you run out of memory
EPOCHS       = 3        # increase for better accuracy
TRAIN_SAMPLES = 200     # use more for better accuracy (up to 320,000)
VAL_SAMPLES   = 50

with open("label_names.json") as f:
    LABEL_NAMES = json.load(f)

# ── Load Processor & Model ───────────────────────────────────
print("Loading Donut processor and model...")
print("(Downloads ~800MB on first run)")

processor = DonutProcessor.from_pretrained(MODEL_NAME)
model     = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# Resize image processor to our target size
processor.image_processor.size = {"height": IMAGE_SIZE[0], "width": IMAGE_SIZE[1]}

# Add our class labels as special tokens so decoder can output them
new_tokens = [f"<{label.replace(' ', '_')}>" for label in LABEL_NAMES]
processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
model.decoder.resize_token_embeddings(len(processor.tokenizer))

# Set generation config
model.config.pad_token_id       = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s>"])[0]

# ── Load Dataset ─────────────────────────────────────────────
print("\nLoading dataset...")
raw = load_dataset("rvl_cdip")
train_raw = raw["train"].select(range(TRAIN_SAMPLES))
val_raw   = raw["validation"].select(range(VAL_SAMPLES))

# ── Preprocessing ─────────────────────────────────────────────
def preprocess(examples):
    images = [img.convert("RGB") for img in examples["image"]]

    # Build target text: <label_name>
    labels = examples["label"]
    target_texts = []
    for label_id in labels:
        label_str = LABEL_NAMES[label_id].replace(" ", "_")
        target_texts.append(f"<{label_str}>")

    # Process images
    pixel_values = processor(
        images,
        return_tensors="pt",
        data_format="channels_first"
    ).pixel_values

    # Tokenize target texts
    labels_enc = processor.tokenizer(
        target_texts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids

    # Replace padding token id with -100 so loss ignores them
    labels_enc[labels_enc == processor.tokenizer.pad_token_id] = -100

    return {
        "pixel_values": pixel_values,
        "labels":       labels_enc,
    }

print("Preprocessing training data...")
train_dataset = train_raw.map(
    preprocess,
    batched=True,
    batch_size=8,
    remove_columns=train_raw.column_names
)
train_dataset.set_format("torch")

print("Preprocessing validation data...")
val_dataset = val_raw.map(
    preprocess,
    batched=True,
    batch_size=8,
    remove_columns=val_raw.column_names
)
val_dataset.set_format("torch")

# ── Training Arguments ────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir          = OUTPUT_DIR,
    num_train_epochs    = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    warmup_steps        = 50,
    weight_decay        = 0.01,
    logging_dir         = "./logs",
    logging_steps       = 10,
    evaluation_strategy = "epoch",
    save_strategy       = "epoch",
    load_best_model_at_end = True,
    predict_with_generate  = True,
    fp16               = torch.cuda.is_available(),  # faster on GPU
    report_to          = "none",
)

# ── Trainer ───────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model         = model,
    args          = training_args,
    train_dataset = train_dataset,
    eval_dataset  = val_dataset,
    tokenizer     = processor.tokenizer,
)

print("\nStarting fine-tuning...")
print(f"  Model: {MODEL_NAME}")
print(f"  Training samples: {TRAIN_SAMPLES}")
print(f"  Epochs: {EPOCHS}")
print(f"  Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU (slow — use Google Colab for GPU)'}")
print()

trainer.train()

# ── Save ──────────────────────────────────────────────────────
print(f"\nSaving fine-tuned model to {OUTPUT_DIR}/")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("\nDone! Run step3_predict.py to classify a document.")
