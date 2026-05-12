# OCR + Document Classification Pipeline
## Using Donut (Document Understanding Transformer)

---

## What is Donut?

Donut is a pretrained model by Naver Clova AI.
- No OCR step needed — reads the image directly
- Encoder: Swin Transformer (understands visual layout)
- Decoder: BART (generates the document label as text)
- Pretrained on millions of document images

---

## Setup (do this once)

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. (Recommended) Use a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux

pip install -r requirements.txt
```

---

## Run the pipeline step by step

### Step 1 — Download dataset

```bash
python step1_prepare_dataset.py
```

Downloads RVL-CDIP dataset from HuggingFace.
Saves `label_names.json`.

---

### Step 2 — Fine-tune Donut

```bash
python step2_finetune.py
```

- Downloads pretrained Donut model (~800MB, first run only)
- Fine-tunes it on RVL-CDIP document images
- Saves your fine-tuned model to `./donut-finetuned/`

**⚠ Note:** Fine-tuning on CPU is very slow.
Use Google Colab (free GPU) for faster training.
Change TRAIN_SAMPLES in the file to use more data.

---

### Step 3 — Classify a document

```bash
python step3_predict.py path/to/your/document.jpg
```

Example:
```bash
python step3_predict.py invoice.jpg
```

Output:
```
Document Type : INVOICE
Confidence    : 94.3%
Raw token     : <invoice>
```

---

### Step 4 — Evaluate accuracy

```bash
python step4_evaluate.py
```

Runs the model on 100 test images and prints:
- Overall accuracy %
- Per-class precision, recall, F1
- Most common predictions

---

## Document types the model can classify

| ID | Label |
|----|-------|
| 0  | letter |
| 1  | form |
| 2  | email |
| 3  | handwritten |
| 4  | advertisement |
| 5  | scientific report |
| 6  | scientific publication |
| 7  | specification |
| 8  | file folder |
| 9  | news article |
| 10 | budget |
| 11 | invoice |
| 12 | presentation |
| 13 | questionnaire |
| 14 | resume |
| 15 | memo |

---

## Project structure

```
donut_pipeline/
├── requirements.txt          # Python packages
├── step1_prepare_dataset.py  # Download RVL-CDIP
├── step2_finetune.py         # Fine-tune Donut
├── step3_predict.py          # Classify one image
├── step4_evaluate.py         # Measure accuracy
├── label_names.json          # Auto-generated
└── donut-finetuned/          # Auto-generated (saved model)
```

---

## Pretrained model on HuggingFace

`naver-clova-ix/donut-base`
https://huggingface.co/naver-clova-ix/donut-base
