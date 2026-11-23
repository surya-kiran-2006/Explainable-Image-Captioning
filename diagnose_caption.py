from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipImageProcessor,
    BlipProcessor,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get(
    "BLIP_MODEL_DIR", str(BASE_DIR / "models" / "blip-flickr8k-final")
)).expanduser()

if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python diagnose_caption.py /full/path/to/test_image.jpg\n"
        "Stop the Gradio app before running this separate diagnostic."
    )
image_path = Path(sys.argv[1]).expanduser()
if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")
if not (MODEL_DIR / "model.safetensors").is_file():
    raise FileNotFoundError(f"Fine-tuned checkpoint not found: {MODEL_DIR}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Checkpoint:", MODEL_DIR)
print("Device:", device)
print("Loading tokenizer from the fine-tuned checkpoint...")
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR), local_files_only=True, use_fast=True
)
if (MODEL_DIR / "preprocessor_config.json").is_file():
    image_processor = BlipImageProcessor.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
else:
    print(
        "No preprocessor_config.json found; using the standard BLIP image "
        "processor from Hugging Face. Local fine-tuned tokenizer is retained."
    )
    image_processor = BlipImageProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
processor = BlipProcessor(
    image_processor=image_processor, tokenizer=tokenizer
)

print("Loading fine-tuned caption model...")
model = BlipForConditionalGeneration.from_pretrained(
    str(MODEL_DIR), local_files_only=True, use_safetensors=True,
    torch_dtype=torch.float32
).to(device)
model.eval()

model_vocab = int(model.config.text_config.vocab_size)
tokenizer_vocab = len(tokenizer)
print("Model vocabulary size:", model_vocab)
print("Saved tokenizer vocabulary size:", tokenizer_vocab)
if model_vocab != tokenizer_vocab:
    raise RuntimeError(
        "Tokenizer/model vocabulary mismatch. Do not generate captions until "
        "the original training tokenizer is recovered. "
        f"Model={model_vocab}; tokenizer={tokenizer_vocab}."
    )

image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}
with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=30)
caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
print("\nGenerated caption:", caption)
