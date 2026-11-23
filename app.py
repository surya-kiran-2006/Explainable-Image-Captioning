from __future__ import annotations

import csv
import gc
import math
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

# Set these before importing torch. Unsupported MPS operators can fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import gradio as gr
import numpy as np
import pytesseract
import torch
from matplotlib import colormaps
from PIL import Image, ImageEnhance, ImageOps
from transformers import (
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    BlipImageProcessor,
    BlipProcessor,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VQA_MODEL_ID = "Salesforce/blip-vqa-base"
CAPTION_MAX_NEW_TOKENS = 30
OCCLUSION_GRID_SIZE = 8
OCCLUSION_IMAGE_SIZE = 384


def choose_device() -> torch.device:
    requested = os.environ.get("BLIP_DEVICE", "auto").strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError(
                "BLIP_DEVICE=mps was requested, but PyTorch MPS is unavailable. "
                "Run python check_mac.py and verify your macOS/PyTorch installation."
            )
        return torch.device("mps")
    if requested not in ("auto", ""):
        raise ValueError("BLIP_DEVICE must be 'auto', 'mps', or 'cpu'.")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = choose_device()
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"Selected device: {DEVICE}")
if DEVICE.type == "mps":
    print("Apple Metal MPS acceleration is enabled.")
    try:
        torch.set_num_threads(4)
    except Exception:
        pass
else:
    print("MPS is not active; inference will run on CPU.")


def locate_checkpoint() -> Path:
    configured = os.environ.get("BLIP_MODEL_DIR", "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            MODELS_DIR / "blip-flickr8k-final",
            MODELS_DIR / "blip-flickr8k-checkpoint" / "blip-flickr8k-final",
            BASE_DIR / "blip-flickr8k-final",
        ]
    )

    for candidate in candidates:
        if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
            return candidate.resolve()

    # Search only project model folders, not the entire home directory.
    matches = [
        p.parent
        for p in MODELS_DIR.rglob("model.safetensors")
        if (p.parent / "config.json").is_file()
    ] if MODELS_DIR.exists() else []
    unique = list(dict.fromkeys(p.resolve() for p in matches))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise RuntimeError(
            "More than one BLIP checkpoint was found under the models folder. "
            "Set BLIP_MODEL_DIR to the intended blip-flickr8k-final folder."
        )

    raise FileNotFoundError(
        "Fine-tuned BLIP checkpoint not found. Place the folder containing "
        "config.json and model.safetensors at: "
        f"{MODELS_DIR / 'blip-flickr8k-final'}\n"
        "Or set BLIP_MODEL_DIR to its full path. See README.md, section 3."
    )


MODEL_DIR = locate_checkpoint()
print("Fine-tuned checkpoint:", MODEL_DIR)


def _parameter_device_counts(model: torch.nn.Module) -> dict[str, int]:
    return dict(Counter(str(p.device) for p in model.parameters()))


def _load_caption_model():
    print("Loading the tokenizer saved with the fine-tuned checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        use_fast=True,
    )

    if (MODEL_DIR / "preprocessor_config.json").is_file():
        print("Loading BLIP image processor from the checkpoint...")
        image_processor = BlipImageProcessor.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
        )
    else:
        print(
            "Checkpoint has no preprocessor_config.json; loading the standard "
            "BLIP image preprocessing configuration. The fine-tuned tokenizer "
            "will still be used from the local checkpoint."
        )
        image_processor = BlipImageProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

    processor = BlipProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
    )

    print("Loading fine-tuned Flickr8k BLIP checkpoint on CPU...")
    model, loading_info = BlipForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )

    # BLIP shares the decoder projection matrix with the input word embeddings.
    # SafeTensors may store the shared tensor only once, so the loader can report
    # the output alias as missing/newly initialized. In Transformers 4.49,
    # BlipForConditionalGeneration itself does NOT implement
    # get_output_embeddings(); the nested text_decoder does.
    decoder = model.text_decoder
    decoder.tie_weights()

    input_layer = decoder.get_input_embeddings()
    output_layer = decoder.get_output_embeddings()
    if output_layer is None:
        raise RuntimeError("BLIP text decoder did not expose its output projection.")
    if input_layer.weight.shape != output_layer.weight.shape:
        raise RuntimeError(
            "BLIP input/output embedding dimensions do not match: "
            f"input={tuple(input_layer.weight.shape)}, "
            f"output={tuple(output_layer.weight.shape)}."
        )

    # Explicitly restore the shared Parameter, independent of the outer
    # BlipForConditionalGeneration wrapper's embedding helper methods.
    output_layer.weight = input_layer.weight
    if decoder.get_output_embeddings().weight is not decoder.get_input_embeddings().weight:
        raise RuntimeError("Could not restore BLIP's tied decoder/input embedding weights.")

    # BLIP's stock configuration uses text vocab_size=30524 and bos_token_id=30522.
    # A BERT tokenizer length of 30522 is expected; do NOT force equality or resize.
    model_vocab_size = int(model.config.text_config.vocab_size)
    tokenizer_vocab_size = len(tokenizer)
    bos_id = int(model.config.text_config.bos_token_id)
    print("Model text vocabulary size:", model_vocab_size)
    print("Checkpoint tokenizer length:", tokenizer_vocab_size)
    print("BLIP decoder start token ID:", bos_id)

    if model_vocab_size < tokenizer_vocab_size:
        raise RuntimeError(
            "Tokenizer has more token IDs than the model embedding table can "
            f"represent: tokenizer={tokenizer_vocab_size}, model={model_vocab_size}."
        )
    if not (0 <= bos_id < model_vocab_size):
        raise RuntimeError(
            f"BLIP decoder start token ID {bos_id} is outside model vocabulary "
            f"size {model_vocab_size}."
        )

    missing = loading_info.get("missing_keys", [])
    unexpected = loading_info.get("unexpected_keys", [])
    non_tied_missing = [
        key for key in missing
        if key not in {
            "text_decoder.cls.predictions.decoder.weight",
            "text_decoder.cls.predictions.decoder.bias",
        }
    ]
    if non_tied_missing:
        raise RuntimeError(
            "Checkpoint is missing unexpected model weights: "
            + ", ".join(non_tied_missing[:20])
        )
    if unexpected:
        print("Unexpected checkpoint keys:", unexpected[:20])
    if missing:
        print(
            "Missing tied decoder alias keys (expected for shared weights):",
            missing,
        )

    model.eval()

    print(f"Moving complete caption model to {DEVICE}...")
    try:
        model.to(DEVICE)
    except Exception as exc:
        if DEVICE.type != "mps":
            raise
        print("MPS placement failed; falling back to CPU. Reason:", repr(exc))
        model = model.to("cpu")
        actual_device = torch.device("cpu")
    else:
        actual_device = DEVICE

    counts = _parameter_device_counts(model)
    expected = str(actual_device)
    print("Caption parameter device counts:", counts)
    if set(counts) != {expected}:
        print("Model placement was mixed; moving caption model back to CPU.")
        model.to("cpu")
        actual_device = torch.device("cpu")

    # Verify shared decoder weights again after device placement. Reassign the
    # same Parameter in case device conversion recreated a parameter alias.
    decoder = model.text_decoder
    input_layer = decoder.get_input_embeddings()
    output_layer = decoder.get_output_embeddings()
    if input_layer.weight.shape != output_layer.weight.shape:
        raise RuntimeError("BLIP decoder/input embedding shapes changed during device placement.")
    output_layer.weight = input_layer.weight
    if decoder.get_output_embeddings().weight is not decoder.get_input_embeddings().weight:
        raise RuntimeError(
            "BLIP decoder/input embedding weights are not shared after device "
            "placement. Refusing inference to avoid an uninitialized decoder."
        )

    model.eval()
    print("Caption model ready on", actual_device)
    print("Tied decoder weights verified.")
    return processor, model, actual_device


caption_processor, caption_model, MODEL_DEVICE = _load_caption_model()


def _pil_rgb(image) -> Image.Image:
    if image is None:
        raise ValueError("Please upload an image first.")
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    return image.convert("RGB")


def _move_inputs(batch, device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def generate_caption(image) -> str:
    image = _pil_rgb(image)
    inputs = caption_processor(images=image, return_tensors="pt")
    inputs = _move_inputs(inputs, MODEL_DEVICE)
    with torch.inference_mode():
        # Match the original working inference call in the training notebook:
        # use the checkpoint's saved generation configuration and greedy
        # decoding (the library default), rather than overriding beam search.
        output_ids = caption_model.generate(
            **inputs,
            max_new_tokens=CAPTION_MAX_NEW_TOKENS,
        )
    caption = caption_processor.decode(output_ids[0], skip_special_tokens=True).strip()
    if not caption:
        raise RuntimeError("The model returned an empty caption.")
    return caption


def caption_loss(image, caption: str) -> float:
    image = _pil_rgb(image)
    encoded = caption_processor(
        images=image,
        text=str(caption),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=CAPTION_MAX_NEW_TOKENS,
    )
    encoded = _move_inputs(encoded, MODEL_DEVICE)
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100

    with torch.inference_mode():
        result = caption_model(
            pixel_values=encoded["pixel_values"],
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=labels,
            return_dict=True,
        )
    loss_value = float(result.loss.detach().float().cpu().item())
    if not math.isfinite(loss_value):
        raise RuntimeError("Caption loss was not finite; explainability cannot continue.")
    return loss_value


def generate_heatmap(image, caption: str, grid_size: int = OCCLUSION_GRID_SIZE):
    original = _pil_rgb(image).resize(
        (OCCLUSION_IMAGE_SIZE, OCCLUSION_IMAGE_SIZE), Image.Resampling.BICUBIC
    )
    original_array = np.asarray(original, dtype=np.uint8)
    height, width, _ = original_array.shape
    baseline_loss = caption_loss(original, caption)
    scores = np.zeros((grid_size, grid_size), dtype=np.float32)
    rows = []

    for row in range(grid_size):
        y0 = int(row * height / grid_size)
        y1 = int((row + 1) * height / grid_size)
        for col in range(grid_size):
            x0 = int(col * width / grid_size)
            x1 = int((col + 1) * width / grid_size)
            modified = original_array.copy()
            patch = modified[y0:y1, x0:x1]
            fill_rgb = np.mean(patch, axis=(0, 1), keepdims=True).astype(np.uint8)
            modified[y0:y1, x0:x1] = fill_rgb

            masked_loss = caption_loss(Image.fromarray(modified), caption)
            importance = max(0.0, masked_loss - baseline_loss)
            scores[row, col] = importance
            rows.append({
                "row": row,
                "column": col,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "baseline_loss": baseline_loss,
                "occluded_loss": masked_loss,
                "importance": importance,
            })

    max_score = float(scores.max())
    normalized = scores / max_score if max_score > 0 else scores
    normalized_image = Image.fromarray(
        np.uint8(np.clip(normalized * 255.0, 0, 255))
    ).resize((width, height), Image.Resampling.BILINEAR)
    color_array = np.uint8(
        colormaps["jet"](np.asarray(normalized_image) / 255.0)[..., :3] * 255
    )
    heatmap_image = Image.fromarray(color_array, mode="RGB")
    overlay = Image.blend(original, heatmap_image, alpha=0.42)

    csv_path = RESULTS_DIR / "latest_attributions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    heatmap_image.save(RESULTS_DIR / "latest_attribution_heatmap.png")
    overlay.save(RESULTS_DIR / "latest_explainability_overlay.png")
    best = max(rows, key=lambda item: item["importance"])
    details = (
        f"Method: caption-focused occlusion ({grid_size} × {grid_size} = {grid_size**2} regions)\n"
        f"Baseline caption loss: {baseline_loss:.4f}\n"
        f"Highest-scoring region: row {best['row'] + 1}, column {best['column'] + 1}\n"
        f"Maximum loss increase: {best['importance']:.4f}\n"
        f"Saved attribution CSV: {csv_path}"
    )
    return heatmap_image, overlay, details


def caption_function(image, explain):
    if image is None:
        return "Please upload an image.", None, None, ""
    caption = generate_caption(image)
    if not explain:
        return caption, None, None, "Enable explainability to run the 64-region analysis."
    heatmap, overlay, details = generate_heatmap(image, caption)
    return caption, heatmap, overlay, details


# VQA is loaded only when the user first opens the VQA workflow.
vqa_processor: Optional[BlipProcessor] = None
vqa_model: Optional[BlipForQuestionAnswering] = None
VQA_DEVICE = DEVICE


def load_vqa_model():
    global vqa_processor, vqa_model, VQA_DEVICE
    if vqa_model is not None:
        return

    print("Loading BLIP VQA model (first use only)...")
    processor = BlipProcessor.from_pretrained(VQA_MODEL_ID, use_fast=False)
    model = BlipForQuestionAnswering.from_pretrained(
        VQA_MODEL_ID,
        torch_dtype=torch.float32,
    )
    model.eval()

    try:
        model.to(DEVICE)
        vqa_device = DEVICE
        counts = _parameter_device_counts(model)
        if set(counts) != {str(DEVICE)}:
            raise RuntimeError(f"Mixed VQA parameter devices: {counts}")
    except Exception as exc:
        if DEVICE.type != "mps":
            raise
        print("VQA model could not be placed on MPS; using CPU instead:", repr(exc))
        del model
        gc.collect()
        if hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        model = BlipForQuestionAnswering.from_pretrained(
            VQA_MODEL_ID,
            torch_dtype=torch.float32,
        ).eval()
        model.to("cpu")
        vqa_device = torch.device("cpu")

    vqa_model = model.eval()
    vqa_processor = processor
    VQA_DEVICE = vqa_device
    print("VQA model ready on", VQA_DEVICE)


def vqa_function(image, question):
    if image is None:
        return "Please upload an image."
    question = str(question or "").strip()
    if not question:
        return "Please enter a question about the image."

    load_vqa_model()
    image = _pil_rgb(image)
    inputs = vqa_processor(images=image, text=question, return_tensors="pt")
    inputs = _move_inputs(inputs, VQA_DEVICE)
    with torch.inference_mode():
        output_ids = vqa_model.generate(**inputs, max_new_tokens=20)
    return vqa_processor.decode(output_ids[0], skip_special_tokens=True).strip()


TESSERACT_READY = shutil.which("tesseract") is not None
if TESSERACT_READY:
    try:
        print("Tesseract version:", pytesseract.get_tesseract_version())
    except Exception as exc:
        TESSERACT_READY = False
        print("Tesseract executable found but not usable:", repr(exc))
else:
    print("Tesseract not found. OCR will be unavailable until it is installed with Homebrew.")


def ocr_function(image, x0, y0, x1, y1, psm, upscale, threshold):
    if image is None:
        return None, "Please upload an image.", "No image supplied."
    if not TESSERACT_READY:
        return None, "Tesseract OCR is not installed.", "Install it with: brew install tesseract"

    source = _pil_rgb(image)
    left_pct, top_pct, right_pct, bottom_pct = map(float, (x0, y0, x1, y1))
    if not (0 <= left_pct < right_pct <= 100 and 0 <= top_pct < bottom_pct <= 100):
        return None, "Invalid crop coordinates.", "Set left < right and top < bottom, between 0 and 100."

    width, height = source.size
    box = (
        int(width * left_pct / 100),
        int(height * top_pct / 100),
        max(1, int(width * right_pct / 100)),
        max(1, int(height * bottom_pct / 100)),
    )
    crop = source.crop(box)
    if crop.width < 2 or crop.height < 2:
        return None, "Selected region is too small.", "Expand the crop region and try again."

    processed = ImageOps.grayscale(crop)
    processed = ImageOps.autocontrast(processed)
    processed = ImageEnhance.Sharpness(processed).enhance(1.5)
    if upscale:
        processed = processed.resize(
            (processed.width * 3, processed.height * 3),
            Image.Resampling.LANCZOS,
        )
    if threshold:
        processed = processed.point(lambda value: 255 if value > 160 else 0)

    psm_value = int(psm)
    config = f"--oem 3 --psm {psm_value}"
    extracted = pytesseract.image_to_string(processed, config=config).strip()
    data = pytesseract.image_to_data(
        processed, config=config, output_type=pytesseract.Output.DICT
    )
    confidences = []
    for value in data.get("conf", []):
        try:
            confidence = float(value)
            if confidence >= 0:
                confidences.append(confidence)
        except (TypeError, ValueError):
            pass
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0

    processed.save(RESULTS_DIR / "latest_ocr_crop.png")
    (RESULTS_DIR / "latest_ocr_text.txt").write_text(extracted, encoding="utf-8")
    diagnostic = (
        f"Crop size: {crop.width} × {crop.height}\n"
        f"Processed size: {processed.width} × {processed.height}\n"
        f"Tesseract PSM: {psm_value}\n"
        f"Mean word confidence: {mean_confidence:.1f}/100\n"
        f"Saved output folder: {RESULTS_DIR}"
    )
    return processed, extracted or "No text detected.", diagnostic


with gr.Blocks(title="Visual AI Caption Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Visual AI Caption Assistant
    ### Explainable Image Captioning Using BLIP

    Generate image captions, inspect caption-focused occlusion heatmaps,
    ask questions about images, and extract text using OCR.
    ---
    """)

    with gr.Tabs():
        with gr.Tab("Image Captioning & Explainability"):
            gr.Markdown("### Generate captions and explain model predictions")
            with gr.Row():
                with gr.Column(scale=1):
                    cap_image = gr.Image(type="pil", label="Upload an Image")
                    explain = gr.Checkbox(
                        value=False,
                        label="Generate Explainability Heatmap",
                        info="Runs caption-focused occlusion over 64 image regions; this can take several minutes on a laptop.",
                    )
                    caption_btn = gr.Button("Generate Caption", variant="primary")
                with gr.Column(scale=1):
                    caption_result = gr.Textbox(label="Generated Caption", lines=3, interactive=False)
                    explanation_details = gr.Textbox(label="Explanation Details", lines=5, interactive=False)
            with gr.Row():
                heatmap_result = gr.Image(label="Attribution Heatmap")
                overlay_result = gr.Image(label="Explainability Overlay")
            caption_btn.click(
                fn=caption_function,
                inputs=[cap_image, explain],
                outputs=[caption_result, heatmap_result, overlay_result, explanation_details],
            )

        with gr.Tab("Visual Question Answering"):
            gr.Markdown("""
            ### Ask questions about your uploaded image
            Examples: What objects are present? What is the person doing? What colour is the car?
            The VQA model downloads the first time this tab is used.
            """)
            with gr.Row():
                with gr.Column():
                    vqa_image = gr.Image(type="pil", label="Upload Image")
                    question = gr.Textbox(label="Ask a Question", placeholder="What is happening in this image?")
                    vqa_btn = gr.Button("Get Answer", variant="primary")
                with gr.Column():
                    answer = gr.Textbox(label="BLIP VQA Answer", lines=5, interactive=False)
            vqa_btn.click(fn=vqa_function, inputs=[vqa_image, question], outputs=answer)

        with gr.Tab("OCR Text Extraction"):
            gr.Markdown("""
            ### Extract text from images
            Upload an image, select the region containing text, and extract it using Tesseract OCR.
            Coordinates are percentages of the original image dimensions.
            """)
            with gr.Row():
                with gr.Column():
                    ocr_image = gr.Image(type="pil", label="Upload Image")
                    gr.Markdown("#### Select text region")
                    with gr.Row():
                        x0 = gr.Slider(0, 100, value=0, step=1, label="Left (%)")
                        y0 = gr.Slider(0, 100, value=0, step=1, label="Top (%)")
                    with gr.Row():
                        x1 = gr.Slider(0, 100, value=100, step=1, label="Right (%)")
                        y1 = gr.Slider(0, 100, value=100, step=1, label="Bottom (%)")
                    psm = gr.Dropdown(
                        choices=[("Sparse Text", 11), ("Block of Text", 6), ("Uniform Text Block", 3), ("Sparse Text with OSD", 12)],
                        value=11,
                        label="OCR Mode",
                    )
                    upscale = gr.Checkbox(value=True, label="Upscale image crop 3×")
                    threshold = gr.Checkbox(value=False, label="Apply image thresholding")
                    ocr_btn = gr.Button("Extract Text", variant="primary")
                with gr.Column():
                    processed_image = gr.Image(label="Preprocessed Image")
                    extracted_text = gr.Textbox(label="Extracted Text", lines=12, interactive=False)
                    ocr_diagnostic = gr.Textbox(label="OCR Diagnostic", lines=5, interactive=False)
            ocr_btn.click(
                fn=ocr_function,
                inputs=[ocr_image, x0, y0, x1, y1, psm, upscale, threshold],
                outputs=[processed_image, extracted_text, ocr_diagnostic],
            )

        with gr.Tab("About Project"):
            gr.Markdown("""
            # Explainable Image Captioning
            ### Project Overview
            This project uses a fine-tuned BLIP vision-language model trained on Flickr8k.
            It generates natural-language image descriptions and visualizes regions whose
            occlusion increases the loss of the generated caption.

            ### Technologies Used
            | Component | Technology |
            |---|---|
            | Vision-Language Model | Fine-tuned BLIP |
            | Dataset | Flickr8k |
            | Deep Learning | PyTorch |
            | Explainability | Caption-focused Occlusion Attribution |
            | Visual Question Answering | BLIP VQA |
            | OCR | Tesseract |
            | Interface | Gradio |

            ### Previously recorded model evaluation
            | Metric | Score |
            |---|---:|
            | BLEU-1 | 0.7052 |
            | BLEU-2 | 0.5479 |
            | BLEU-3 | 0.4157 |
            | BLEU-4 | 0.3088 |
            | METEOR | 0.2778 |
            | CIDEr | 0.9318 |

            These are the previously saved results from the 1,000-image Flickr8k test evaluation;
            this dashboard does not recompute them.

            ### Explainable AI
            Each of 64 image regions is replaced with its local mean colour. A larger increase
            in the fixed caption's loss receives a higher attribution score. This is a
            perturbation-based explanation, not proof of causality.
            """)

    gr.Markdown("---\n**Explainable Image Captioning | Fine-tuned BLIP | Flickr8k**")


demo.queue(default_concurrency_limit=1)
print("Gradio dashboard is ready.")
print("Open the local URL shown below (normally http://127.0.0.1:7860).")
demo.launch(
    share=False,
    debug=False,
    show_error=True,
    server_name="127.0.0.1",
    max_file_size="12mb",
)
