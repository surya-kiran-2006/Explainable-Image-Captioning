# Explainable Image Captioning — Mac (Apple Silicon)

This is the local macOS version of the Flickr8k fine-tuned BLIP project. It keeps the original four-tab Gradio interface:

1. Image Captioning & Explainability
2. Visual Question Answering (BLIP VQA; loaded only when requested)
3. OCR Text Extraction (Tesseract)
4. About Project

**Your fine-tuned checkpoint is not included in this ZIP.** This version is designed to connect to the `blip-flickr8k-final` folder that you have already downloaded from Google Drive. It creates a symbolic link, so it does not make another copy of the model files.

## Mac requirements

- Apple Silicon Mac (M1/M2/M3/M4); intended for an M2 Mac with 16 GB unified memory.
- macOS 14 or later for the pinned PyTorch 2.11 MPS baseline.
- Native Apple Silicon Terminal (`uname -m` should print `arm64`).
- Python 3.11.
- Internet access for first-time download of the BLIP VQA model. The fine-tuned tokenizer is loaded locally from your checkpoint. Your copy has `processor_config.json` but no `preprocessor_config.json`, so the app loads the standard BLIP image-processor configuration from Hugging Face and combines it with your checkpoint tokenizer. The image preprocessing is the standard BLIP preprocessing; token IDs remain checkpoint-local.

PyTorch uses Apple's Metal Performance Shaders (MPS) backend on Apple Silicon. It is not CUDA. The app automatically chooses MPS when available and otherwise falls back to CPU. Unsupported MPS operations are allowed to fall back to CPU where PyTorch supports that behavior.

## 1. Check your Mac

Open Terminal and run:

```bash
uname -m
sw_vers -productVersion
```

For native Apple Silicon, `uname -m` should print `arm64`.

## 2. Extract this project

Extract `Explainable_Image_Captioning_Mac.zip`. Open Terminal and enter the extracted project folder. For example:

```bash
cd ~/Downloads/Explainable_Image_Captioning_Mac
```

Use your actual extracted folder location if it is different.

## 3. Connect your EXISTING checkpoint folder

You already have a folder named `blip-flickr8k-final` containing files such as `config.json`, `generation_config.json`, `model.safetensors`, `processor_config.json`, `tokenizer_config.json`, and `tokenizer.json`.

In Finder, locate the **blip-flickr8k-final folder itself**. Select it and press **Option + Command + C** to copy its pathname. Then, in Terminal (from the project folder), run:

```bash
./connect_checkpoint.sh
```

When prompted, paste the copied path and press Enter. The script checks that the selected folder contains `config.json` and `model.safetensors`, then creates this link:

```text
Explainable_Image_Captioning_Mac/models/blip-flickr8k-final
```

It does not move, rename, delete, or duplicate your original checkpoint. You can also provide the path directly:

```bash
./connect_checkpoint.sh "/full/path/to/blip-flickr8k-final"
```

Use the path to the **inner `blip-flickr8k-final` folder**, not its parent folder and not the individual `model.safetensors` file.

After connecting, you should see:

```text
models/
└── blip-flickr8k-final -> /your/original/location/blip-flickr8k-final
```

## 4. Install prerequisites

Install Python 3.11 and Tesseract OCR with Homebrew:

```bash
brew install python@3.11
brew install tesseract
```

If Homebrew is not installed, install it first from the official Homebrew website.

## 5. Set up the Python environment

From the project folder, run:

```bash
chmod +x setup_mac.sh run_mac.sh connect_checkpoint.sh
./setup_mac.sh
```

This creates an isolated `.venv` and installs the project dependencies. It does not modify your system Python. It installs PyTorch 2.11.0 and the pinned Transformers/Gradio dependencies used by this package.

Verify Apple GPU availability:

```bash
source .venv/bin/activate
python check_mac.py
```

For MPS acceleration, look for:

```text
MPS available: True
SUCCESS: Apple GPU acceleration is available.
```

If MPS reports unavailable, the app can still run on CPU, but inference will be slower. Do not install CUDA on an M2 Mac.

## 6. Start the application

```bash
./run_mac.sh
```

Wait for Gradio to print the local address, normally:

`http://127.0.0.1:7860`

Open it in Safari or Chrome. The app uses `share=False`; it runs on your Mac and does not create a public Gradio share link. Keep Terminal open while using it. Press `Control + C` to stop the app.

## 7. First test sequence

1. Open **Image Captioning & Explainability**.
2. Upload a small JPG/PNG (under 12 MB).
3. Leave explainability unchecked and click **Generate Caption**.
4. After normal captioning works, enable explainability for one image. The 8×8 occlusion method performs 64 additional model-loss evaluations and may take several minutes on a laptop.
5. Open VQA and ask a short question. The first request downloads the public `Salesforce/blip-vqa-base` model (about 1.5 GB); later requests use the local Hugging Face cache.
6. Test OCR with an image containing readable text.

## Memory guidance for 16 GB unified memory

- Close memory-heavy apps before starting the project.
- Caption BLIP loads when the app starts; VQA loads only when first requested.
- Test normal captioning before explainability or VQA.
- Explainability is computationally expensive because it evaluates 64 masked image regions.
- If MPS runs out of memory, stop the app, close other apps, and relaunch. For troubleshooting only, use CPU mode:

```bash
BLIP_DEVICE=cpu ./run_mac.sh
```

## Output files

Generated results are saved under `outputs/`:

- `latest_attributions.csv`
- `latest_attribution_heatmap.png`
- `latest_explainability_overlay.png`
- `latest_ocr_crop.png`
- `latest_ocr_text.txt`

## Troubleshooting

### Checkpoint not found

Run `./connect_checkpoint.sh` again and provide the full path to the inner `blip-flickr8k-final` folder. Confirm that it contains both `config.json` and `model.safetensors`.

### `MPS available: False`

Check that Terminal is native Apple Silicon (`uname -m` → `arm64`), macOS is compatible, and the check is running inside this project's `.venv`.

### Hugging Face download/network error

The fine-tuned caption weights and tokenizer are local. Because your checkpoint does not contain `preprocessor_config.json`, the app downloads the standard BLIP image-processor configuration on first launch, then combines it with the local fine-tuned tokenizer. The separate VQA model downloads only when first requested. Do not substitute a separately downloaded base caption tokenizer.

### OCR says Tesseract is not installed

Run `brew install tesseract`, stop the app with `Control + C`, and relaunch it.

### Port 7860 is already in use

Stop the other Gradio process or change `server_port=7860` at the bottom of `app.py` to `7861`.

## Project results shown in About tab

The BLEU-1/2/3/4, METEOR, and CIDEr scores shown in the About tab are previously recorded results from the 1,000-image Flickr8k test evaluation. They are displayed as reported results and are not recalculated by this local app.


## Caption tokenizer compatibility fix

The original training notebook saved both the model and processor into `blip-flickr8k-final`, and its successful test loaded the processor from that same folder. This Mac app follows that exact behavior. It checks that the checkpoint tokenizer vocabulary size matches the model text vocabulary before inference. Caption generation uses the checkpoint's saved generation configuration, matching the original working inference call.


## Verify a caption independently of Gradio

Stop the Gradio app with `Control + C`, activate `.venv`, and run:

```bash
python diagnose_caption.py "/full/path/to/a/test-image.jpg"
```

This diagnostic reads both the model and processor from the same checkpoint directory, verifies vocabulary-size compatibility, and prints one caption. It helps separate model/tokenizer problems from dashboard upload or callback problems.


## Checkpoint image-processor file naming

Your supplied checkpoint folder contains `processor_config.json` rather than `preprocessor_config.json`. Transformers 4.49 expects `preprocessor_config.json` when `BlipProcessor.from_pretrained(checkpoint_folder)` is called, so that direct load fails. The updated app loads `AutoTokenizer` from your local checkpoint, loads the standard `BlipImageProcessor` from `Salesforce/blip-image-captioning-base`, and constructs a `BlipProcessor` from those two components. This preserves the checkpoint's tokenizer while using BLIP's standard image preprocessing. If the tokenizer/model vocabulary check fails, stop and report the printed sizes instead of changing tokenizer files.
