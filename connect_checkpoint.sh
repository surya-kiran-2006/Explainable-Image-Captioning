#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$PROJECT_DIR/models"
TARGET="$MODELS_DIR/blip-flickr8k-final"

if [[ $# -ge 1 ]]; then
  SOURCE="$1"
else
  echo "Paste the full path to your EXISTING blip-flickr8k-final folder."
  echo "Tip: In Finder, select the folder and use Option + Command + C to copy its pathname."
  read -r -p "Checkpoint folder path: " SOURCE
fi

# Remove optional surrounding quotes while preserving spaces in the path.
case "$SOURCE" in
  \"*\") SOURCE="${SOURCE:1:${#SOURCE}-2}" ;;
  \'*\') SOURCE="${SOURCE:1:${#SOURCE}-2}" ;;
esac

if [[ "$SOURCE" == "~" ]]; then
  SOURCE="$HOME"
elif [[ "$SOURCE" == "~/"* ]]; then
  SOURCE="$HOME/${SOURCE:2}"
fi

if [[ ! -d "$SOURCE" ]]; then
  echo "ERROR: Folder not found: $SOURCE"
  echo "Run this script again and paste the path to the blip-flickr8k-final folder itself."
  exit 1
fi

SOURCE="$(cd "$SOURCE" && pwd -P)"

if [[ ! -f "$SOURCE/config.json" || ! -f "$SOURCE/model.safetensors" ]]; then
  echo "ERROR: This folder does not look like the expected BLIP checkpoint."
  echo "Required files: config.json and model.safetensors"
  echo "Selected folder: $SOURCE"
  exit 1
fi

mkdir -p "$MODELS_DIR"

if [[ -L "$TARGET" ]]; then
  CURRENT="$(readlink "$TARGET" || true)"
  if [[ "$CURRENT" == "$SOURCE" ]]; then
    echo "Checkpoint is already connected: $TARGET"
    exit 0
  fi
  rm "$TARGET"
elif [[ -d "$TARGET" ]]; then
  if [[ -f "$TARGET/config.json" && -f "$TARGET/model.safetensors" ]]; then
    echo "A checkpoint already exists at: $TARGET"
    echo "Nothing was changed. Remove or rename it only if you intend to replace it."
    exit 1
  fi
  echo "ERROR: $TARGET already exists and is not a valid checkpoint link."
  echo "Please inspect this folder before changing it."
  exit 1
elif [[ -e "$TARGET" ]]; then
  echo "ERROR: A file already exists at: $TARGET"
  exit 1
fi

ln -s "$SOURCE" "$TARGET"

if [[ -f "$TARGET/config.json" && -f "$TARGET/model.safetensors" ]]; then
  echo
  echo "SUCCESS: Existing checkpoint connected without copying it."
  echo "Source: $SOURCE"
  echo "Project link: $TARGET"
  echo "Model size: $(du -h "$TARGET/model.safetensors" | awk '{print $1}')"
else
  echo "ERROR: Checkpoint link verification failed."
  exit 1
fi
