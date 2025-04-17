#!/usr/bin/env bash
set -euo pipefail

# Absolute paths
ROOT_DIR="$(pwd)"
VENV_DIR="$ROOT_DIR/venv"
MODELS_DIR="$ROOT_DIR/dataset/raw/models"

echo "Working in $ROOT_DIR"
mkdir -p "$MODELS_DIR"

echo "====== Fetch Models Script ======"

# 1) Create & activate virtualenv if needed
if [ ! -d "$VENV_DIR" ]; then
  echo "➤ Creating virtual environment at $VENV_DIR (copies only)…"
  python3 -m venv --copies "$VENV_DIR" \
    || { echo "ERROR: venv creation failed at $VENV_DIR"; exit 1; }
  if [ ! -d "$VENV_DIR/bin" ]; then
    echo "ERROR: missing $VENV_DIR/bin after venv creation"; exit 1
  fi

  echo "➤ Activating venv and installing dependencies…"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip setuptools wheel
  pip install --no-cache-dir -r "$ROOT_DIR/requirements.txt"
else
  echo "➤ Activating existing venv…"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

# 2) Model‑download helper
download_model() {
  local MODEL_TYPE=$1
  local MODEL_NAME=$2
  # safe filename
  local SAFE_NAME="${MODEL_NAME//[^a-zA-Z0-9._-]/_}"
  local OUTPUT_FILE="$MODELS_DIR/$SAFE_NAME.pt"

  if [ -f "$OUTPUT_FILE" ]; then
    echo "✔ $SAFE_NAME already exists, skipping."
    return
  fi

  echo "➤ Downloading $MODEL_NAME as $OUTPUT_FILE…"
  if [ "$MODEL_TYPE" = "hf" ]; then
    python - <<EOF
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("$MODEL_NAME", torch_dtype=torch.float16)
torch.save(model, "$OUTPUT_FILE")
print("  ✔ Saved HF model to $OUTPUT_FILE")
EOF
  elif [ "$MODEL_TYPE" = "timm" ]; then
    python - <<EOF
import torch, timm
model = timm.create_model("$MODEL_NAME", pretrained=True)
torch.save(model, "$OUTPUT_FILE")
print("  ✔ Saved TIMM model to $OUTPUT_FILE")
EOF
  else
    echo "ERROR: unknown model type '$MODEL_TYPE'" >&2
    exit 1
  fi
}

# 3) Download list
download_model "hf"   "meta-llama/Llama-2-7b-hf"
download_model "hf"   "meta-llama/Llama-2-13b-hf"
download_model "hf"   "mistralai/Mixtral-8x7B-v0.1"
download_model "timm" "resnet50"
download_model "timm" "resnet101"
download_model "timm" "vit_huge_patch14_224"

echo "✅ All models downloaded to $MODELS_DIR"
