#!/bin/bash
set -e

MODELS_DIR="dataset/raw/models"
mkdir -p $MODELS_DIR

echo "Fetching models for dataset generation..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install torch transformers timm 
else
  source venv/bin/activate
fi

# Function to download and save model
download_model() {
  MODEL_TYPE=$1
  MODEL_NAME=$2
  OUTPUT_FILE="$MODELS_DIR/${MODEL_NAME// /_}.pt"
  
  if [ -f "$OUTPUT_FILE" ]; then
    echo "Model $MODEL_NAME already exists. Skipping."
    return 0
  fi
  
  echo "Downloading $MODEL_NAME..."
  
  if [ "$MODEL_TYPE" == "hf" ]; then
    python -c "
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('$MODEL_NAME', torch_dtype=torch.float16)
torch.save(model, '$OUTPUT_FILE')
print('Saved $MODEL_NAME to $OUTPUT_FILE')
"
  elif [ "$MODEL_TYPE" == "timm" ]; then
    python -c "
import torch
import timm
model = timm.create_model('$MODEL_NAME', pretrained=True)
torch.save(model, '$OUTPUT_FILE')
print('Saved $MODEL_NAME to $OUTPUT_FILE')
"
  fi
}

# Download HuggingFace models
download_model "hf" "meta-llama/Llama-2-7b-hf"
download_model "hf" "meta-llama/Llama-2-13b-hf"
download_model "hf" "mistralai/Mixtral-8x7B-v0.1"

# Download timm models
download_model "timm" "resnet50"
download_model "timm" "resnet101"
download_model "timm" "vit_huge_patch14_224"

echo "All models downloaded successfully!" 