#!/usr/bin/env python3
import os
import json
import random
import argparse
import uuid
from pathlib import Path

import torch
import tvm
from tvm import relay
from tvm.relay import transform

def get_shape_variations(model_type):
    """Generate shape variations for different model types."""
    if "llama" in model_type.lower():
        # Llama models: batch_size, seq_length
        return [
            (1, 512),
            (1, 1024),
            (1, 2048),
            (2, 512),
            (2, 1024),
            (4, 256),
            (4, 512),
        ]
    elif "mixtral" in model_type.lower():
        # Mixtral models: batch_size, seq_length
        return [
            (1, 256),
            (1, 512),
            (1, 1024),
            (2, 256),
            (2, 512),
            (4, 128),
        ]
    elif "resnet" in model_type.lower():
        # ResNet models: batch_size, channels, height, width
        return [
            (1, 3, 224, 224),
            (1, 3, 256, 256),
            (2, 3, 224, 224),
            (4, 3, 224, 224),
            (8, 3, 224, 224),
            (16, 3, 224, 224),
            (32, 3, 224, 224),
        ]
    elif "vit" in model_type.lower():
        # ViT models: batch_size, channels, height, width
        return [
            (1, 3, 224, 224),
            (1, 3, 256, 256),
            (1, 3, 384, 384),
            (2, 3, 224, 224),
            (4, 3, 224, 224),
            (8, 3, 224, 224),
        ]
    else:
        # Default shapes for other models
        return [
            (1, 3, 224, 224),
            (2, 3, 224, 224),
            (4, 3, 224, 224),
            (8, 3, 224, 224),
        ]

def convert_pytorch_to_relay(model_path, model_type, shape, seed):
    """Convert PyTorch model to Relay IR."""
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Create input tensor based on model type and shape
    if "llama" in model_type.lower() or "mixtral" in model_type.lower():
        # For transformer models: (batch_size, seq_length)
        batch_size, seq_length = shape
        input_shape = (batch_size, seq_length)
        input_data = torch.randint(0, 32000, input_shape)
        input_name = "input_ids"
    else:
        # For vision models: (batch_size, channels, height, width)
        batch_size, channels, height, width = shape
        input_shape = (batch_size, channels, height, width)
        input_data = torch.randn(input_shape)
        input_name = "input"
    
    # Convert to Relay
    mod, params = relay.frontend.from_pytorch(model, {input_name: input_shape})
    
    # Apply necessary transformations
    mod = transform.InferType()(mod)
    
    return mod, params

def save_relay_module(mod, params, output_path):
    """Save Relay module to JSON and params to binary file."""
    with open(output_path + ".json", "w") as fo:
        fo.write(tvm.ir.save_json(mod))
    
    with open(output_path + ".params", "wb") as fo:
        fo.write(relay.save_param_dict(params))

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch models to Relay IR with shape jittering")
    parser.add_argument("--models_dir", default="dataset/raw/models", help="Directory containing the downloaded models")
    parser.add_argument("--output_dir", default="dataset/raw", help="Output directory for Relay modules")
    parser.add_argument("--num_seeds", type=int, default=3, help="Number of random seeds to use")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of models
    model_files = list(Path(args.models_dir).glob("*.pt"))
    
    for model_file in model_files:
        model_type = model_file.stem
        print(f"Processing model: {model_type}")
        
        # Get shape variations for this model type
        shape_variations = get_shape_variations(model_type)
        
        # Process each shape variation with different seeds
        for shape in shape_variations:
            shape_str = "_".join(map(str, shape))
            
            for seed in range(args.num_seeds):
                print(f"  Shape: {shape}, Seed: {seed}")
                
                try:
                    # Generate a unique graph ID
                    graph_id = str(uuid.uuid4())
                    
                    # Convert to Relay
                    mod, params = convert_pytorch_to_relay(
                        str(model_file), model_type, shape, seed
                    )
                    
                    # Save the module and params
                    output_base = os.path.join(
                        args.output_dir, 
                        f"{model_type}_{shape_str}_seed{seed}_{graph_id}"
                    )
                    save_relay_module(mod, params, output_base)
                    
                    # Save metadata
                    metadata = {
                        "graph_id": graph_id,
                        "model_type": model_type,
                        "shape": shape,
                        "seed": seed
                    }
                    with open(output_base + ".meta.json", "w") as f:
                        json.dump(metadata, f, indent=2)
                    
                    print(f"  Saved to {output_base}.json")
                
                except Exception as e:
                    print(f"  Error processing {model_type} with shape {shape}, seed {seed}: {e}")
    
    print("Conversion complete!")

if __name__ == "__main__":
    main() 