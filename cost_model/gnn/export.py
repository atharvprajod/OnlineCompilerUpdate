#!/usr/bin/env python3
import os
import argparse
import json
import torch
import numpy as np
import onnx
from onnx import numpy_helper
import tensorflow as tf
from pytorch_lightning import seed_everything

from model import FusionGNN, HuberLoss
from train import GNNLightningModule

def create_dummy_pyg_data(node_dim=42, edge_dim=6, num_nodes=10, num_edges=20):
    """Create a dummy PyG data object for tracing."""
    # Create random node features
    x = torch.randn(num_nodes, node_dim)
    
    # Create random edge indices (ensure valid indices)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Create random edge attributes
    edge_attr = torch.randn(num_edges, edge_dim)
    
    # Create dummy data object with required fields
    dummy_data = {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'batch': torch.zeros(num_nodes, dtype=torch.long)
    }
    
    return dummy_data

def trace_model_forward(model, dummy_data):
    """Create a traced forward function for the GNN model."""
    class TracedModule(torch.nn.Module):
        def __init__(self, gnn_model):
            super(TracedModule, self).__init__()
            self.gnn_model = gnn_model
            
        def forward(self, x, edge_index, edge_attr):
            # Create a dictionary-like object to mimic PyG data
            class DummyData:
                pass
            
            data = DummyData()
            data.x = x
            data.edge_index = edge_index
            data.edge_attr = edge_attr
            data.batch = None
            
            # Forward pass through the GNN model
            return self.gnn_model(data)
    
    # Create a wrapper model for tracing
    traced_module = TracedModule(model)
    
    # Set model to evaluation mode
    traced_module.eval()
    
    # Create example inputs for tracing
    x = dummy_data['x']
    edge_index = dummy_data['edge_index']
    edge_attr = dummy_data['edge_attr']
    
    # Use torch.jit.trace to create a traced model
    traced_model = torch.jit.trace(
        traced_module,
        (x, edge_index, edge_attr),
        check_inputs=[(x, edge_index, edge_attr)]
    )
    
    return traced_model

def export_to_onnx(model, dummy_data, output_path):
    """Export the model to ONNX format."""
    # Create a traced model
    traced_model = trace_model_forward(model, dummy_data)
    
    # Export to ONNX
    x = dummy_data['x']
    edge_index = dummy_data['edge_index']
    edge_attr = dummy_data['edge_attr']
    
    torch.onnx.export(
        traced_model,
        (x, edge_index, edge_attr),
        output_path,
        input_names=['node_features', 'edge_indices', 'edge_features'],
        output_names=['delta_latency'],
        dynamic_axes={
            'node_features': {0: 'num_nodes'},
            'edge_indices': {1: 'num_edges'},
            'edge_features': {0: 'num_edges'}
        },
        opset_version=12,
        export_params=True,
        verbose=True
    )
    
    print(f"ONNX model exported to {output_path}")
    
    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    return onnx_model

def convert_onnx_to_tflite(onnx_model_path, tflite_output_path, quantize=True):
    """Convert ONNX model to TensorFlow Lite format."""
    try:
        import onnx_tf
        
        # Convert ONNX to TensorFlow
        tf_model_path = tflite_output_path.replace('.tflite', '_tf')
        os.makedirs(tf_model_path, exist_ok=True)
        
        # Convert ONNX to TensorFlow SavedModel
        onnx_tf.backend.prepare(onnx.load(onnx_model_path)).export_graph(tf_model_path)
        
        # Load the SavedModel
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        
        # Set optimization flags
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if quantize:
            # Enable full integer quantization
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            # Define representative dataset for calibration
            def representative_dataset():
                for _ in range(100):
                    # Generate random data for calibration
                    num_nodes = np.random.randint(5, 100)
                    num_edges = np.random.randint(10, 200)
                    
                    # Node features
                    node_features = np.random.rand(num_nodes, 42).astype(np.float32)
                    
                    # Edge indices
                    edge_indices = np.random.randint(0, num_nodes, (2, num_edges)).astype(np.int32)
                    
                    # Edge features
                    edge_features = np.random.rand(num_edges, 6).astype(np.float32)
                    
                    yield [node_features, edge_indices, edge_features]
            
            converter.representative_dataset = representative_dataset
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the TFLite model
        with open(tflite_output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model exported to {tflite_output_path}")
        
        # Print model size
        tflite_size = os.path.getsize(tflite_output_path) / 1024.0
        print(f"TFLite model size: {tflite_size:.2f} KB")
        
        return tflite_output_path
    
    except ImportError:
        print("onnx-tf not found. Install with 'pip install onnx-tf'")
        return None

def benchmark_tflite_model(tflite_model_path):
    """Benchmark inference speed of the TFLite model."""
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print model details
    print("Input details:", input_details)
    print("Output details:", output_details)
    
    # Create random test data
    num_nodes = 50
    num_edges = 100
    node_dim = input_details[0]['shape'][1]
    edge_dim = input_details[2]['shape'][1]
    
    # Prepare inputs
    node_features = np.random.rand(num_nodes, node_dim).astype(np.float32)
    edge_indices = np.random.randint(0, num_nodes, (2, num_edges)).astype(np.int32)
    edge_features = np.random.rand(num_edges, edge_dim).astype(np.float32)
    
    # Set inputs
    interpreter.set_tensor(input_details[0]['index'], node_features)
    interpreter.set_tensor(input_details[1]['index'], edge_indices)
    interpreter.set_tensor(input_details[2]['index'], edge_features)
    
    # Benchmark inference time
    import time
    num_runs = 100
    
    # Warmup
    for _ in range(10):
        interpreter.invoke()
    
    # Timed runs
    start_time = time.time()
    for _ in range(num_runs):
        interpreter.invoke()
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / num_runs
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Average inference time: {avg_time_ms:.4f} ms")
    print(f"Throughput: {1000/avg_time_ms:.2f} inferences/s")
    
    return avg_time_ms

def main():
    parser = argparse.ArgumentParser(description="Export GNN model to ONNX and TFLite")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", default="model_zoo", help="Output directory for exported models")
    parser.add_argument("--onnx", action="store_true", help="Export to ONNX format")
    parser.add_argument("--tflite", action="store_true", help="Export to TensorFlow Lite format")
    parser.add_argument("--tflite_int8", action="store_true", help="Quantize TFLite model to int8")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark exported model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    # Extract hyperparameters
    hparams = checkpoint["hyper_parameters"]
    
    # Create model with same architecture
    model = FusionGNN(
        node_in_channels=42,  # Default value, will be overridden if in checkpoint
        edge_in_channels=6,   # Default value, will be overridden if in checkpoint
        hidden_channels=hparams.get("hidden_channels", 128),
        num_layers=hparams.get("num_layers", 2)
    )
    
    # Create Lightning module
    lightning_model = GNNLightningModule(
        model=model,
        learning_rate=hparams.get("learning_rate", 3e-4),
        weight_decay=hparams.get("weight_decay", 1e-5),
        delta=hparams.get("delta", 1.0)
    )
    
    # Load weights
    lightning_model.load_state_dict(checkpoint["state_dict"])
    
    # Set model to evaluation mode
    model = lightning_model.model
    model.eval()
    
    # Create dummy data for tracing
    dummy_data = create_dummy_pyg_data()
    
    # Export to ONNX if requested
    onnx_path = None
    if args.onnx:
        onnx_path = os.path.join(args.output_dir, "gnn_fusion.onnx")
        export_to_onnx(model, dummy_data, onnx_path)
    
    # Export to TFLite if requested
    tflite_path = None
    if args.tflite or args.tflite_int8:
        if not onnx_path:
            onnx_path = os.path.join(args.output_dir, "gnn_fusion.onnx")
            export_to_onnx(model, dummy_data, onnx_path)
        
        if args.tflite_int8:
            tflite_path = os.path.join(args.output_dir, "gnn_fusion_int8.tflite")
            convert_onnx_to_tflite(onnx_path, tflite_path, quantize=True)
        else:
            tflite_path = os.path.join(args.output_dir, "gnn_fusion.tflite")
            convert_onnx_to_tflite(onnx_path, tflite_path, quantize=False)
    
    # Benchmark if requested
    if args.benchmark and tflite_path:
        benchmark_tflite_model(tflite_path)
    
    # Save commit hash
    try:
        import subprocess
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        with open(os.path.join(args.output_dir, "commit_hash.txt"), "w") as f:
            f.write(commit_hash)
    except:
        print("Unable to get git commit hash")
    
    # Save model summary
    model_summary = {
        "checkpoint": args.checkpoint,
        "export_date": str(datetime.now()),
        "node_dim": dummy_data["x"].shape[1],
        "edge_dim": dummy_data["edge_attr"].shape[1],
        "hidden_dim": hparams.get("hidden_channels", 128),
        "num_layers": hparams.get("num_layers", 2)
    }
    
    with open(os.path.join(args.output_dir, "model_summary.json"), "w") as f:
        json.dump(model_summary, f, indent=2)
    
    print("Export complete!")

if __name__ == "__main__":
    from datetime import datetime
    main() 