#!/usr/bin/env python3
import os
import argparse
import glob
import time
import json
import numpy as np
import torch
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_graph_dataset(data_root):
    """Load all graph data files into a list."""
    graph_files = glob.glob(os.path.join(data_root, "*.pt"))
    graphs = []
    
    print(f"Loading {len(graph_files)} graph files...")
    for graph_file in tqdm(graph_files):
        try:
            graph = torch.load(graph_file)
            graph.graph_id = os.path.basename(graph_file).split('.')[0]
            graphs.append(graph)
        except Exception as e:
            print(f"Error loading {graph_file}: {e}")
    
    return graphs

def prepare_graph_for_tflite(graph):
    """Extract features from a PyG graph for TFLite inference."""
    # Extract node features
    node_features = graph.x.numpy()
    
    # Extract edge indices
    edge_indices = graph.edge_index.numpy()
    
    # Extract edge features
    edge_features = graph.edge_attr.numpy()
    
    # Extract edge target values (ground truth)
    edge_targets = graph.y.numpy()
    
    return node_features, edge_indices, edge_features, edge_targets

def benchmark_tflite_model(model_path, graphs, output_dir=None):
    """Benchmark TFLite model performance on real graph data."""
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare for results
    predictions = []
    targets = []
    latencies = []
    errors = []
    
    # Process each graph
    for graph in tqdm(graphs, desc="Evaluating graphs"):
        try:
            # Extract features
            node_features, edge_indices, edge_features, edge_target = prepare_graph_for_tflite(graph)
            
            # Track per-graph inference time
            start_time = time.time()
            
            # Set input tensors
            interpreter.resize_tensor_input(input_details[0]['index'], node_features.shape)
            interpreter.resize_tensor_input(input_details[1]['index'], edge_indices.shape)
            interpreter.resize_tensor_input(input_details[2]['index'], edge_features.shape)
            interpreter.allocate_tensors()
            
            interpreter.set_tensor(input_details[0]['index'], node_features.astype(np.float32))
            interpreter.set_tensor(input_details[1]['index'], edge_indices.astype(np.int32))
            interpreter.set_tensor(input_details[2]['index'], edge_features.astype(np.float32))
            
            # Run inference
            interpreter.invoke()
            
            # Get the prediction
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            # Record inference time
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # ms
            
            # Store results
            predictions.append(prediction)
            targets.append(edge_target)
            latencies.append(inference_time)
            
            # Calculate prediction error
            abs_error = np.abs(prediction - edge_target)
            mae = np.mean(abs_error)
            rmse = np.sqrt(np.mean(np.square(abs_error)))
            
            errors.append({
                'graph_id': graph.graph_id,
                'num_nodes': len(node_features),
                'num_edges': len(edge_features),
                'inference_time_ms': inference_time,
                'mae': float(mae),
                'rmse': float(rmse)
            })
            
        except Exception as e:
            print(f"Error processing graph {graph.graph_id}: {e}")
    
    # Calculate overall metrics
    all_preds = np.vstack(predictions)
    all_targets = np.vstack(targets)
    
    # Calculate errors
    abs_error = np.abs(all_preds - all_targets)
    overall_mae = np.mean(abs_error)
    overall_rmse = np.sqrt(np.mean(np.square(abs_error)))
    
    # Calculate average inference time
    avg_inference_time = np.mean(latencies)
    total_edges = sum(len(graph.edge_attr) for graph in graphs)
    throughput = total_edges / (sum(latencies) / 1000)  # edges per second
    
    # Summarize results
    result_summary = {
        'model_path': model_path,
        'num_graphs': len(graphs),
        'total_edges': int(total_edges),
        'overall_mae': float(overall_mae),
        'overall_rmse': float(overall_rmse),
        'avg_inference_time_ms': float(avg_inference_time),
        'throughput_edges_per_sec': float(throughput),
        'graph_details': errors
    }
    
    # Print summary
    print(f"\nModel: {model_path}")
    print(f"Total graphs: {len(graphs)}")
    print(f"Total edges: {total_edges}")
    print(f"Overall MAE: {overall_mae:.6f}")
    print(f"Overall RMSE: {overall_rmse:.6f}")
    print(f"Average inference time: {avg_inference_time:.4f} ms")
    print(f"Throughput: {throughput:.2f} edges/second")
    
    # Save results if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'benchmark_results.json')
        with open(result_file, 'w') as f:
            json.dump(result_summary, f, indent=2)
        
        # Generate error distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(abs_error, bins=50, alpha=0.7)
        plt.axvline(overall_mae, color='r', linestyle='--', label=f'MAE: {overall_mae:.4f}')
        plt.axvline(overall_rmse, color='g', linestyle='--', label=f'RMSE: {overall_rmse:.4f}')
        plt.title('Error Distribution')
        plt.xlabel('Absolute Error (ms)')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
        
        # Generate predicted vs. actual plot (sample 1000 points for clarity)
        idx = np.random.choice(len(all_preds), size=min(1000, len(all_preds)), replace=False)
        plt.figure(figsize=(10, 6))
        plt.scatter(all_targets[idx], all_preds[idx], alpha=0.5)
        
        # Plot ideal prediction line
        min_val = min(np.min(all_targets), np.min(all_preds))
        max_val = max(np.max(all_targets), np.max(all_preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('Predicted vs. Actual Δ-Latency')
        plt.xlabel('Actual Δ-Latency (ms)')
        plt.ylabel('Predicted Δ-Latency (ms)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pred_vs_actual.png'))
        
        print(f"Results saved to {result_file}")
    
    return result_summary

def main():
    parser = argparse.ArgumentParser(description="Benchmark GNN fusion model")
    parser.add_argument("--model", required=True, help="Path to the TFLite model file")
    parser.add_argument("--graphs", required=True, help="Path to directory containing test graphs")
    parser.add_argument("--output_dir", default="benchmark_results", help="Output directory for benchmark results")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of graphs to evaluate")
    args = parser.parse_args()
    
    # Load graph dataset
    graphs = load_graph_dataset(args.graphs)
    
    # Limit number of graphs if specified
    if args.limit and args.limit < len(graphs):
        graphs = graphs[:args.limit]
    
    # Run benchmark
    result_summary = benchmark_tflite_model(args.model, graphs, args.output_dir)
    
    # Create output dir with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"benchmark_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model size info
    model_size_kb = os.path.getsize(args.model) / 1024.0
    print(f"Model size: {model_size_kb:.2f} KB")
    
    # Check if model meets requirements from blueprint
    requirements = {
        'max_inference_time_ms': 1.0,  # ≤ 1 ms
        'max_model_size_kb': 300.0,    # ≤ 300 KB
        'min_throughput': 1e6,         # ≥ 1e6 edges/s
        'max_rmse_pct': 10.0           # ≤ 10% mean Δ-lat
    }
    
    # Calculate mean absolute Δ-latency for percentage error calculation
    mean_abs_delta = np.mean(np.abs([graph.y.numpy() for graph in graphs]))
    rmse_pct = (result_summary['overall_rmse'] / mean_abs_delta) * 100
    
    # Check requirements
    checks = {
        'Inference time': result_summary['avg_inference_time_ms'] <= requirements['max_inference_time_ms'],
        'Model size': model_size_kb <= requirements['max_model_size_kb'],
        'Throughput': result_summary['throughput_edges_per_sec'] >= requirements['min_throughput'],
        'RMSE': rmse_pct <= requirements['max_rmse_pct']
    }
    
    # Print requirements check
    print("\nRequirements Check:")
    for check, passed in checks.items():
        print(f"{check}: {'✓' if passed else '✗'}")
    
    # Save requirements check
    with open(os.path.join(output_dir, 'requirements_check.json'), 'w') as f:
        json.dump({
            'requirements': requirements,
            'actual': {
                'inference_time_ms': result_summary['avg_inference_time_ms'],
                'model_size_kb': model_size_kb,
                'throughput': result_summary['throughput_edges_per_sec'],
                'rmse_pct': rmse_pct
            },
            'checks': checks,
            'all_passed': all(checks.values())
        }, f, indent=2)
    
    print("\nBenchmark complete!")

if __name__ == "__main__":
    main() 