#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.preprocessing import StandardScaler

def one_hot_encode(value, categories):
    """Create a one-hot encoding for a value given the list of categories."""
    encoding = [0] * len(categories)
    if value in categories:
        encoding[categories.index(value)] = 1
    return encoding

def get_op_type_categories():
    """Define categories for operator types."""
    return [
        "nn.conv2d", "add", "multiply", "subtract", "divide",
        "nn.dense", "nn.bias_add", "nn.batch_norm", "nn.max_pool2d", "nn.avg_pool2d",
        "nn.global_avg_pool2d", "nn.softmax", "nn.relu", "tanh", "sigmoid",
        "exp", "log", "sqrt", "rsqrt", "clip", "cast", "reshape", "transpose",
        "concatenate", "split", "expand_dims", "squeeze", "take", "strided_slice",
        "nn.conv2d_transpose", "nn.upsampling", "zeros", "ones", "nn.dropout"
    ]

def get_dtype_categories():
    """Define categories for data types."""
    return ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"]

def log2_scale(value):
    """Apply log2 scaling to a value, handling zeros and negatives."""
    if value is None or pd.isna(value) or value <= 0:
        return 0.0
    return np.log2(float(value) + 1.0)

def build_node_features(op_type, flops, tensor_bytes, dtype, shape_hash):
    """Build feature vector for a node."""
    op_categories = get_op_type_categories()
    dtype_categories = get_dtype_categories()
    
    # One-hot encode operator type (32 dims)
    op_one_hot = one_hot_encode(op_type, op_categories)
    
    # If operator is not in our categories, use a default category
    if sum(op_one_hot) == 0:
        op_one_hot[-1] = 1  # Use the last category as "other"
    
    # Log2 scale for FLOPs (1 dim)
    log_flops = log2_scale(flops)
    
    # Log2 scale for tensor bytes (1 dim)
    log_tensor_bytes = log2_scale(tensor_bytes)
    
    # One-hot encode dtype (7 dims)
    dtype_one_hot = one_hot_encode(dtype, dtype_categories)
    
    # Shape hash (used as-is, 1 dim)
    # Normalize the hash to [0, 1] by taking modulo 256 and dividing
    norm_shape_hash = (int(shape_hash) % 256) / 255.0 if shape_hash else 0.0
    
    # Combine all features
    features = op_one_hot + [log_flops, log_tensor_bytes] + dtype_one_hot + [norm_shape_hash]
    
    # Ensure feature dimension is fixed (32 + 1 + 1 + 7 + 1 = 42 dimensions)
    assert len(features) == len(op_categories) + 10
    
    return features

def build_edge_features(is_legal, data_stride, hbm_free, sm_occ, chiplet_id):
    """Build feature vector for an edge."""
    # Legal fusion flag (1 dim)
    legal = [1.0] if is_legal else [0.0]
    
    # Data stride bucket (1 dim)
    # 0 = coalesced, 1 = strided
    stride = [0.0] if data_stride == "coalesced" else [1.0]
    
    # Runtime counters (3 dims)
    # Normalize to [0, 1]
    hbm = [float(hbm_free) / 100.0 if hbm_free is not None and not pd.isna(hbm_free) else 0.5]
    sm = [float(sm_occ) / 100.0 if sm_occ is not None and not pd.isna(sm_occ) else 0.5]
    chiplet = [float(chiplet_id) / 8.0 if chiplet_id is not None and not pd.isna(chiplet_id) else 0.0]
    
    # Combine all features (6 dimensions total)
    features = legal + stride + hbm + sm + chiplet
    
    return features

def calculate_tensor_bytes(shape_hash, dtype):
    """Estimate tensor bytes from shape hash and dtype (placeholder)."""
    # In a real implementation, this would decode the shape hash
    # and calculate actual tensor size
    # For this skeleton, we'll just return a random value
    bytes_per_element = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
        "int32": 4,
        "int64": 8,
        "bool": 1
    }.get(dtype, 4)
    
    # Use shape_hash to generate a plausible size
    # In real implementation, you'd decode the actual shape
    shape_factor = (int(shape_hash) % 1000) + 1 if shape_hash else 1000
    return shape_factor * bytes_per_element

def build_pyg_graph(csv_path):
    """Build a PyTorch Geometric graph from edge records CSV."""
    try:
        # Read edge records
        df = pd.read_csv(csv_path)
        
        # Skip if empty
        if len(df) == 0:
            print(f"Empty CSV file: {csv_path}")
            return None
        
        # Create dictionaries to map operator names to node indices
        op_to_idx = {}
        node_features = []
        
        # Create edge index and edge attributes
        edge_index = []
        edge_attr = []
        edge_labels = []
        
        # Process each edge
        for i, row in df.iterrows():
            op_a = row["op_a"]
            op_b = row["op_b"]
            
            # Add nodes if they don't exist
            if op_a not in op_to_idx:
                # Calculate tensor bytes from shape hash and dtype
                tensor_bytes_a = calculate_tensor_bytes(row["shape_hash_a"], row["dtype_a"])
                
                # Add node features
                op_to_idx[op_a] = len(node_features)
                node_features.append(
                    build_node_features(
                        op_a, row["flops_a"], tensor_bytes_a, row["dtype_a"], row["shape_hash_a"]
                    )
                )
            
            if op_b not in op_to_idx:
                # Calculate tensor bytes from shape hash and dtype
                tensor_bytes_b = calculate_tensor_bytes(row["shape_hash_b"], row["dtype_b"])
                
                # Add node features
                op_to_idx[op_b] = len(node_features)
                node_features.append(
                    build_node_features(
                        op_b, row["flops_b"], tensor_bytes_b, row["dtype_b"], row["shape_hash_b"]
                    )
                )
            
            # Add edge
            src_idx = op_to_idx[op_a]
            dst_idx = op_to_idx[op_b]
            edge_index.append([src_idx, dst_idx])
            
            # All edges in our dataset are legal fusion candidates
            is_legal = True
            
            # Determine data stride (simplified placeholder)
            # In a real implementation, you'd analyze actual memory access patterns
            data_stride = "coalesced"
            
            # Add edge features
            edge_attr.append(
                build_edge_features(
                    is_legal, data_stride, 
                    row["hbm_free"], row["sm_occ"], row["chiplet_id"]
                )
            )
            
            # Add edge label (delta latency)
            delta_latency = float(row["delta_latency_ms"])
            edge_labels.append([delta_latency])
        
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        y = torch.tensor(edge_labels, dtype=torch.float)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=len(node_features)
        )
        
        # Add metadata
        data.graph_id = os.path.basename(csv_path).split('.')[0]
        
        return data
    
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Prepare graph data for GNN training")
    parser.add_argument("--input_dir", default="dataset/edge_records", help="Directory containing edge records CSV files")
    parser.add_argument("--output_dir", default="dataset/graphs", help="Output directory for PyG graph files")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of CSV files (excluding raw_edges.csv)
    csv_files = [f for f in glob.glob(os.path.join(args.input_dir, "*.csv")) 
                if os.path.basename(f) != "raw_edges.csv"]
    
    print(f"Found {len(csv_files)} graph files to process")
    
    # Process each file
    for csv_file in tqdm(csv_files, desc="Processing graphs"):
        graph_id = os.path.basename(csv_file).split('.')[0]
        output_path = os.path.join(args.output_dir, f"{graph_id}.pt")
        
        # Skip if already processed
        if os.path.exists(output_path):
            print(f"Graph {graph_id} already processed, skipping")
            continue
        
        # Build PyG graph
        graph = build_pyg_graph(csv_file)
        
        if graph is not None:
            # Save the graph
            torch.save(graph, output_path)
            print(f"Saved graph {graph_id} with {graph.num_nodes} nodes and {graph.num_edges} edges")
    
    print("All graphs processed successfully")

if __name__ == "__main__":
    main() 