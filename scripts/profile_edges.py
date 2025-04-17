#!/usr/bin/env python3
import os
import argparse
import json
import csv
import time
import concurrent.futures
from pathlib import Path
import numpy as np

import tvm
from tvm import relay, runtime
from tvm.contrib import graph_executor
import tvm.contrib.debugger.debug_executor as debug_executor

def load_relay_module(relay_json_path, params_path=None):
    """Load a Relay module from JSON and params files."""
    with open(relay_json_path, "r") as f:
        relay_json = f.read()
    
    mod = tvm.ir.load_json(relay_json)
    
    if params_path and os.path.exists(params_path):
        with open(params_path, "rb") as f:
            params = relay.load_param_dict(f.read())
    else:
        params = {}
    
    return mod, params

def get_graph_id_from_path(path):
    """Extract graph_id from file path based on metadata."""
    meta_path = str(path).replace(".json", ".meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            return metadata.get("graph_id", os.path.basename(path).split('_')[-1].split('.')[0])
    
    # Fall back to extracting UUID from filename
    base = os.path.basename(path)
    parts = base.split('_')
    if len(parts) > 3:
        return parts[-1].split('.')[0]
    
    return "unknown"

def create_random_inputs(mod):
    """Create random inputs for a Relay module."""
    inputs = {}
    input_shapes = {}
    
    # Get the main function
    main_func = mod["main"]
    
    # Get input shapes from function parameters
    for i, param in enumerate(main_func.params):
        param_name = param.name_hint if param.name_hint else f"input_{i}"
        shape = [int(dim) for dim in param.checked_type().shape]
        dtype = param.checked_type().dtype
        
        # Create random input with appropriate shape and dtype
        if "int" in dtype:
            inputs[param_name] = np.random.randint(0, 100, size=shape).astype(dtype)
        else:
            inputs[param_name] = np.random.uniform(-1, 1, size=shape).astype(dtype)
        
        input_shapes[param_name] = shape
    
    return inputs, input_shapes

def compile_and_profile(mod, params, target, edge_id=None, force_fuse=False, graph_id="unknown"):
    """Compile a Relay module and profile its execution time."""
    # Set up compilation options
    target_str = str(target)
    
    # Create pass context with appropriate options
    pass_ctx_config = {
        "relay.FuseOps.max_depth": 100,
        "relay.FuseOps.fuse_opt_level": 2,
        "graph_id": graph_id
    }
    
    if edge_id is not None and force_fuse:
        pass_ctx_config["force_fuse"] = edge_id
    
    with tvm.transform.PassContext(opt_level=3, config=pass_ctx_config):
        if edge_id is not None and force_fuse:
            # Add the forced fusion pass to the pipeline
            seq = tvm.transform.Sequential([
                relay.transform.InferType(),
                tvm.relay.transform.FuseOps(fuse_opt_level=2),
                relay.transform.InferType()
            ])
            mod = seq(mod)
        
        # Compile the model
        lib = relay.build(mod, target=target, params=params)
    
    # Create graph executor
    ctx = tvm.device(target_str, 0)
    module = graph_executor.GraphModule(lib["default"](ctx))
    
    # Create random inputs
    inputs, _ = create_random_inputs(mod)
    
    # Set inputs
    for name, data in inputs.items():
        module.set_input(name, data)
    
    # Profile execution time
    num_runs = 100  # Number of runs for averaging
    
    # Warmup runs
    for _ in range(10):
        module.run()
    
    # Timed runs
    start_time = time.time()
    for _ in range(num_runs):
        module.run()
    
    # Ensure all kernels have completed
    ctx.sync()
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / num_runs
    
    return avg_time_ms

def process_graph(relay_json_path, output_dir, target_str="cuda"):
    """Process a graph to get edge latency differences."""
    try:
        print(f"Processing {relay_json_path}...")
        
        # Load the Relay module
        params_path = relay_json_path.replace(".json", ".params")
        mod, params = load_relay_module(relay_json_path, params_path)
        
        # Get graph ID
        graph_id = get_graph_id_from_path(relay_json_path)
        
        # Create target
        target = tvm.target.Target(target_str)
        
        # First, run with normal fusion to capture edges
        # This will populate edge records in dataset/edge_records/raw_edges.csv
        with tvm.transform.PassContext(opt_level=3, config={"dump_fusion_edges": True, "graph_id": graph_id}):
            relay.build(mod, target=target, params=params)
        
        # Now read the edge records
        edge_file = os.path.join(output_dir, "raw_edges.csv")
        if not os.path.exists(edge_file):
            print(f"No edge records found for {relay_json_path}")
            return
        
        # Filter edges for current graph
        edges = []
        with open(edge_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["graph_id"] == graph_id:
                    edges.append(row)
        
        if not edges:
            print(f"No edges found for graph {graph_id}")
            return
        
        print(f"Found {len(edges)} edges for graph {graph_id}")
        
        # Create output file for this graph
        graph_output_file = os.path.join(output_dir, f"{graph_id}.csv")
        with open(graph_output_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow([
                "edge_id", "op_a", "op_b", "flops_a", "flops_b", 
                "shape_hash_a", "shape_hash_b", "dtype_a", "dtype_b",
                "hbm_free", "sm_occ", "chiplet_id", "delta_latency_ms"
            ])
        
        # Compile and profile baseline (no forced fusion)
        baseline_time = compile_and_profile(mod, params, target, graph_id=graph_id)
        print(f"Baseline time: {baseline_time:.4f} ms")
        
        # Process each edge
        for edge in edges:
            edge_id = int(edge["edge_id"])
            print(f"  Processing edge {edge_id}: {edge['op_a']} -> {edge['op_b']}")
            
            try:
                # Compile and profile with forced fusion for this edge
                fused_time = compile_and_profile(
                    mod, params, target, 
                    edge_id=edge_id, force_fuse=True, 
                    graph_id=graph_id
                )
                
                # Calculate latency difference
                delta_latency = baseline_time - fused_time
                
                # Update the edge record
                edge["delta_latency_ms"] = f"{delta_latency:.6f}"
                
                # Write to graph output file
                with open(graph_output_file, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        edge["edge_id"], edge["op_a"], edge["op_b"], 
                        edge["flops_a"], edge["flops_b"],
                        edge["shape_hash_a"], edge["shape_hash_b"], 
                        edge["dtype_a"], edge["dtype_b"],
                        edge["hbm_free"], edge["sm_occ"], edge["chiplet_id"],
                        edge["delta_latency_ms"]
                    ])
                
                print(f"    Baseline: {baseline_time:.4f} ms, Fused: {fused_time:.4f} ms, Delta: {delta_latency:.4f} ms")
            
            except Exception as e:
                print(f"    Error processing edge {edge_id}: {e}")
        
        print(f"Completed processing graph {graph_id}")
        return graph_id
    
    except Exception as e:
        print(f"Error processing {relay_json_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Profile fusion edge latencies")
    parser.add_argument("--input_dir", default="dataset/raw", help="Directory containing Relay JSON files")
    parser.add_argument("--output_dir", default="dataset/edge_records", help="Output directory for edge records")
    parser.add_argument("--target", default="cuda", help="Target device (e.g., cuda, llvm)")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all Relay JSON files
    relay_files = list(Path(args.input_dir).glob("*.json"))
    relay_files = [f for f in relay_files if not f.name.endswith(".meta.json")]
    
    print(f"Found {len(relay_files)} Relay files to process")
    
    # Process each graph sequentially or in parallel
    if args.max_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(process_graph, str(relay_file), args.output_dir, args.target): relay_file
                for relay_file in relay_files
            }
            
            for future in concurrent.futures.as_completed(futures):
                relay_file = futures[future]
                try:
                    graph_id = future.result()
                    if graph_id:
                        print(f"Completed graph {graph_id} from {relay_file}")
                except Exception as e:
                    print(f"Error processing {relay_file}: {e}")
    else:
        for relay_file in relay_files:
            process_graph(str(relay_file), args.output_dir, args.target)
    
    print("All graphs processed successfully")

if __name__ == "__main__":
    main() 