#!/bin/bash
set -e

# Configuration
TARGET="cuda"
MAX_WORKERS=1
BATCH_SIZE=64
HIDDEN_CHANNELS=128
LEARNING_RATE=3e-4
MAX_EPOCHS=30
EXPORT_DIR="model_zoo"

# Create necessary directories
mkdir -p dataset/raw dataset/graphs dataset/edge_records model_zoo

echo "==========================================="
echo "Phase 1: Adaptive Fusion GNN Pretraining"
echo "==========================================="

# Step 1: Download model checkpoints
echo "Step 1: Downloading model checkpoints..."
./scripts/fetch_models.sh
echo "Step 1: Done"

# Step 2: Convert models to Relay IR
echo "Step 2: Converting models to Relay IR..."
python scripts/relay_import.py --models_dir dataset/raw/models --output_dir dataset/raw
echo "Step 2: Done"

# Step 3: Profile fusion edges
echo "Step 3: Profiling fusion edges..."
python scripts/profile_edges.py --input_dir dataset/raw --output_dir dataset/edge_records --target $TARGET --max_workers $MAX_WORKERS
echo "Step 3: Done"

# Step 4: Prepare graph dataset
echo "Step 4: Preparing graph dataset..."
python tools/prepare_graphs.py --input_dir dataset/edge_records --output_dir dataset/graphs
echo "Step 4: Done"

# Step 5: Train GNN model
echo "Step 5: Training GNN model..."
mkdir -p cost_model/checkpoints
python -m cost_model.gnn.train \
        --data_root dataset/graphs \
        --batch_size $BATCH_SIZE \
        --hidden_channels $HIDDEN_CHANNELS \
        --learning_rate $LEARNING_RATE \
        --max_epochs $MAX_EPOCHS
echo "Step 5: Done"

# Find the best checkpoint
BEST_CHECKPOINT=$(find cost_model/checkpoints -name "*.ckpt" | grep -v "last.ckpt" | head -n 1)
echo "Best checkpoint: $BEST_CHECKPOINT"

# Step 6: Export model
echo "Step 6: Exporting model..."
mkdir -p $EXPORT_DIR
python -m cost_model.gnn.export \
        --checkpoint $BEST_CHECKPOINT \
        --output_dir $EXPORT_DIR \
        --tflite_int8
echo "Step 6: Done"

# Step 7: Benchmark model
echo "Step 7: Benchmarking model..."
python -m cost_model.gnn.benchmark \
        --model $EXPORT_DIR/gnn_fusion_int8.tflite \
        --graphs dataset/graphs/test \
        --output_dir $EXPORT_DIR/benchmark_results
echo "Step 7: Done"

echo "==========================================="
echo "Pipeline completed successfully!"
echo "==========================================="

# Print the path to the exported model
echo "Exported model: $EXPORT_DIR/gnn_fusion_int8.tflite"
echo "Benchmark results: $EXPORT_DIR/benchmark_results" 