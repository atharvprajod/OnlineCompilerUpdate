<<<<<<< HEAD
# OnlineCompilerUpdate

Current Meta: Compiler updates based on static graph with some heuristic/cheap measure where given some static graph you determine the optimal run config(in our case we only look at the best computational graph sequence which is NP-hard problem meaning we can't solve search problem using traditional optimization framework; alternative configs would mostly be device allocation which becomes more important for distributed training where theres additional constraint of bandwidth latency; also dif problem is scheduling which from my understanding is just how do i optimize the hyperparams(kind of ?) when given some graph so more so mem allocation stuff). given all this, we instead want to we instead want to break from the static assumption entirely, and propose a system where:
- instead of given static make some optimization
- i wait till runtime so we instead , observe actual input shapes, routing decisions, and device load, then choose fusion/scheduling decisions just-in-time—based on a lightweight, learned decision policy

Expected Plan: 
- Start by training a 1mb GNN-cost model using some kind of Bandit-feedback system(need to decide the loss):  based on all this, we instead 
=======
# Adaptive Fusion GNN Model Pretraining

This repository contains code for Phase 1 of the Adaptive Fusion project, which focuses on offline data generation and GNN pretraining for TVM operator fusion decisions.

## Prerequisites

- TVM >= 0.14.0
- PyTorch >= 2.0
- CUDA-capable GPU for training and profiling

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/adaptive-fusion.git
cd adaptive-fusion
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
adaptive‑fusion/
│
├── third_party/          # pinned TVM, PyTorch, pybind11
├── scripts/              # bash entrypoints
│   ├── fetch_models.sh   # download model checkpoints
│   ├── relay_import.py   # convert models to Relay IR
│   └── profile_edges.py  # profile fusion edges
├── tvm_passes/           # C++: AdaptiveFusionPass.*
├── dataset/              # raw traces ➜ processed tensors
│   ├── raw/              # raw model IR and metadata
│   ├── graphs/           # processed PyG graph data
│   └── edge_records/     # latency measurements
├── cost_model/           # ML model for prediction
│   ├── gnn/              # GNN implementation
│   │   ├── model.py      # model architecture
│   │   ├── train.py      # training script
│   │   └── export.py     # model export utilities
└── tools/                # data preparation and analysis
    └── prepare_graphs.py # convert edge records to graphs
```

## Workflow

### 1. Data Generation

#### 1.1 Download model checkpoints:

```bash
./scripts/fetch_models.sh
```

This will download several model architectures from HuggingFace and timm, including:
- Llama-2-7B, Llama-2-13B
- Mixtral-8x7B
- ResNet-50, ResNet-101
- ViT-Huge/14

#### 1.2 Convert models to Relay IR:

```bash
python scripts/relay_import.py
```

This converts each model to TVM's Relay IR representation with various input shapes.

#### 1.3 Profile fusion edges:

```bash
python scripts/profile_edges.py --target cuda --max_workers 1
```

This profiles each potential fusion edge to measure the latency impact.

### 2. Dataset Preparation

Convert raw edge records to PyTorch Geometric graph format:

```bash
python tools/prepare_graphs.py
```

### 3. GNN Training

Train the GNN model:

```bash
python -m cost_model.gnn.train \
        --data_root=dataset/graphs \
        --batch_size=64 \
        --hidden_channels=128 \
        --learning_rate=3e-4 \
        --max_epochs=30
```

### 4. Model Export

Export the trained model to deployable formats:

```bash
python cost_model.gnn.export \
        --checkpoint=cost_model/checkpoints/best.ckpt \
        --tflite_int8
```

### 5. Benchmark

Evaluate the model's performance:

```bash
python cost_model.gnn.benchmark \
        --model=model_zoo/gnn_fusion_int8.tflite \
        --graphs=dataset/graphs/test
```

## Performance Requirements

The exported model should meet the following requirements:
- Forward pass time on 10k-edge graph: ≤ 1 ms
- Model size: ≤ 300 KB
- Batch inference throughput: ≥ 1M edges/second
- Test RMSE: ≤ 10% of mean delta latency

## License

[Your License] 
>>>>>>> 03c90cd (Initial commit: Adding model conversion and fetching scripts)
