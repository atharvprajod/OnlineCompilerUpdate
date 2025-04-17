#!/usr/bin/env python3
import os
import glob
import random
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data, Batch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np

from model import FusionGNN, HuberLoss

class GNNLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, weight_decay=1e-5, delta=1.0):
        super(GNNLightningModule, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = HuberLoss(delta=delta)
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, data):
        return self.model(data)
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        pred = self(batch)
        loss = self.criterion(pred, batch.y)
        
        # Metrics
        mae = torch.abs(pred - batch.y).mean()
        rmse = torch.sqrt(torch.mean((pred - batch.y) ** 2))
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True)
        self.log('train_rmse', rmse, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        pred = self(batch)
        loss = self.criterion(pred, batch.y)
        
        # Metrics
        mae = torch.abs(pred - batch.y).mean()
        rmse = torch.sqrt(torch.mean((pred - batch.y) ** 2))
        
        # Log metrics
        self.log('val_loss', loss)
        self.log('val_mae', mae)
        self.log('val_rmse', rmse)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        # Forward pass
        pred = self(batch)
        loss = self.criterion(pred, batch.y)
        
        # Metrics
        mae = torch.abs(pred - batch.y).mean()
        rmse = torch.sqrt(torch.mean((pred - batch.y) ** 2))
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_mae', mae)
        self.log('test_rmse', rmse)
        
        return {'loss': loss, 'pred': pred, 'target': batch.y}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch'
            }
        }

def load_graph_dataset(data_root):
    """Load all graph data files into a list."""
    graph_files = glob.glob(os.path.join(data_root, "*.pt"))
    graphs = []
    
    for graph_file in graph_files:
        try:
            graph = torch.load(graph_file)
            graph.graph_id = os.path.basename(graph_file).split('.')[0]
            graphs.append(graph)
        except Exception as e:
            print(f"Error loading {graph_file}: {e}")
    
    return graphs

def split_dataset_by_graph_id(graphs, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """Split the dataset by graph ID to avoid data leakage."""
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get unique graph IDs
    graph_ids = set(graph.graph_id for graph in graphs)
    graph_ids = list(graph_ids)
    
    # Shuffle graph IDs
    random.shuffle(graph_ids)
    
    # Calculate split indices
    n_graphs = len(graph_ids)
    n_train = int(n_graphs * train_ratio)
    n_val = int(n_graphs * val_ratio)
    
    # Split graph IDs
    train_ids = graph_ids[:n_train]
    val_ids = graph_ids[n_train:n_train + n_val]
    test_ids = graph_ids[n_train + n_val:]
    
    # Create splits
    train_graphs = [graph for graph in graphs if graph.graph_id in train_ids]
    val_graphs = [graph for graph in graphs if graph.graph_id in val_ids]
    test_graphs = [graph for graph in graphs if graph.graph_id in test_ids]
    
    print(f"Train: {len(train_graphs)} graphs, {sum(g.num_edges for g in train_graphs)} edges")
    print(f"Val: {len(val_graphs)} graphs, {sum(g.num_edges for g in val_graphs)} edges")
    print(f"Test: {len(test_graphs)} graphs, {sum(g.num_edges for g in test_graphs)} edges")
    
    return train_graphs, val_graphs, test_graphs

def main():
    parser = argparse.ArgumentParser(description="Train GNN for fusion prediction")
    parser.add_argument("--data_root", default="dataset/graphs", help="Root directory of graph dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--hidden_channels", type=int, default=128, help="Hidden size for GNN")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--huber_delta", type=float, default=1.0, help="Delta for Huber loss")
    parser.add_argument("--max_epochs", type=int, default=30, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", default="cost_model/checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    args = parser.parse_args()
    
    # Set random seeds
    pl.seed_everything(args.seed)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create output directory name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"gnn_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load dataset
    print(f"Loading graphs from {args.data_root}")
    graphs = load_graph_dataset(args.data_root)
    print(f"Loaded {len(graphs)} graphs with {sum(g.num_edges for g in graphs)} edges total")
    
    # Split dataset
    train_graphs, val_graphs, test_graphs = split_dataset_by_graph_id(
        graphs, 
        train_ratio=0.7, 
        val_ratio=0.2, 
        test_ratio=0.1, 
        seed=args.seed
    )
    
    # Create data loaders
    train_loader = PyGDataLoader(
        train_graphs, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = PyGDataLoader(
        val_graphs, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = PyGDataLoader(
        test_graphs, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Infer input dimensions from first graph
    node_in_channels = train_graphs[0].x.shape[1]
    edge_in_channels = train_graphs[0].edge_attr.shape[1]
    
    # Create model
    model = FusionGNN(
        node_in_channels=node_in_channels,
        edge_in_channels=edge_in_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers
    )
    
    # Create Lightning module
    lightning_model = GNNLightningModule(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        delta=args.huber_delta
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename="gnn-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=args.patience,
        verbose=True,
        mode="min"
    )
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(run_dir, "logs"),
        name="fusion_gnn"
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
        accelerator="auto"
    )
    
    # Train model
    trainer.fit(lightning_model, train_loader, val_loader)
    
    # Test model
    results = trainer.test(lightning_model, test_loader)
    
    # Save test results
    with open(os.path.join(run_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Save best model path
    with open(os.path.join(run_dir, "best_model_path.txt"), "w") as f:
        f.write(checkpoint_callback.best_model_path)
    
    print(f"Training completed. Best model saved at {checkpoint_callback.best_model_path}")
    print(f"Test results: {results}")

if __name__ == "__main__":
    main() 