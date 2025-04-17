import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch

class FusionGNN(nn.Module):
    def __init__(
        self, 
        node_in_channels=42,  # 32 (op type) + 2 (flops, tensor_bytes) + 7 (dtype) + 1 (shape hash)
        edge_in_channels=6,   # 1 (legal) + 1 (stride) + 3 (runtime) + 1 (chiplet)
        hidden_channels=128, 
        num_layers=2
    ):
        super(FusionGNN, self).__init__()
        
        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # Node encoder (first GraphSAGE layer)
        self.node_encoder = SAGEConv(
            in_channels=node_in_channels,
            out_channels=hidden_channels
        )
        
        # Additional GraphSAGE layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                SAGEConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels
                )
            )
        
        # Edge predictor MLP
        edge_mlp_in = hidden_channels * 2 + edge_in_channels  # Source + target node features + edge features
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_mlp_in, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
    
    def encode_nodes(self, x, edge_index, batch=None):
        """Encode node features using GraphSAGE layers."""
        # Initial node encoding
        h = self.node_encoder(x, edge_index).relu()
        
        # Additional layers
        for conv in self.convs:
            h = conv(h, edge_index).relu()
        
        return h
    
    def forward(self, data):
        """Forward pass of the GNN model."""
        # Unpack the data
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Encode nodes
        node_embeddings = self.encode_nodes(x, edge_index, data.batch if hasattr(data, 'batch') else None)
        
        # Get source and target node embeddings for each edge
        src_idx, dst_idx = edge_index
        src_embeddings = node_embeddings[src_idx]
        dst_embeddings = node_embeddings[dst_idx]
        
        # Concatenate source, target embeddings with edge features
        edge_features = torch.cat([src_embeddings, dst_embeddings, edge_attr], dim=1)
        
        # Predict delta latency for each edge
        delta_latency = self.edge_mlp(edge_features)
        
        return delta_latency
    
    def predict_graph(self, data):
        """Predict delta latency for an entire graph."""
        # Ensure the model is in evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            delta_latency = self.forward(data)
            
            # Return predictions
            return delta_latency

# Huber loss function with delta parameter
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred, target):
        abs_diff = torch.abs(pred - target)
        quadratic = torch.min(abs_diff, torch.tensor(self.delta, device=pred.device))
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic.pow(2) + self.delta * linear
        return loss.mean() 