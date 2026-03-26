"""Graph Convolutional Network (GCN) model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """A simple GCN layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """Initializes the GCNLayer."""
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier initialization of learnable parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GCN layer."""
        num_nodes = x.size(0)
        device = x.device

        adj = torch.zeros((num_nodes, num_nodes), device=device, dtype=x.dtype)
        adj[edge_index[0], edge_index[1]] = 1.0 

        # add self-loops if not already present
        self_loops=torch.arange(num_nodes, device=device)
        adj[self_loops, self_loops] = 1.0  # Ensure self-loops are included 
        
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)  # Compute D^(-1/2)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)  # D^(-1/2) A D^(-1/2)

        support = x @ self.weight
        out = norm_adj @ support

        if self.bias is not None:
            out = out + self.bias

        return out


class GCN(nn.Module):
    """Two-layer Graph Convolutional Network for node classification."""

    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        dropout: float = 0.5,
    ) -> None:
        """Initializes the GCN model."""
        super().__init__()
        self.dropout = dropout

        self.gc1 = GCNLayer(nfeat, nhid)
        self.gc2 = GCNLayer(nhid, nclass)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GCN model."""
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return x

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns node embeddings from the first GCN layer."""
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        return x
