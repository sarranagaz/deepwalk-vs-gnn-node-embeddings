"""Graph Attention Network (GAT) implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

from src.datasets.graph_data import load_data


class GATLayer(nn.Module):  # type: ignore
    """Multi-head Graph Attention Layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        dropout: float = 0.6,
        alpha: float = 0.2,
        concat: bool = True,
    ) -> None:
        """Initializes the GATLayer.

        Args:
            in_features: Number of input features per node.
            out_features: Number of output features per head.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            alpha: Negative slope for LeakyReLU.
            concat: Whether to concatenate the heads' outputs (True) or average them (False).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # Linear projection for all heads at once
        self.W = nn.Parameter(torch.empty(in_features, num_heads * out_features))

        # Separate attention vectors for source and destination nodes
        self.a_src = nn.Parameter(torch.empty(num_heads, out_features))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_features))

        self.leakyrelu = nn.LeakyReLU(alpha)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset learnable parameters."""
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the GAT layer.

        Args:
            x: Node features [N, in_features]
            edge_index: Edge index [2, E], where src -> dst
            return_attention: Whether to return attention weights

        Returns:
            If return_attention=False:
                - concat=True:  [N, num_heads * out_features]
                - concat=False: [N, out_features]
            If return_attention=True:
                tuple(output, alpha)
        """
        num_nodes = x.size(0)
        src, dst = edge_index  # [E], [E]

        # 1) Linear projection
        # [N, in_features] -> [N, H * F_out] -> [N, H, F_out]
        h = torch.matmul(x, self.W).view(num_nodes, self.num_heads, self.out_features)

        # 2) Compute attention logits
        # [N, H]
        att_src = (h * self.a_src).sum(dim=-1)
        att_dst = (h * self.a_dst).sum(dim=-1)

        # [E, H]
        e = self.leakyrelu(att_src[src] + att_dst[dst])

        # 3) Normalize over incoming edges for each destination node
        alpha = softmax(e, dst)  # [E, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 4) Aggregate messages
        # [E, H, F_out]
        messages = h[src] * alpha.unsqueeze(-1)

        # [N, H, F_out]
        out = torch.zeros(
            num_nodes,
            self.num_heads,
            self.out_features,
            device=x.device,
            dtype=x.dtype,
        )
        out.index_add_(0, dst, messages)

        if self.concat:
            out = out.reshape(num_nodes, self.num_heads * self.out_features)
            out = F.elu(out)
        else:
            out = out.mean(dim=1)

        if return_attention:
            return out, alpha
        return out


class GAT(nn.Module):  # type: ignore
    """Two-layer multi-head GAT."""

    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        nheads: int = 8,
        dropout: float = 0.6,
        alpha: float = 0.2,
    ) -> None:
        """Initializes the GAT model.

        Args:
            nfeat: Number of input features per node.
            nhid: Number of hidden units per head in the first layer.
            nclass: Number of output classes.
            nheads: Number of attention heads in the first layer.
            dropout: Dropout probability.
            alpha: Negative slope for LeakyReLU.
        """
        super().__init__()
        self.dropout = dropout

        self.gat1 = GATLayer(
            in_features=nfeat,
            out_features=nhid,
            num_heads=nheads,
            dropout=dropout,
            alpha=alpha,
            concat=True,
        )

        self.gat2 = GATLayer(
            in_features=nhid * nheads,
            out_features=nclass,
            num_heads=1,
            dropout=dropout,
            alpha=alpha,
            concat=False,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GAT model.

        Args:
            x: Node features [N, in_features]
            edge_index: Edge index [2, E], where src -> dst

        Returns:
            Logits [N, nclass]
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Gets the node embeddings from the first GAT layer.

        Args:
            x: Node features [N, in_features]
            edge_index: Edge index [2, E], where src -> dst

        Returns:
            Node embeddings [N, num_heads * nhid]
        """
        x = self.gat1(x, edge_index)
        return x

    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Gets the attention weights from the first GAT layer.

        Args:
            x: Node features [N, in_features]
            edge_index: Edge index [2, E], where src -> dst

        Returns:
            Attention weights [E, num_heads]
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        _, alpha = self.gat1(x, edge_index, return_attention=True)
        return alpha


if __name__ == "__main__":
    data = load_data(name="Cora", root="data", self_loops=True)

    model = GAT(
        nfeat=data.num_features,
        nhid=8,
        nclass=data.num_classes,
        nheads=8,
        dropout=0.6,
        alpha=0.2,
    )

    logits = model(data.x, data.edge_index)
    print("Logits shape:", logits.shape)

    embeddings = model.get_embeddings(data.x, data.edge_index)
    print("Embeddings shape:", embeddings.shape)

    alpha = model.get_attention_weights(data.x, data.edge_index)
    print("Attention shape:", alpha.shape)
