"""Data loading and preprocessing for graph datasets."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch_geometric.datasets import Planetoid


@dataclass
class GraphData:
    """A simple data class to hold graph data for GNNs."""

    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_nodes: int
    num_features: int
    num_classes: int

    def to(self, device: torch.device) -> "GraphData":
        """Transfers all tensors in the GraphData to the specified device."""
        return GraphData(
            x=self.x.to(device),
            y=self.y.to(device),
            edge_index=self.edge_index.to(device),
            train_mask=self.train_mask.to(device),
            val_mask=self.val_mask.to(device),
            test_mask=self.test_mask.to(device),
            num_nodes=self.num_nodes,
            num_features=self.num_features,
            num_classes=self.num_classes,
        )


# Cora dataset:
def load_cora_data(root: str = "data", self_loops: bool = True) -> GraphData:
    """Loads the Cora dataset and returns it as a GraphData object.

    Args:
        root: The root directory where the dataset will be saved.
        self_loops: Whether to add self-loops to the edge index.

    Returns:
        GraphData: A GraphData object containing the Cora dataset.
    """
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]

    x = data.x.float()
    y = data.y.long()
    edge_index = data.edge_index.long()
    if self_loops:
        edge_index = add_self_loops(edge_index, num_nodes=x.shape[0])
    train_mask = data.train_mask.bool()
    val_mask = data.val_mask.bool()
    test_mask = data.test_mask.bool()

    graph_data = GraphData(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=x.shape[0],
        num_features=x.shape[1],
        num_classes=dataset.num_classes,
    )

    return graph_data


def load_pubmed_data(root: str = "data", self_loops: bool = True) -> GraphData:
    """Loads the PubMed dataset and returns it as a GraphData object.

    Args:
        root: The root directory where the dataset will be saved.
        self_loops: Whether to add self-loops to the edge index.

    Returns:
        GraphData: A GraphData object containing the PubMed dataset.
    """
    dataset = Planetoid(root=root, name="PubMed")
    data = dataset[0]

    x = data.x.float()
    y = data.y.long()
    edge_index = data.edge_index.long()
    if self_loops:
        edge_index = add_self_loops(edge_index, num_nodes=x.shape[0])
    train_mask = data.train_mask.bool()
    val_mask = data.val_mask.bool()
    test_mask = data.test_mask.bool()

    graph_data = GraphData(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=x.shape[0],
        num_features=x.shape[1],
        num_classes=dataset.num_classes,
    )

    return graph_data


# Utility functions
def mask_to_index(mask: torch.Tensor) -> torch.Tensor:
    """Converts a boolean mask to indices.

    Args:
        mask: A boolean tensor where True indicates the selected indices.

    Returns:
        A tensor containing the indices where the mask is True.
    """
    return torch.where(mask)[0]


def index_to_mask(index: torch.Tensor, size: int) -> torch.Tensor:
    """Converts indices to a boolean mask.

    Args:
        index: A tensor containing the indices to be masked.
        size: The total size of the mask.

    Returns:
        A boolean tensor where True indicates the selected indices.
    """
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


class NodeIndexDataset(torch.utils.data.Dataset):  # type: ignore
    """A simple dataset class for node indices."""

    def __init__(self, indices: torch.Tensor) -> None:
        """Initializes the dataset with the given indices."""
        self.indices = indices

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns the sample at the given index."""
        return self.indices[idx]


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Adds self-loops to the edge index.

    Args:
        edge_index: A tensor of shape [2, num_edges] containing the edge indices.
        num_nodes: The total number of nodes in the graph.

    Returns:
        A tensor of shape [2, num_edges + num_nodes] containing the edge indices with self-loops added.
    """
    device = edge_index.device
    self_loops = torch.arange(num_nodes, device=device)
    self_loops = self_loops.unsqueeze(0).repeat(2, 1)  # [2, num_nodes]

    edge_index_with_loops = torch.cat([edge_index, self_loops], dim=1)

    edge_index_with_loops = torch.unique(edge_index_with_loops, dim=1)

    return edge_index_with_loops


def build_split_loaders(
    graph_data: GraphData,
    batch_size: int = 128,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Builds DataLoaders for the training, validation, and test splits of the graph data.

    Args:
        graph_data: The GraphData object containing the graph data and masks.
        batch_size: The batch size for the DataLoaders.
        num_workers: The number of worker processes to use for data loading.

    Returns:
        A tuple containing the DataLoaders for the training, validation, and test splits.
    """
    train_idx = mask_to_index(graph_data.train_mask)
    val_idx = mask_to_index(graph_data.val_mask)
    test_idx = mask_to_index(graph_data.test_mask)

    train_loader = DataLoader(
        NodeIndexDataset(train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        NodeIndexDataset(val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        NodeIndexDataset(test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def compute_degrees(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Computes the degree of each node in the graph.

    Args:
        edge_index: A tensor of shape [2, num_edges] containing the edge indices.
        num_nodes: The total number of nodes in the graph.

    Returns:
        A tensor of shape [num_nodes] containing the degree of each node.
    """
    src = edge_index[0]
    deg = torch.bincount(src, minlength=num_nodes)
    return deg


# Sanity check
def describe_graph(graph_data: GraphData) -> None:
    """Sanity check function to print out the properties of the graph data."""
    print(f"num_nodes     : {graph_data.num_nodes}")
    print(f"num_features  : {graph_data.num_features}")
    print(f"num_classes   : {graph_data.num_classes}")
    print(f"x shape       : {tuple(graph_data.x.shape)}")
    print(f"y shape       : {tuple(graph_data.y.shape)}")
    print(f"edge_index    : {tuple(graph_data.edge_index.shape)}")
    print(f"train nodes   : {int(graph_data.train_mask.sum())}")
    print(f"val nodes     : {int(graph_data.val_mask.sum())}")
    print(f"test nodes    : {int(graph_data.test_mask.sum())}")


def plot_class_distribution(y: torch.Tensor, save_path: str | None = None) -> None:
    """Plots the class distribution of the node labels.

    Args:
        y: A tensor of shape [num_nodes] containing the class labels for each node.
        save_path: If provided, saves the plot to the given path. Otherwise, displays the plot.
    """
    counts = torch.bincount(y).cpu()

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(counts)), counts.numpy())
    plt.xlabel("Class")
    plt.ylabel("Number of nodes")
    plt.title("Class distribution")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()

    plt.close()


def plot_degree_distribution(edge_index: torch.Tensor, num_nodes: int, save_path: str | None = None) -> None:
    """Plots the degree distribution of the graph.

    Args:
        edge_index: A tensor of shape [2, num_edges] containing the edge indices.
        num_nodes: The total number of nodes in the graph.
        save_path: If provided, saves the plot to the given path. Otherwise, displays the plot.
    """
    deg = compute_degrees(edge_index, num_nodes).cpu()

    plt.figure(figsize=(6, 4))
    plt.hist(deg.numpy(), bins=30)
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.title("Degree distribution")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    graph_data = load_cora_data(root="data", self_loops=True)
    describe_graph(graph_data)

    train_loader, val_loader, test_loader = build_split_loaders(graph_data, batch_size=64)

    first_batch = next(iter(train_loader))
    print("First train batch shape:", first_batch.shape)
    print("First train batch:", first_batch[:10])

    plot_class_distribution(graph_data.y, "outputs/cora_class_distribution.png")
    plot_degree_distribution(graph_data.edge_index, graph_data.num_nodes, "outputs/cora_degree_distribution.png")
