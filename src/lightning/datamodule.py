"""Data module for handling graph data in PyTorch Lightning."""

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

from src.GAT.datasets.graph_data import GraphData, load_cora_data, load_pubmed_data


class SingleItemDataset(Dataset):  # type: ignore
    """A simple dataset that returns a single item, used for graph data where the entire graph is one sample."""

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return 1

    def __getitem__(self, idx: int) -> int:
        """Returns the single item in the dataset."""
        return 0


class CoraDataModule(pl.LightningDataModule):  # type: ignore
    """PyTorch Lightning DataModule for the Cora dataset."""

    def __init__(self, root: str = "data", self_loops: bool = True) -> None:
        """Initializes the CoraDataModule."""
        super().__init__()
        self.root = root
        self.self_loops = self_loops
        self.graph_data: GraphData | None = None

    def setup(self, stage: str | None = None) -> None:
        """Sets up the dataset for training, validation, and testing."""
        if self.graph_data is None:
            self.graph_data = load_cora_data(root=self.root, self_loops=self.self_loops)

    def train_dataloader(self) -> DataLoader:
        """Returns the training dataloader."""
        return DataLoader(SingleItemDataset(), batch_size=1, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataloader."""
        return DataLoader(SingleItemDataset(), batch_size=1, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Returns the test dataloader."""
        return DataLoader(SingleItemDataset(), batch_size=1, shuffle=False)


class PubMedDataModule(pl.LightningDataModule):  # type: ignore
    """PyTorch Lightning DataModule for the PubMed dataset."""

    def __init__(self, root: str = "data", self_loops: bool = True) -> None:
        """Initializes the PubMedDataModule."""
        super().__init__()
        self.root = root
        self.self_loops = self_loops
        self.graph_data: GraphData | None = None

    def setup(self, stage: str | None = None) -> None:
        """Sets up the dataset for training, validation, and testing."""
        if self.graph_data is None:
            self.graph_data = load_pubmed_data(root=self.root, self_loops=self.self_loops)

    def train_dataloader(self) -> DataLoader:
        """Returns the training dataloader."""
        return DataLoader(SingleItemDataset(), batch_size=1, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataloader."""
        return DataLoader(SingleItemDataset(), batch_size=1, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Returns the test dataloader."""
        return DataLoader(SingleItemDataset(), batch_size=1, shuffle=False)
