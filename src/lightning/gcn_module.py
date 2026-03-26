"""LightningModule for the GCN model."""

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from src.models.gcn import GCN


class LitGCN(pl.LightningModule):  # type: ignore
    """PyTorch Lightning wrapper for GCN."""

    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        dropout: float = 0.5,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
    ) -> None:
        """Initializes the Lightning GCN module."""
        super().__init__()
        self.save_hyperparameters()

        self.model = GCN(
            nfeat=nfeat,
            nhid=nhid,
            nclass=nclass,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x, edge_index)

    def _shared_step(self, stage: str) -> dict[str, torch.Tensor]:
        """Shared logic for train, validation, and test."""
        graph = self.trainer.datamodule.graph_data

        x = graph.x.to(self.device)
        y = graph.y.to(self.device)
        edge_index = graph.edge_index.to(self.device)

        if stage == "train":
            mask = graph.train_mask
        elif stage == "val":
            mask = graph.val_mask
        elif stage == "test":
            mask = graph.test_mask
        else:
            raise ValueError(f"Unknown stage: {stage}")

        mask = mask.to(self.device)

        logits = self(x, edge_index)
        loss = F.cross_entropy(logits[mask], y[mask])

        preds = logits[mask].argmax(dim=1)
        acc = (preds == y[mask]).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return {
            "loss": loss,
            "preds": preds.detach(),
            "targets": y[mask].detach(),
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step."""
        out = self._shared_step("train")
        return out["loss"]

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Validation step."""
        self._shared_step("val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Test step."""
        self._shared_step("test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def get_embeddings(self) -> torch.Tensor:
        """Returns node embeddings from the first GCN layer."""
        graph = self.trainer.datamodule.graph_data
        x = graph.x.to(self.device)
        edge_index = graph.edge_index.to(self.device)
        return self.model.get_embeddings(x, edge_index)
