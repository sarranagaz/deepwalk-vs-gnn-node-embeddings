"""Graph Attention Network (GAT) LightningModule implementation."""

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from src.models.gat import GAT


class LitGAT(pl.LightningModule):  # type: ignore
    """LightningModule for training a Graph Attention Network (GAT) on graph data."""

    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        nheads: int = 8,
        dropout: float = 0.6,
        alpha: float = 0.2,
        lr: float = 0.005,
        weight_decay: float = 5e-4,
    ) -> None:
        """Initializes the LitGAT model."""
        super().__init__()
        self.save_hyperparameters()

        self.model = GAT(
            nfeat=nfeat,
            nhid=nhid,
            nclass=nclass,
            nheads=nheads,
            dropout=dropout,
            alpha=alpha,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GAT model."""
        return self.model(x, edge_index)

    def _shared_step(self, stage: str) -> dict[str, torch.Tensor]:
        """Performs the shared step for training, validation, or testing."""
        graph = self.trainer.datamodule.graph_data
        x = graph.x
        y = graph.y
        edge_index = graph.edge_index

        logits = self(x, edge_index)

        if stage == "train":
            mask = graph.train_mask
        elif stage == "val":
            mask = graph.val_mask
        elif stage == "test":
            mask = graph.test_mask
        else:
            raise ValueError(f"Unknown stage: {stage}")

        loss = F.cross_entropy(logits[mask], y[mask])
        preds = logits.argmax(dim=1)
        acc = (preds[mask] == y[mask]).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return {
            "loss": loss,
            "logits": logits.detach(),
            "preds": preds.detach(),
        }

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for optimizing the model on the training set."""
        out = self._shared_step("train")
        return out["loss"]

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step for evaluating the model on the validation set."""
        self._shared_step("val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step for evaluating the model on the test set."""
        self._shared_step("test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Returns the optimizer for training."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def get_embeddings(self) -> torch.Tensor:
        """Returns the node embeddings from the first GAT layer."""
        graph = self.trainer.datamodule.graph_data
        self.eval()
        with torch.no_grad():
            return self.model.get_embeddings(graph.x, graph.edge_index)

    def get_attention_weights(self) -> torch.Tensor:
        """Returns the attention weights from the first GAT layer."""
        graph = self.trainer.datamodule.graph_data
        self.eval()
        with torch.no_grad():
            return self.model.get_attention_weights(graph.x, graph.edge_index)
