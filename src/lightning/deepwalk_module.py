""" DeepWalk Lightning Module """

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from src.models.deepwalk import DeepWalk


class LitDeepWalk(pl.LightningModule):  # type: ignore
    """LightningModule wrapping DeepWalk for node classification."""

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        nclass: int,
        embedding_dim: int = 128,
        walk_length: int = 40,
        num_walks: int = 10,
        window_size: int = 5,
        w2v_epochs: int = 5,
        classifier_hidden_dim: int = 64,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        workers: int = 1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = DeepWalk(
            embedding_dim=embedding_dim,
            num_features=num_features,  
            walk_length=walk_length,
            num_walks=num_walks,
            window_size=window_size,
            w2v_epochs=w2v_epochs,
            classifier_hidden_dim=classifier_hidden_dim,
            nclass=nclass,
            workers=workers,
            seed=seed,
        )

        self.embeddings: torch.Tensor | None = None
        self.num_nodes = num_nodes

    def setup(self, stage: str | None = None) -> None:
        """Build DeepWalk embeddings once from the graph structure."""
        if self.embeddings is not None:
            return

        graph = self.trainer.datamodule.graph_data
        edge_index = graph.edge_index.cpu()

        self.embeddings = self.model.fit_embeddings(
            edge_index=edge_index,
            num_nodes=graph.num_nodes,
        )

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """Return logits for given node indices."""
        if self.embeddings is None:
            raise RuntimeError("Embeddings are not initialized. setup() must run first.")

        graph = self.trainer.datamodule.graph_data
        dw = self.embeddings.to(self.device)[node_indices]
        feat = graph.x.to(self.device)[node_indices]
        x = torch.cat([dw, feat], dim=1)

        return self.model(x)

    def _shared_step(self, stage: str) -> dict[str, torch.Tensor]:
        """Shared logic for train/val/test."""
        graph = self.trainer.datamodule.graph_data
        y = graph.y.to(self.device)

        if stage == "train":
            mask = graph.train_mask
        elif stage == "val":
            mask = graph.val_mask
        elif stage == "test":
            mask = graph.test_mask
        else:
            raise ValueError(f"Unknown stage: {stage}")

        node_indices = torch.where(mask)[0].to(self.device)
        logits = self(node_indices)
        targets = y[node_indices]

        loss = F.cross_entropy(logits, targets)
        preds = logits.argmax(dim=1)
        acc = (preds == targets).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return {
            "loss": loss,
            "preds": preds.detach(),
            "targets": targets.detach(),
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
        """Return optimizer."""
        return torch.optim.Adam(
            self.model.classifier.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def get_embeddings(self) -> torch.Tensor:
        """Return DeepWalk embeddings for all nodes."""
        if self.embeddings is None:
            raise RuntimeError("Embeddings are not initialized.")
        return self.embeddings.to(self.device)