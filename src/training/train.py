"""Training script for the GAT model using PyTorch Lightning."""

from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from src.lightning.datamodule import GraphDataModule
from src.lightning.gat_module import LitGAT


def main(data: str = "Cora") -> None:
    """Main function to train the GAT model on the Cora or PubMed dataset."""
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    if data in {"Cora", "Citeseer", "PubMed"}:
        datamodule = GraphDataModule(name=data, root="data", self_loops=True)
    else:
        raise ValueError(f"Unsupported dataset: {data}. Supported datasets are 'Cora', 'Citeseer', and 'PubMed'.")

    datamodule.setup()

    graph = datamodule.graph_data

    model = LitGAT(
        nfeat=graph.num_features,  # type: ignore
        nhid=8,
        nclass=graph.num_classes,  # type: ignore
        nheads=8,
        dropout=0.6,
        alpha=0.2,
        lr=0.005,
        weight_decay=5e-4,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-gat-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    early_stopping = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=50,
    )

    logger = CSVLogger(save_dir=str(output_dir), name="lightning_logs")

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=1,
        default_root_dir=str(output_dir),
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main(data="Citeseer")
