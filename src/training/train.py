"""Training script for GAT, DeepWalk and GCN using PyTorch Lightning."""

from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.lightning.datamodule import GraphDataModule
from src.lightning.deepwalk_module import LitDeepWalk
from src.lightning.gat_module import LitGAT
from src.lightning.gcn_module import LitGCN


def main(data: str = "Cora", model_name: str = "deepwalk") -> None:
    """Train a model on the selected graph dataset."""
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    if data in {"Cora", "Citeseer", "PubMed"}:
        datamodule = GraphDataModule(name=data, root="data", self_loops=True)
    else:
        raise ValueError(f"Unsupported dataset: {data}. Supported datasets are 'Cora', 'Citeseer', and 'PubMed'.")

    datamodule.setup()
    graph = datamodule.graph_data

    if model_name.lower() == "gat":
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
        checkpoint_name = f"best-gat-{data}-" + "{epoch:02d}-{val_acc:.4f}"
        logger_name = "lightning_logs_gat"

    elif model_name.lower() == "deepwalk":
        model = LitDeepWalk(
            num_nodes=graph.num_nodes,  # type: ignore
            nclass=graph.num_classes,  # type: ignore
            num_features=graph.num_features,  # type: ignore
            embedding_dim=128,
            walk_length=40,
            num_walks=20,
            window_size=10,
            w2v_epochs=5,
            classifier_hidden_dim=64,
            lr=0.01,
            weight_decay=5e-4,
            workers=1,
            seed=42,
        )
        checkpoint_name = f"best-deepwalk-{data}-" + "{epoch:02d}-{val_acc:.4f}"
        logger_name = "lightning_logs_deepwalk"
    elif model_name.lower() == "gcn":
        model = LitGCN(
            nfeat=graph.num_features,  # type: ignore
            nhid=16,
            nclass=graph.num_classes,  # type: ignore
            dropout=0.5,
            lr=0.01,
            weight_decay=5e-4,
        )
        checkpoint_name = f"best-gcn-{data}-" + "{epoch:02d}-{val_acc:.4f}"
        logger_name = "lightning_logs_gcn"

    else:
        raise ValueError("model_name must be either 'gat', 'deepwalk', or 'gcn'.")

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename=checkpoint_name,
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    early_stopping = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=50,
    )

    logger = CSVLogger(save_dir=str(output_dir), name=logger_name)

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
    main(data="PubMed", model_name="gcn")
