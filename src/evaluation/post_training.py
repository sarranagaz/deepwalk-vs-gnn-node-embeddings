"""Post-training evaluation and visualization for the GAT or DeepWalk model."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from src.lightning.datamodule import GraphDataModule
from src.lightning.gat_module import LitGAT
from src.lightning.deepwalk_module import LitDeepWalk


def load_best_model(checkpoint_path: str, datamodule: GraphDataModule, model_name: str = "gat") -> LitGAT | LitDeepWalk:
    """Loads the best model checkpoint for evaluation."""
    graph = datamodule.graph_data

    if model_name == "gat":
        model = LitGAT.load_from_checkpoint(
            checkpoint_path,
            nfeat=graph.num_features,  # type: ignore
            nhid=8,
            nclass=graph.num_classes,  # type: ignore
            nheads=8,
            dropout=0.6,
            alpha=0.2,
            lr=0.005,
            weight_decay=5e-4,
        )
    elif model_name == "deepwalk":
        model = LitDeepWalk.load_from_checkpoint(
            checkpoint_path,
            num_nodes=graph.num_nodes,  # type: ignore
            nclass=graph.num_classes,  # type: ignore
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
    else:
        raise ValueError("model_name must be either 'gat' or 'deepwalk'.")

    model.eval()
    return model  # type: ignore


def make_post_training_figures(
    checkpoint_path: str,
    output_dir: str = "outputs/figures",
    data: str = "Cora",
    model_name: str = "gat",
) -> None:
    """Generates post-training evaluation figures."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if data in ["Cora", "Citeseer", "PubMed"]:
        datamodule = GraphDataModule(name=data, root="data", self_loops=True)
    else:
        raise ValueError(f"Unsupported dataset: {data}. Supported datasets are 'Cora', 'Citeseer', and 'PubMed'.")

    datamodule.setup()
    graph = datamodule.graph_data

    model = load_best_model(checkpoint_path, datamodule, model_name=model_name)

    class DummyTrainer:
        def __init__(self, datamodule: GraphDataModule) -> None:
            self.datamodule = datamodule

    model.trainer = DummyTrainer(datamodule)  # type: ignore
    if model_name == "deepwalk":
        model.setup()  # type: ignore
    
    with torch.no_grad():
        if model_name == "gat":
            logits = model(graph.x, graph.edge_index)  # type: ignore
        elif model_name == "deepwalk":
            node_indices = torch.arange(graph.num_nodes)
            logits = model(node_indices)  # type: ignore

        preds = logits.argmax(dim=1).cpu().numpy()

    y_true = graph.y[graph.test_mask].cpu().numpy()  # type: ignore
    y_pred = preds[graph.test_mask.cpu().numpy()]  # type: ignore

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    plt.title("Test confusion matrix")
    plt.tight_layout()
    plt.savefig(outdir / "confusion_matrix.png", dpi=200)
    plt.close()

    with torch.no_grad():
        embeddings = model.get_embeddings().cpu().numpy()  # type: ignore

    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    emb_2d = tsne.fit_transform(embeddings)
    labels = graph.y.cpu().numpy()  # type: ignore

    plt.figure(figsize=(7, 6))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, s=10)
    plt.title(f"t-SNE of {model_name} embeddings")
    plt.tight_layout()
    plt.savefig(outdir / "tsne_embeddings.png", dpi=200)
    plt.close()

    if model_name == "gat":
        with torch.no_grad():
            attention_weights = model.get_attention_weights().cpu().numpy()  # type: ignore

        plt.figure(figsize=(10, 6))
        for i in range(10):
            plt.plot(attention_weights[i], label=f"Node {i}")
        plt.xlabel("Neighbor index")
        plt.ylabel("Attention weight")
        plt.title("Attention weights for the first 10 nodes")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "attention_weights.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    checkpoint_path = "outputs/checkpoints/best-deepwalk-Citeseer-epoch=07-val_acc=0.6020.ckpt"
    make_post_training_figures(
        checkpoint_path,
        output_dir="outputs/figures/deepwalk/citeseer",
        data="Citeseer",
        model_name="deepwalk",
    )