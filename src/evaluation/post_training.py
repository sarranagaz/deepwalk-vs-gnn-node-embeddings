"""Post-training evaluation and visualization for the GAT model."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from src.lightning.datamodule import GraphDataModule
from src.lightning.gat_module import LitGAT


def load_best_model(checkpoint_path: str, datamodule: GraphDataModule) -> LitGAT:
    """Loads the best model checkpoint for evaluation.

    Args:
        checkpoint_path: Path to the best model checkpoint file.
        datamodule: The data module used for training, needed to get the graph data for initializing the model.

    Returns:
        LitGAT: The loaded model ready for evaluation.
    """
    graph = datamodule.graph_data
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
    model.eval()
    return model  # type: ignore


def make_post_training_figures(checkpoint_path: str, output_dir: str = "outputs/figures", data: str = "Cora") -> None:
    """Generates post-training evaluation figures such as confusion matrix, attention weights, and t-SNE of embeddings.

    Args:
        checkpoint_path: Path to the best model checkpoint file.
        output_dir: Directory where the generated figures will be saved.
        data: The dataset to evaluate on (either "Cora" or "PubMed").
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if data in ["Cora", "Citeseer", "PubMed"]:
        datamodule = GraphDataModule(name=data, root="data", self_loops=True)
    else:
        raise ValueError(f"Unsupported dataset: {data}. Supported datasets are 'Cora', 'Citeseer', and 'PubMed'.")
        
    datamodule.setup()
    graph = datamodule.graph_data

    model = load_best_model(checkpoint_path, datamodule)

    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)  # type: ignore
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
        embeddings = model.model.get_embeddings(graph.x, graph.edge_index).cpu().numpy()  # type: ignore

    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    emb_2d = tsne.fit_transform(embeddings)
    labels = graph.y.cpu().numpy()  # type: ignore

    plt.figure(figsize=(7, 6))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, s=10)
    plt.title("t-SNE of GAT embeddings")
    plt.tight_layout()
    plt.savefig(outdir / "tsne_embeddings.png", dpi=200)
    plt.close()

    with torch.no_grad():
        attention_weights = model.model.get_attention_weights(graph.x, graph.edge_index).cpu().numpy()  # type: ignore
    # For simplicity, we will just plot the attention weights for the first 10 nodes
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
    checkpoint_path = "outputs/checkpoints/best-gat-epoch=21-val_acc=0.6980.ckpt"
    make_post_training_figures(checkpoint_path, output_dir="outputs/figures/citeseer", data="Citeseer")
