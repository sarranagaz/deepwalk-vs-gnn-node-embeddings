"""Plotting utilities for visualizing training curves of the GAT model."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(
    metrics_csv_path: str,
    output_dir: str = "outputs/figures",
) -> None:
    """Plots the training and validation loss and accuracy curves from the CSV log file.

    Args:
        metrics_csv_path: Path to the CSV file containing the logged metrics from training.
        output_dir: Directory where the generated plots will be saved.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_csv_path)

    epoch_df = df.groupby("epoch", as_index=False).last()

    plt.figure(figsize=(6, 4))
    if "train_loss" in epoch_df.columns:
        plt.plot(epoch_df["epoch"], epoch_df["train_loss"], label="train_loss")
    if "val_loss" in epoch_df.columns:
        plt.plot(epoch_df["epoch"], epoch_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    if "train_acc" in epoch_df.columns:
        plt.plot(epoch_df["epoch"], epoch_df["train_acc"], label="train_acc")
    if "val_acc" in epoch_df.columns:
        plt.plot(epoch_df["epoch"], epoch_df["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "accuracy_curve.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    plot_training_curves("outputs/checkpoints/best-gat-epoch=29-val_acc=0.8040.ckpt")
