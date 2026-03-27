# DeepWalk vs GNNs for Node Embeddings

## Overview

This project compares **structure-only** and **feature-aware** approaches to node classification on citation networks, implementing three methods from scratch:

| Method | Description |
|--------|-------------|
| **DeepWalk** | Random walks + Skip-gram to learn structural embeddings, treating graphs as NLP sequences |
| **GCN** | Spectral graph convolutions that aggregate neighbor features through learnable weights |
| **GAT** | Attention-weighted message passing that adaptively scores neighbor importance |

Evaluated on **Cora**, **Citeseer**, and **PubMed** using accuracy, Macro F1, t-SNE visualizations, and confusion matrices.

> **Core question:** When is graph structure alone sufficient, and when do feature-aware models provide a clear advantage?

---

## Project Structure
```
src/
├── datasets/
├── models/
├── lightning/
├── training/
└── evaluation/
outputs/figures/
```

---

## Installation
```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

---

## Usage

Configure the model and dataset in `src/training/train.py`, then run:
```bash
python src/training/train.py
```

| Option | Values |
|--------|--------|
| `model` | `deepwalk`, `gcn`, `gat` |
| `dataset` | `cora`, `citeseer`, `pubmed` |

---

## Results

Feature-aware models consistently outperform DeepWalk. GAT achieves the best results on Cora and Citeseer; GCN leads on PubMed.

---


## References

- **DeepWalk** — Perozzi et al., KDD 2014. [arXiv:1403.6652](https://arxiv.org/abs/1403.6652)  
- **GCN** — Kipf & Welling, ICLR 2017. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)  
- **GAT** — Veličković et al., ICLR 2018. [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)