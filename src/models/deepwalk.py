"""DeepWalk model implementation."""

import random

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec


class DeepWalk(nn.Module):
    """DeepWalk encoder and classifier for node classification."""

    def __init__(
        self,
        num_features: int,
        embedding_dim: int = 128,
        walk_length: int = 40,
        num_walks: int = 10,
        window_size: int = 5,
        w2v_epochs: int = 5,
        classifier_hidden_dim: int = 64,
        nclass: int = 7,
        workers: int = 1,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.w2v_epochs = w2v_epochs
        self.classifier_hidden_dim = classifier_hidden_dim
        self.nclass = nclass
        self.workers = workers
        self.seed = seed
        self.num_features = num_features

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim + num_features, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(classifier_hidden_dim, nclass),
        )

        self.embeddings: torch.Tensor | None = None

    @staticmethod
    def build_adjacency_list(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
        """Build adjacency list from edge_index."""
        adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
        src_nodes = edge_index[0].tolist()
        dst_nodes = edge_index[1].tolist()

        for src, dst in zip(src_nodes, dst_nodes):
            adjacency[src].append(dst)

        return adjacency

    def random_walk(self, adjacency: list[list[int]], start_node: int) -> list[str]:
        """Generate one random walk starting from a node."""
        walk = [start_node]

        while len(walk) < self.walk_length:
            current = walk[-1]
            neighbors = adjacency[current]
            if not neighbors:
                break
            walk.append(random.choice(neighbors))

        return [str(node) for node in walk]

    def generate_walks(self, adjacency: list[list[int]]) -> list[list[str]]:
        """Generate all random walks for DeepWalk."""
        random.seed(self.seed)
        np.random.seed(self.seed)

        nodes = list(range(len(adjacency)))
        walks: list[list[str]] = []

        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk(adjacency, node))

        return walks

    def fit_embeddings(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Fit DeepWalk embeddings from graph structure."""
        adjacency = self.build_adjacency_list(edge_index=edge_index, num_nodes=num_nodes)
        walks = self.generate_walks(adjacency)

        w2v = Word2Vec(
            sentences=walks,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=0,
            sg=1,
            hs=1,
            workers=self.workers,
            epochs=self.w2v_epochs,
            seed=self.seed,
        )

        emb_matrix = np.zeros((num_nodes, self.embedding_dim), dtype=np.float32)
        for node in range(num_nodes):
            emb_matrix[node] = w2v.wv[str(node)]

        self.embeddings = torch.tensor(emb_matrix, dtype=torch.float32)
        return self.embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits from input representations."""
        return self.classifier(x)

    def get_embeddings(self) -> torch.Tensor:
        """Return learned DeepWalk embeddings."""
        if self.embeddings is None:
            raise RuntimeError("Embeddings are not initialized.")
        return self.embeddings