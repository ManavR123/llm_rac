"""Approximate Nearest Neighbors index for storing and querying embeddings."""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, NamedTuple, Optional, Set

import numpy as np
import pandas as pd
from hnswlib import Index as HnswIndex  # pylint: disable=no-name-in-module
from sentence_transformers import SentenceTransformer

INDEX_FILE_NAME = "index.bin"
METADATA_FILE_NAME = "metadata.json"
INDEX_PARAMS_FILE_NAME = "index_params.json"


class ClassificationExample(NamedTuple):
    """A single example for classification."""

    text: str
    label: str


class IndexParams(NamedTuple):
    """Parameters for building an index."""

    output_dim: int
    ef: int = 200
    ef_construction: int = 500
    M: int = 64


class ANN:
    """Wrapper class around HNSW for managing an index of embeddings and their labels."""

    def __init__(
        self,
        num_items: int,
        index_params: IndexParams,
        *,
        metadata: Optional[Dict[int, ClassificationExample]] = None,
    ):
        self.index_params = index_params
        self._hnsw = HnswIndex(space="ip", dim=self.index_params.output_dim)
        self._hnsw.init_index(num_items, ef_construction=self.index_params.ef_construction, M=self.index_params.M)
        self._hnsw.set_ef(self.index_params.ef)
        self.metadata: Dict[int, ClassificationExample] = metadata or {}

    @classmethod
    def build_index_from_data(
        cls,
        data: pd.DataFrame,
        model: SentenceTransformer,
        max_index_size: Optional[int] = None,
        batch_size: int = 32,
    ) -> ANN:
        """Create an ANN index from a dataframe with columns "text" and "label"."""
        embedding_items = data.sample(min(len(data), max_index_size)) if max_index_size else data
        index = cls(
            num_items=len(embedding_items),
            index_params=IndexParams(
                output_dim=model.get_sentence_embedding_dimension(),
            ),
        )
        for i in range(0, len(embedding_items), batch_size):
            items = embedding_items.iloc[i : i + batch_size]
            texts = [item["text"] for _, item in items.iterrows()]
            labels = [item["label"] for _, item in items.iterrows()]
            embs = model.encode(texts)
            ids = list(range(i, min(len(embedding_items), i + batch_size)))
            index.add_items(ids, texts, labels, embs)
        return index

    def add_items(
        self,
        ids: List[int],
        texts: List[str],
        labels: List[str],
        vectors: np.ndarray,
        num_threads: int = -1,
    ):
        """Add a list of vectors and their corresponding IDs to the index"""
        if len(ids) == 0:
            # The hnsw api throws "wrong dimensionality of the vectors" if passed an empty list.
            return
        self._hnsw.add_items(vectors, ids, num_threads)
        for index_id, text, label in zip(ids, texts, labels):
            self.metadata[index_id] = ClassificationExample(text=text, label=label)

    def save(self, output_path: str) -> None:
        """Save the index and metadata to disk."""
        self._hnsw.save_index(os.path.join(output_path, INDEX_FILE_NAME))
        with open(os.path.join(output_path, INDEX_PARAMS_FILE_NAME), "w", encoding="utf-8") as file:
            json.dump(self.index_params._asdict(), file)
        with open(os.path.join(output_path, METADATA_FILE_NAME), "w", encoding="utf-8") as file:
            json.dump(self.metadata, file)

    @classmethod
    def load(cls, output_path: str, max_elements: int = 0) -> ANN:
        """Load the index and metadata from disk."""
        with open(os.path.join(output_path, INDEX_PARAMS_FILE_NAME), "r", encoding="utf-8") as file:
            index_params = IndexParams(**json.load(file))
        with open(os.path.join(output_path, METADATA_FILE_NAME), "r", encoding="utf-8") as file:
            serialized_metdata = json.load(file)
        metadata = {int(index_id): ClassificationExample(*example) for index_id, example in serialized_metdata.items()}
        index = cls(
            num_items=len(metadata),
            index_params=index_params,
            metadata=metadata,
        )
        index._hnsw.load_index(os.path.join(output_path, INDEX_FILE_NAME), max_elements=max_elements)
        return index

    def size(self):
        """Return the number of items in the index."""
        return self._hnsw.get_current_count()

    def get_similar_examples(self, vector: np.ndarray, k: int, min_score: float = 0.0) -> List[ClassificationExample]:
        """Return the k nearest neighbors to a vector."""
        k = min(k, self.size())
        try:
            ids, distances = self._hnsw.knn_query(vector, k)
        except RuntimeError:
            logging.warning("Runtime error during kNN, k (%d) is probably too high wrt index size (%d)", k, self.size())
            return []
        # hnsw returns 1 - dot.product, add clipping to clean any FP precision issues
        distances = np.clip(1 - distances[0], -1, 1).tolist()
        return [self.metadata[index_id] for index_id, distance in zip(ids[0], distances) if distance >= min_score]

    def get_labels(self) -> Set[str]:
        """Return the set of labels in the index."""
        return set(example.label for example in self.metadata.values())
