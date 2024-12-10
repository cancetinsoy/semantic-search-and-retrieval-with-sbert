from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.search import aggregate, search

DocID = Union[int, str]


class VectorDatabase:
    def __init__(self, encoder: SentenceTransformer):
        self.doc_ids = {}
        self.doc_vectors = np.array([])
        self.encoder = encoder

    def encode_docs(self, doc_paths: Dict[DocID, Path], batch_size=500):
        self.doc_ids = {}
        doc_vectors = []

        doc_texts = []
        for i, (doc_id, doc_path) in tqdm(
            enumerate(doc_paths.items()), total=len(doc_paths)
        ):
            with open(doc_path) as file:
                doc_text = file.read()
                doc_texts.append(doc_text)
                self.doc_ids[i] = doc_id
            if i % batch_size == 0 and i != 0:
                doc_vectors.append(np.array(self.encoder.encode(doc_texts)))
                doc_texts = []

        self.doc_vectors = np.concatenate(doc_vectors)

    def encode_text(self, text: Union[str, Sequence[str]]) -> np.ndarray:
        return np.array(self.encoder.encode(text))

    def search_queries(
        self, queries: Sequence[str], k: int, batch_size=5000, verbose=True
    ) -> np.ndarray:
        query_vectors = self.encode_text(queries)

        top_indices, _ = search(
            self.doc_vectors, query_vectors, k, batch_size, load_bar=verbose
        )
        vectorized_translate = np.vectorize(self.translate_id)

        return vectorized_translate(top_indices)

    def ntotal(self):
        return len(self.doc_vectors)

    def translate_id(self, index: int) -> DocID:
        return self.doc_ids[index]

    def get_vectors(self):
        return self.doc_vectors

    def save_vectors(self, path: Union[str, Path]):
        np.save(path, self.doc_vectors)

    def save_ids(self, path: Union[str, Path]):
        pd.DataFrame(self.doc_ids.items(), columns=["index", "doc_id"]).to_csv(
            path, index=False
        )

    def load_vectors(self, path: Union[str, Path]):
        self.doc_vectors = np.load(path)

    def load_ids(self, path: Union[str, Path]):
        self.doc_ids = pd.read_csv(path).set_index("index").to_dict()["doc_id"]

    def save_database(self, path: Union[str, Path]):
        Path.mkdir(Path(path), exist_ok=True, parents=True)
        self.save_vectors(Path(path) / "vectors.npy")
        self.save_ids(Path(path) / "ids.csv")

    def load_database(self, path: Union[str, Path]):
        self.load_vectors(Path(path) / "vectors.npy")
        self.load_ids(Path(path) / "ids.csv")

    def train(self):
        pass


class ClusterDatabase(VectorDatabase):
    def __init__(self, encoder: SentenceTransformer, n_clusters: int = 5):
        self.doc_ids = {}
        self.doc_vectors = np.array([])
        self.encoder = encoder
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.doc_ids_split = []

    def encode_docs(self, doc_paths: Dict[DocID, Path], batch_size=500):
        self.doc_vectors = []
        self.doc_ids = {}

        doc_texts = []
        doc_vectors = []
        for i, (doc_id, doc_path) in tqdm(
            enumerate(doc_paths.items()), total=len(doc_paths)
        ):
            with open(doc_path) as file:
                doc_text = file.read()
                doc_texts.append(doc_text)
                self.doc_ids[i] = doc_id
            if i % batch_size == 0 and i != 0:
                doc_vectors.append(np.array(self.encoder.encode(doc_texts)))
                doc_texts = []

        doc_vectors = np.concatenate(doc_vectors)
        self.train()

    def search_queries(
        self, queries: Sequence[str], k: int, top_c: int, verbose=True
    ) -> np.ndarray:
        query_vectors = self.encode_text(queries)
        top_c = min(top_c, self.n_clusters)

        # Compute similarity of cluster centers to queries
        sims = np.dot(self.kmeans.cluster_centers_, query_vectors.T)
        query_top_k_clusters = np.argsort(-sims, axis=0)[:top_c]

        retrieved_doc_ids = np.empty((top_c * k, len(queries)))
        retrieved_sims = np.empty((top_c * k, len(queries)))

        if verbose:
            iterator = tqdm(
                enumerate(self.doc_ids_split),
                total=self.n_clusters,
                desc="Searching clusters",
            )
        else:
            iterator = enumerate(self.doc_ids_split)

        for c, cluster_doc_ids in iterator:
            cluster_doc_vectors = self.doc_vectors[cluster_doc_ids]
            for i in range(top_c):
                queries_per_cluster = np.where(query_top_k_clusters[i, :] == c)[0]

                if queries_per_cluster.size == 0:
                    continue

                top_k_cluster_indices, top_k_sims = search(
                    cluster_doc_vectors,
                    query_vectors[queries_per_cluster],
                    k,
                    load_bar=False,
                )
                top_k_indices = cluster_doc_ids[top_k_cluster_indices]

                retrieved_doc_ids[i * k : (i + 1) * k, queries_per_cluster] = (
                    top_k_indices
                )
                retrieved_sims[i * k : (i + 1) * k, queries_per_cluster] = top_k_sims

        retrieved_doc_ids, _ = aggregate(retrieved_doc_ids, retrieved_sims, k)
        vectorized_translate = np.vectorize(self.translate_id)
        return vectorized_translate(retrieved_doc_ids)

    def train(self):
        self.kmeans.fit(self.doc_vectors)
        cluster_labels = self.kmeans.labels_
        self.doc_ids_split = [
            np.where(cluster_labels == i)[0] for i in range(self.n_clusters)
        ]

    def load_vectors(self, path: Union[str, Path]):
        self.doc_vectors = np.load(path)
        self.train()

    def load_ids(self, path: Union[str, Path]):
        self.doc_ids = pd.read_csv(path).set_index("index").to_dict()["doc_id"]


# class HNSWDatabase(VectorDatabase):
#     def __init__(
#         self,
#         encoder: SentenceTransformer,
#         space: str = "cosine",
#         ef_construction: int = 200,
#         M: int = 16,
#     ):
#         super().__init__(encoder)
#         self.space = space
#         self.ef_construction = ef_construction
#         self.M = M
#         self.index = hnswlib.Index(
#             space=space, dim=encoder.get_sentence_embedding_dimension()
#         )
#         self.index.init_index(max_elements=0, ef_construction=ef_construction, M=M)
#         self.index.set_ef(ef_construction)

#     def encode_docs(self, doc_paths: Dict[DocID, Path], batch_size=500):
#         self.doc_ids = {}
#         doc_vectors = []

#         doc_texts = []
#         for i, (doc_id, doc_path) in tqdm(
#             enumerate(doc_paths.items()), total=len(doc_paths)
#         ):
#             with open(doc_path) as file:
#                 doc_text = file.read()
#                 doc_texts.append(doc_text)
#                 self.doc_ids[i] = doc_id
#             if i % batch_size == 0 and i != 0:
#                 vectors = np.array(self.encoder.encode(doc_texts))
#                 self.index.add_items(vectors, np.arange(len(vectors)))
#                 doc_vectors.append(vectors)
#                 doc_texts = []

#         if doc_texts:
#             vectors = np.array(self.encoder.encode(doc_texts))
#             self.index.add_items(vectors, np.arange(len(vectors)))
#             doc_vectors.append(vectors)

#         self.doc_vectors = np.concatenate(doc_vectors)

#     def search_queries(self, queries: Sequence[str], k: int) -> np.ndarray:
#         query_vectors = self.encode_text(queries)
#         labels, distances = self.index.knn_query(query_vectors, k=k)
#         vectorized_translate = np.vectorize(self.translate_id)
#         return vectorized_translate(labels)

#     def save_index(self, path: Union[str, Path]):
#         self.index.save_index(str(path))

#     def load_index(self, path: Union[str, Path]):
#         self.index.load_index(str(path))
#         self.index.set_ef(self.ef_construction)

#     def save_database(self, path: Union[str, Path]):
#         super().save_database(path)
#         self.save_index(Path(path) / "hnsw_index.bin")

#     def load_database(self, path: Union[str, Path]):
#         super().load_database(path)
#         self.load_index(Path(path) / "hnsw_index.bin")
