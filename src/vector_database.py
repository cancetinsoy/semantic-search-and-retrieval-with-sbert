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
    """
    A class to handle vector-based document storage and retrieval
    using SentenceTransformer.

    Attributes
    ----------
    doc_ids : dict
        A dictionary to store document IDs.
    doc_vectors : np.ndarray
        An array to store document vectors.
    encoder : SentenceTransformer
        An encoder to encode documents.
    """

    def __init__(self, encoder: SentenceTransformer):
        """
        Initializes the VectorDatabase with a given encoder.

        Parameters
        ----------
        encoder : SentenceTransformer
            The encoder used to transform documents and queries into vectors.
        """
        self.doc_ids = {}
        self.doc_vectors = np.array([])
        self.encoder = encoder

    def encode_docs(self, doc_paths: Dict[DocID, Path], batch_size=500):
        """
        Encodes documents from given paths and stores their vectors.

        Parameters
        ----------
        doc_paths : Dict[DocID, Path]
            A dictionary mapping document IDs to their file paths.
        batch_size : int, optional
            The number of documents to process in each batch (default is 500).
        """
        self.doc_ids = {}
        doc_vectors = []

        doc_texts = []
        for i, (doc_id, doc_path) in tqdm(
            enumerate(doc_paths.items()), total=len(doc_paths)
        ):
            # Read the document text from the file
            with open(doc_path) as file:
                doc_text = file.read()
                doc_texts.append(doc_text)
                # Store the document ID
                self.doc_ids[i] = doc_id
                # Encode documents in batches
                if i % batch_size == 0 and i != 0:
                    # Encode the batch of document texts and store the vectors
                    doc_vectors.append(np.array(self.encoder.encode(doc_texts)))
                    doc_texts = []

        # Encode any remaining documents
        if doc_texts:
            doc_vectors.append(np.array(self.encoder.encode(doc_texts)))

        # Concatenate all document vectors into a single array
        self.doc_vectors = np.concatenate(doc_vectors)

        # Train the database if needed
        self.train()

    def encode_text(self, text: Union[str, Sequence[str]]) -> np.ndarray:
        """
        Encodes a given text or sequence of texts into vectors.

        Parameters
        ----------
        text : Union[str, Sequence[str]]
            The text or sequence of texts to encode.

        Returns
        -------
        np.ndarray
            The encoded vectors.
        """
        return np.array(self.encoder.encode(text))

    def search_queries(
        self, queries: Sequence[str], k: int, batch_size=5000, verbose=True
    ) -> np.ndarray:
        """
        Searches for the top k most similar documents for each query.

        Parameters
        ----------
        queries : Sequence[str]
            The queries to search for.
        k : int
            The number of top similar documents to retrieve for each query.
        batch_size : int, optional
            The number of queries to process in each batch (default is 5000).
        verbose : bool, optional
            Whether to display a progress bar (default is True).

        Returns
        -------
        np.ndarray
            The IDs of the top k most similar documents for each query.
        """
        query_vectors = self.encode_text(queries)

        top_indices, _ = search(
            self.doc_vectors, query_vectors, k, batch_size, load_bar=verbose
        )
        vectorized_translate = np.vectorize(self.translate_id)

        return vectorized_translate(top_indices)

    def ntotal(self):
        """
        Returns the total number of document vectors stored.

        Returns
        -------
        int
            The total number of document vectors.
        """
        return len(self.doc_vectors)

    def translate_id(self, index: int) -> DocID:
        """
        Translates an internal index to the corresponding document ID.

        Parameters
        ----------
        index : int
            The internal index.

        Returns
        -------
        DocID
            The corresponding document ID.
        """
        return self.doc_ids[index]

    def get_vectors(self):
        """
        Returns the stored document vectors.

        Returns
        -------
        np.ndarray
            The stored document vectors.
        """
        return self.doc_vectors

    def save_vectors(self, path: Union[str, Path]):
        """
        Saves the document vectors to a file.

        Parameters
        ----------
        path : Union[str, Path]
            The file path to save the vectors.
        """
        np.save(path, self.doc_vectors)

    def save_ids(self, path: Union[str, Path]):
        """
        Saves the document IDs to a CSV file.

        Parameters
        ----------
        path : Union[str, Path]
            The file path to save the IDs.
        """
        pd.DataFrame(self.doc_ids.items(), columns=["index", "doc_id"]).to_csv(
            path, index=False
        )

    def load_vectors(self, path: Union[str, Path]):
        """
        Loads document vectors from a file and trains the database (if implemented).

        Parameters
        ----------
        path : Union[str, Path]
            The file path to load the vectors from.
        """
        self.doc_vectors = np.load(path)
        self.train()

    def load_ids(self, path: Union[str, Path]):
        """
        Loads document IDs from a CSV file.

        Parameters
        ----------
        path : Union[str, Path]
            The file path to load the IDs from.
        """
        self.doc_ids = pd.read_csv(path).set_index("index").to_dict()["doc_id"]

    def save_database(self, path: Union[str, Path]):
        """
        Saves the entire database (vectors and IDs) to a directory.

        Parameters
        ----------
        path : Union[str, Path]
            The directory path to save the database.
        """
        Path.mkdir(Path(path), exist_ok=True, parents=True)
        self.save_vectors(Path(path) / "vectors.npy")
        self.save_ids(Path(path) / "ids.csv")

    def load_database(self, path: Union[str, Path]):
        """
        Loads the entire database (vectors and IDs) from a directory.

        Parameters
        ----------
        path : Union[str, Path]
            The directory path to load the database from.
        """
        self.load_vectors(Path(path) / "vectors.npy")
        self.load_ids(Path(path) / "ids.csv")

    def train(self):
        """
        Placeholder method for training, since it's a default -> not implemented.
        """
        pass


class ClusterDatabase(VectorDatabase):
    """
    A class used to represent a Cluster Database that extends the VectorDatabase.

    Attributes
    ----------
    doc_ids : dict
        A dictionary to store document IDs.
    doc_vectors : np.ndarray
        An array to store document vectors.
    encoder : SentenceTransformer
        An encoder to encode documents.
    n_clusters : int
        Number of clusters to form.
    kmeans : KMeans
        KMeans clustering model.
    doc_ids_split : list
        A list to store document IDs split by clusters.
    """

    def __init__(self, encoder: SentenceTransformer, n_clusters: int = 5):
        """
        Constructs all the necessary attributes for the ClusterDatabase object.

        Parameters
        ----------
        encoder : SentenceTransformer
            An encoder to encode documents.
        n_clusters : int, optional
            Number of clusters to form (default is 5).
        """
        self.doc_ids = {}
        self.doc_vectors = np.array([])
        self.encoder = encoder
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.doc_ids_split = []

    def search_queries(
        self, queries: Sequence[str], k: int, top_c: int, verbose=True
    ) -> np.ndarray:
        """
        Searches for the top k documents for each query.

        Parameters
        ----------
        queries : Sequence[str]
            A sequence of query strings.
        k : int
            Number of top documents to retrieve for each query.
        top_c : int
            Number of top clusters to consider for each query.
        verbose : bool, optional
            Whether to display a progress bar (default is True).

        Returns
        -------
        np.ndarray
            An array of retrieved document IDs for each query.
        """
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
        """
        Trains the KMeans model on the document vectors.
        """
        self.kmeans.fit(self.doc_vectors)
        cluster_labels = self.kmeans.labels_
        self.doc_ids_split = [
            np.where(cluster_labels == i)[0] for i in range(self.n_clusters)
        ]


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
