from pathlib import Path
from typing import Dict, Literal, Sequence, Union

import numpy as np
import pandas as pd
from faiss import IndexFlatL2, IndexHNSWFlat, IndexIVFFlat
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.helpers import documents_chunker
from src.search import aggregate, search

DocID = Union[int, str]


class VectorDatabase:
    """
    A class to handle vector-based document storage and retrieval
    using SentenceTransformer.

    Attributes
    ----------
    doc_ids : list
        A list to store document IDs.
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
        self.doc_ids = []
        self.doc_vectors = np.array([])
        self.encoder = encoder

    def encode_docs(self, doc_paths: Dict[DocID, Path], batch_size: int = 500):
        """
        Encodes documents from given paths and stores their vectors.

        Parameters
        ----------
        doc_paths : Dict[DocID, Path]
            A dictionary mapping document IDs to their file paths.
        batch_size : int, optional
            The number of documents to process in each batch (default is 500).
        """
        self.doc_ids = []
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
                self.doc_ids.append(doc_id)
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
        self,
        queries: Sequence[str],
        k: int,
        batch_size: int = 5000,
        verbose: bool = True,
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
        return self.doc_vectors.shape[0]

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
        df = pd.DataFrame({"index": range(len(self.doc_ids)), "doc_id": self.doc_ids})
        df.to_csv(path, index=False)

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
        df = pd.read_csv(path)
        self.doc_ids = df["doc_id"].tolist()

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
    doc_ids_in_cluster : list
        A list to store document IDs split by clusters.
    """

    def __init__(self, encoder: SentenceTransformer, n_clusters: int = 10):
        """
        Constructs all the necessary attributes for the ClusterDatabase object.

        Parameters
        ----------
        encoder : SentenceTransformer
            An encoder to encode documents.
        n_clusters : int, optional
            Number of clusters to form (default is 5).
        """
        super().__init__(encoder)
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.doc_ids_in_cluster = []

    def search_queries(
        self, queries: Sequence[str], k: int, top_c: int, verbose: bool = True
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
                enumerate(self.doc_ids_in_cluster),
                total=self.n_clusters,
                desc="Searching clusters",
            )
        else:
            iterator = enumerate(self.doc_ids_in_cluster)

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
        self.doc_ids_in_cluster = [
            np.where(cluster_labels == i)[0] for i in range(self.n_clusters)
        ]


class FaissDatabase(VectorDatabase):
    def __init__(
        self,
        model: SentenceTransformer,
        index_type: Literal["Flat", "Clustering", "HNSW"] = "Flat",
        n_clusters: int = 100,
        n_neighbours: int = 32,
        efConstruction: int = 200,
        efSearch: int = 100,
    ):
        super().__init__(model)
        self.index_type = index_type
        self.n_clusters = n_clusters
        self.n_neighbours = n_neighbours
        self.efConstruction = efConstruction
        self.efSearch = efSearch

        embedding_dim = self.encoder.get_sentence_embedding_dimension()

        if index_type == "Flat":
            self.index = IndexFlatL2(embedding_dim)
        elif index_type == "Clustering":
            quantizer = IndexFlatL2(embedding_dim)
            self.index = IndexIVFFlat(quantizer, embedding_dim, n_clusters)
        elif index_type == "HNSW":
            self.index = IndexHNSWFlat(embedding_dim, n_neighbours)
            self.index.hnsw.efConstruction = efConstruction

    def search_queries(self, queries: Sequence[str], k: int, top_c: int = 10, **kwargs):
        if self.index_type == "Clustering":
            self.index.nprobe = top_c
        elif self.index_type == "HNSW":
            self.index.hnsw.efSearch = self.efSearch

        query_vectors = self.encoder.encode(queries)
        _, top_k_indices = self.index.search(query_vectors, k)
        vectorized_translate = np.vectorize(self.translate_id)
        return vectorized_translate(top_k_indices.T)

    def train(self):
        self.index.train(self.doc_vectors)
        self.index.add(self.doc_vectors)


class ChunkedDatabase(VectorDatabase):
    def __init__(self, encoder):
        super().__init__(encoder)
        self.stride = 50
        self.chunk_length = self.encoder.max_seq_length

    def encode_docs(
        self,
        doc_paths: Dict[DocID, Path],
        batch_size: int = 500,
        stride: int = 50,
        chunk_length: int = 256,
    ):
        self.doc_vectors = []
        self.doc_ids = []
        self.stride = stride
        self.chunk_length = min(chunk_length, self.encoder.max_seq_length)

        texts = []

        progress = set()
        pbar = tqdm(total=len(doc_paths), desc=f"{0} document vectors encoded")

        for i, (doc_id, chunk) in enumerate(
            documents_chunker(
                doc_paths, self.encoder.tokenizer, self.chunk_length, self.stride
            )
        ):
            if doc_id not in progress:
                progress.add(doc_id)
                pbar.set_description(f"{i+1} document vectors encoded")
                pbar.update(1)
            texts.append(chunk)
            self.doc_ids.append(doc_id)

            if i % batch_size == 0 and i > 0:
                embeddings = self.encoder.encode(texts)
                self.doc_vectors.append(embeddings)
                texts = []
        print("encoding the last batch of {} documents".format(len(texts)))
        self.doc_vectors.append(self.encoder.encode(texts))

        self.doc_vectors = np.concatenate(self.doc_vectors)


class HnswDatabase(VectorDatabase):
    def __init__(self, encoder, n_neighbors=5):
        """
        Initializes an HNSW-based Vector Database.

        Parameters:
        encoder (object): An encoder object that provides a `.encode()` method to\
              encode queries into vectors.
        n_neighbors (int): The number of neighbors each node tries to connect to\
              at each level in the graph.
        """
        super().__init__(encoder)
        self.highest_levels = []  # Will store the highest assigned level per document
        self.edges = []  # Will store the edges between documents at each level
        self.n_neighbors = n_neighbors

    def search_queries(
        self, queries: Sequence[str], k: int, n_probes: int = 5, verbose=True
    ):
        """
        Searches the top-k nearest documents for a given set of queries.

        The search is performed by:
        1. Encoding the queries into vectors.
        2. Selecting random "probe" nodes in the highest possible layer.
        3. Traversing down the layers, moving from node to node based on similarity\
              to the query.
        4. Once the lowest layer is reached, gather candidates and select the top-k\
              based on similarity.

        Parameters:
        queries (Sequence[str]): A list of query strings.
        k (int): The number of top results to return.
        n_probes (int): The number of probes (starting points) to initialize the search.
        verbose (bool): If True, displays a progress bar for layer traversal.

        Returns:
        np.ndarray: A 2D array (k-by-query) of indices of the top-k closest documents\
              for each query.
        """
        query_vectors = self.encoder.encode(
            queries
        )  # Encode the textual queries into vectors

        # Determine the starting layer: it should be the highest layer that has at
        #  least n_probes documents.
        start_layer = max(
            (i for i in range(len(self.edges)) if len(self.edges[i]) >= n_probes)
        )

        # Randomly select n_probes documents from the chosen start layer to serve as
        # initial probes.
        probes = np.repeat(
            np.random.choice(self.layer(start_layer), n_probes, replace=False)[
                :, np.newaxis
            ],
            query_vectors.shape[0],
            axis=1,
        )

        # Create an iterator (with progress bar if verbose) to move from the top layer
        #  down to layer 0.
        if verbose:
            iterator = tqdm(
                range(start_layer, -1, -1),
                total=start_layer + 1,
                desc="Traversing layers",
            )
        else:
            iterator = range(start_layer, -1, -1)

        # Traverse layers from top to bottom, moving the probes to more similar
        #  neighbors.
        for i in iterator:
            probes = self._traverse_layer(
                probes,
                i,
                query_vectors,
                max_static=50,  # max_static: the maximum iterations with improvement
            )

        # After reaching the final layer, collect top-k documents.
        return self._get_top_k(probes, k, query_vectors)

    def highest_level(self) -> int:
        """
        Returns the highest level assigned to any document in the database.

        Returns:
        int: The highest level (0-based) among all documents, or -1 if no documents.
        """
        if self.highest_levels.size > 0:
            return np.max(self.highest_levels)
        else:
            return -1

    def layer(self, level: int) -> np.ndarray:
        """
        Returns the indices of documents that are at or above the specified level.

        If level == -1, uses the highest existing level in the database.

        Parameters:
        level (int): The layer level to filter documents by.

        Returns:
        np.ndarray: Array of document indices that have a highest_level >= `level`.
        """
        if level == -1:
            level = self.highest_level()
        if self.highest_levels.size > 0:
            return np.where(self.highest_levels >= level)[0]
        else:
            return np.array([])

    def train(self):
        """
        Builds the HNSW graph structure for the documents in the database.

        Steps:
        1. Assign a level to each document.
        2. For each layer, compute edges between documents based on cosine similarity.
        """
        # Assign a level to each document according to a probability distribution.
        self.highest_levels = self._assign_levels(
            self.n_neighbors, len(self.doc_vectors)
        )

        # Initialize the edges list. Each element in `edges` corresponds to a layer.
        self.edges = []

        # Construct layers from level 0 up to the max level.
        for level in tqdm(
            range(self.highest_level() + 1),
            total=self.highest_level() + 1,
            desc="Construction layers",
        ):
            # Get all documents that appear in this layer or above.
            current_layer = self.layer(level)

            # Compute the edges of the current layer using cosine similarity.
            layer_edges = self._make_edges(
                current_layer, self.doc_vectors, self.n_neighbors
            )
            self.edges.append(layer_edges)

    def _traverse_layer(
        self,
        probes: np.ndarray,
        current_level: int,
        query_vectors: np.ndarray,
        max_static: int = 50,
    ):
        """
        Guides the probes through the current layer towards nodes more similar to the\
              queries.

        This method attempts to move each probe from its current position to a neighbor\
              with higher similarity.
        If no neighbor has higher similarity, the probe is considered done at this\
              level.

        Parameters:
        probes (np.ndarray): 2D array where each row corresponds to a single probe and\
              each column to a query.
        current_level (int): The layer at which the traversal is currently taking place.
        query_vectors (np.ndarray): 2D array of encoded queries.
        max_static (int): Maximum number of iterations allowed without progress before\
              stopping.

        Returns:
        np.ndarray: Updated probes after attempting to move them to more similar\
              neighbors.
        """
        done = np.zeros_like(probes, dtype=np.bool_)
        current_layer_idxs = self.layer(current_level)

        # Map probe indices (document IDs) to their positions in the
        # current_layer array.
        probe_edge_idxs = self._get_edge_idxs(probes, current_layer_idxs)

        n_static = 0
        prev_fraction_done = 0

        # Continue until all probes are done or no progress is made.
        while not np.all(done) and max_static > n_static:
            # Remove duplicate probe positions to reduce redundant computations
            new_probes = np.unique(probes, axis=0)
            if probes.shape[0] != new_probes.shape[0]:
                probes = new_probes
                probe_edge_idxs = self._get_edge_idxs(new_probes, current_layer_idxs)

            # Get all neighbors of the current probe positions.
            neighbours = self._get_neighbours(
                probe_edge_idxs, self.edges[current_level]
            )
            nn_vector_idxs = self._get_vector_idxs(neighbours, current_layer_idxs)

            # Compute similarity of probes and their neighbors to the query vectors.
            probe_sims = np.einsum(
                "pqd,qd->pq", self.doc_vectors[probes], query_vectors
            )
            nn_sims = np.einsum(
                "pqnd,qd->pqn", self.doc_vectors[nn_vector_idxs], query_vectors
            )

            # For each probe-query pair, find the single best neighbor
            # (the one with highest similarity).
            best_idxs = np.argmax(nn_sims, axis=-1)
            best_nn_idxs = np.take_along_axis(
                neighbours, best_idxs[..., np.newaxis], axis=-1
            ).squeeze()
            best_sims = np.take_along_axis(
                nn_sims, best_idxs[..., np.newaxis], axis=-1
            ).squeeze()

            # Determine which probes are done (no better neighbor found).
            done = best_sims <= probe_sims
            fraction_done = np.sum(done) / np.size(done)

            # If no progress is made (fraction of done probes unchanged),
            # increment static count.
            if fraction_done == prev_fraction_done:
                n_static += 1
            else:
                n_static = 0
            prev_fraction_done = fraction_done

            # Update the probes that found better neighbors.
            probe_edge_idxs = np.where(done, probe_edge_idxs, best_nn_idxs)
            probes = self._get_vector_idxs(probe_edge_idxs, current_layer_idxs)

        return probes

    def _get_top_k(
        self, probes: np.ndarray, k: int, query_vectors: np.ndarray, level: int = 0
    ):
        """
        Given final probe positions, gather a pool of candidates and select the top-k\
              documents by similarity.

        This is a post-processing step to ensure we have at least k unique candidates.
        If not enough candidates are found, expand the search by looking at neighbors\
              in the given layer.

        Parameters:
        probes (np.ndarray): 2D array of probe positions after full layer traversal.
        k (int): Number of top similar documents to retrieve.
        query_vectors (np.ndarray): 2D array of encoded queries.
        level (int): The layer from which to expand candidates if needed.

        Returns:
        np.ndarray: A 2D array of shape (k, number_of_queries) containing document\
              indices of the top k results.
        """
        top_k = []
        for i in tqdm(
            range(query_vectors.shape[0]),
            total=query_vectors.shape[0],
            desc="Getting top k",
        ):
            # Get unique probe positions for this particular query.
            query_probes = np.unique(probes[:, i])

            # Ensure we have at least k candidates. If not, expand to neighbors.
            while query_probes.shape[0] < k:
                neighbours = self._get_neighbours(
                    self._get_edge_idxs(query_probes, self.layer(level)),
                    self.edges[level],
                )
                query_probes = np.unique(
                    np.concatenate([query_probes, neighbours.flatten()])
                )

            # Compute similarities of candidate documents to the current query.
            query_probe_sims = np.einsum(
                "pd,d->p", self.doc_vectors[query_probes], query_vectors[i]
            )

            # Select top-k candidates based on similarity.
            if query_probe_sims.shape[0] > k:
                top_k_indices = np.argpartition(-query_probe_sims, k)[:k]
            else:
                top_k_indices = np.argsort(-query_probe_sims)[:k]

            top_k.append(query_probes[top_k_indices])
        return np.array(top_k).T

    @staticmethod
    def _make_edges(current_layer, doc_vectors, n_neighbors):
        """
        Creates edges (connections) between documents in the current layer based on\
              cosine similarity.

        For each document in the layer:
        1. Compute its similarity to all other documents in this layer.
        2. Select the top `n_neighbors` most similar documents as edges.

        Parameters:
        current_layer (np.ndarray): Indices of documents in the current layer.
        doc_vectors (np.ndarray): Document embeddings.
        n_neighbors (int): Number of neighbors to connect each document to.

        Returns:
        np.ndarray: An array of shape (len(current_layer), n_neighbors) with indices of\
              connected neighbors.
        """
        k = (
            n_neighbors + 1
        )  # We add 1 because the document itself will appear as the top similarity
        edges = []
        for i in current_layer:
            # Compute similarities to all docs in current_layer
            sims = np.dot(doc_vectors[current_layer], doc_vectors[i])

            # Select the top neighbors (excluding the document itself)
            if sims.shape[0] > k:
                closest_indices = np.argpartition(-sims, k, axis=0)[1:k]
            else:
                closest_indices = np.argsort(-sims, axis=0)[1:k]

            edges.append(closest_indices)
        edges = np.array(edges)
        return edges

    @staticmethod
    def _get_neighbours(edge_idxs: np.ndarray, layer_edges: np.ndarray) -> np.ndarray:
        """
        Retrieve the neighbor indices for given edge indices.

        Parameters:
        edge_idxs (np.ndarray): Indices pointing into `layer_edges`.
        layer_edges (np.ndarray): Edges array for a particular layer.

        Returns:
        np.ndarray: An array of shape analogous to `edge_idxs`, containing the neighbor\
              indices.
        """
        return layer_edges[edge_idxs]

    @staticmethod
    def _get_vector_idxs(
        edge_idxs: np.ndarray, current_layer: np.ndarray
    ) -> np.ndarray:
        """
        Converts edge indices (positions within `current_layer`) back into actual\
              document indices.

        Parameters:
        edge_idxs (np.ndarray): Positions in the current layer array.
        current_layer (np.ndarray): Array of actual document indices at the current\
              layer.

        Returns:
        np.ndarray: Array of document indices corresponding to edge_idxs.
        """
        return np.vectorize(lambda idx: current_layer[idx])(edge_idxs)

    @staticmethod
    def _get_edge_idxs(
        vector_indxs: np.ndarray, current_layer: np.ndarray
    ) -> np.ndarray:
        """
        Converts from actual document indices to their corresponding positions within\
              `current_layer`.

        Parameters:
        vector_indxs (np.ndarray): Array of document indices.
        current_layer (np.ndarray): Array of documents at the current layer.

        Returns:
        np.ndarray: Positions of the document indices in `current_layer`.
        """
        return np.vectorize(lambda idx: np.where(current_layer == idx)[0][0])(
            vector_indxs
        )

    @staticmethod
    def _set_default_probas(n_neighbours: int) -> np.ndarray:
        """
        Compute a probability distribution over levels for assigning documents to\
              layers.

        The distribution is based on a geometric-like decay derived from the number of\
              neighbors.

        Parameters:
        n_neighbours (int): The number of neighbors each node tries to connect to at\
              each level.

        Returns:
        np.ndarray: An array of probabilities for each level.
        """
        # m_L is a parameter controlling the expected level distribution
        m_L = 1 / np.log(n_neighbours)
        nn = 0
        cum_nneighbor_per_level = []
        level = 0
        assign_probas = []
        while True:
            # Probability for the current level
            proba = np.exp(-level / m_L) * (1 - np.exp(-1 / m_L))

            # Stop if probability is extremely low
            if proba < 1e-9:
                break

            assign_probas.append(proba)
            # Track cumulative neighbors (for understanding, not directly used here)
            nn += n_neighbours * 2 if level == 0 else n_neighbours
            cum_nneighbor_per_level.append(nn)
            level += 1

        return np.array(assign_probas) / np.sum(assign_probas)

    @staticmethod
    def _assign_levels(n_neighbours: int, n_docs: int):
        """
        Assign levels to documents based on a probability distribution influenced by\
              n_neighbours.

        Higher levels are assigned increasingly rarely.

        Parameters:
        n_neighbours (int): Number of neighbors to consider for determining the\
              probability distribution.
        n_docs (int): Number of documents to assign levels to.

        Returns:
        np.ndarray: An array of assigned levels (one per document).
        """
        assign_probas = HnswDatabase._set_default_probas(n_neighbours)
        chosen_levels = np.random.choice(
            np.arange(len(assign_probas)), size=n_docs, p=assign_probas
        )
        return chosen_levels
