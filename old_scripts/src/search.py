# src/search.py
from scipy.spatial.distance import cosine

class SearchEngine:
    def __init__(self, document_embeddings):
        """
        Initialize the search engine with precomputed document embeddings.
        :param document_embeddings: Numpy array of document embeddings.
        """
        self.document_embeddings = document_embeddings

    def search(self, query_embedding, top_k=10):
        """
        Search for the top-k documents based on cosine similarity.
        :param query_embedding: Numpy array for the query embedding.
        :param top_k: Number of top results to return.
        :return: List of (document_index, similarity_score).
        """
        similarities = [
            1 - cosine(query_embedding, doc_embedding)
            for doc_embedding in self.document_embeddings
        ]
        ranked_results = sorted(
            enumerate(similarities), key=lambda x: x[1], reverse=True
        )
        return ranked_results[:top_k]