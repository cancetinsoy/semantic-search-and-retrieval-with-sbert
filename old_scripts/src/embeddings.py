# src/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name='all-mpnet-base-v2'):
        """
        Initialize the Sentence-BERT model.
        """
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of texts.
        :param texts: List of strings (documents/queries).
        :return: Numpy array of embeddings.
        """
        return np.array(self.model.encode(texts, convert_to_numpy=True))

    def save_embeddings(self, embeddings, filepath):
        """
        Save embeddings to a file.
        :param embeddings: Numpy array of embeddings.
        :param filepath: Path to save the embeddings.
        """
        np.save(filepath, embeddings)

    def load_embeddings(self, filepath):
        """
        Load embeddings from a file.
        :param filepath: Path to load embeddings from.
        :return: Numpy array of embeddings.
        """
        return np.load(filepath)