# scripts/embed_documents.py
from src.embeddings import EmbeddingModel
from src.preprocessing import preprocess_text
import os

# Paths
data_folder = "../data/full_docs_small"
embeddings_path = "../data/full_docs_embeddings.npy"

# Load documents
def load_texts_from_folder(folder_path):
    """
    Load all text files from a folder, preprocess their content, and return.
    """
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                raw_text = file.read()
                preprocessed_text = preprocess_text(raw_text)
                texts.append(preprocessed_text)
    return texts

documents = load_texts_from_folder(data_folder)

# Generate and save embeddings
model = EmbeddingModel()
document_embeddings = model.generate_embeddings(documents)
model.save_embeddings(document_embeddings, embeddings_path)

print(f"Embedded {len(documents)} documents and saved to {embeddings_path}")