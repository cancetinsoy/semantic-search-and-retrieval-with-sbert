# test_search.py
from src.embeddings import EmbeddingModel
from src.search import SearchEngine
import os

# Paths
data_path = "data/documents.txt"
query_path = "data/queries.txt"
embeddings_path = "data/document_embeddings.npy"

# Load documents
with open(data_path, 'r') as file:
    documents = file.readlines()

# Load queries
with open(query_path, 'r') as file:
    queries = file.readlines()

# Generate and save embeddings
model = EmbeddingModel()
if not os.path.exists(embeddings_path):
    document_embeddings = model.generate_embeddings(documents)
    model.save_embeddings(document_embeddings, embeddings_path)
else:
    document_embeddings = model.load_embeddings(embeddings_path)

# Initialize search engine
search_engine = SearchEngine(document_embeddings)

# Test with a query
query_embedding = model.generate_embeddings([queries[0]])[0]
results = search_engine.search(query_embedding)
print("Top results:", results)