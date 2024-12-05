# scripts/search_query.py
from src.embeddings import EmbeddingModel
from src.search import SearchEngine
import pandas as pd

# Paths
embeddings_path = "data/full_docs_embeddings.npy"
queries_path = "data/dev_small_queries.csv"
data_folder = "data/full_docs_small"

# Load queries
queries_df = pd.read_csv(queries_path)
query_text = queries_df['query'][0]  # Test with the first query

# Load document embeddings
model = EmbeddingModel()
document_embeddings = model.load_embeddings(embeddings_path)

# Initialize search engine
search_engine = SearchEngine(document_embeddings)

# Embed the query and search
query_embedding = model.generate_embeddings([query_text])[0]
results = search_engine.search(query_embedding)

# Print results
print("Query:", query_text)
print("Top Results:")
for doc_index, score in results:
    print(f"Document {doc_index}: {score:.4f}")