# scripts/search_query.py
from src.embeddings import EmbeddingModel
from src.search import SearchEngine
from src.preprocessing import preprocess_text
import pandas as pd

# Paths
embeddings_path = "../data/full_docs_embeddings.npy"
queries_path = "../data/dev_small_queries - dev_small_queries.csv"

# Load queries
queries_df = pd.read_csv(queries_path)
raw_query_text = queries_df['Query'][0]
query_text = preprocess_text(raw_query_text)

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