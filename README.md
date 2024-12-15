# Vector Database and Evaluation Framework

## Overview
This project provides a framework for building and evaluating vector-based document retrieval systems. It supports encoding documents, performing searches, and evaluating performance using metrics like precision, recall, and F1 score.

### Key Components
1. **VectorDatabase**: Generic class for vector storage and retrieval.
2. **ClusterDatabase**: Uses KMeans for clustering documents.
3. **FaissDatabase**: Utilizes FAISS for efficient indexing.
4. **ChunkedDatabase**: Handles chunked document retrieval.
5. **HnswDatabase**: Implements HNSW graphs for fast search.
6. **Evaluation Functions**: Metrics for retrieval performance.

## Installation

### Requirements
- Python 3.8+
- Libraries: `numpy`, `pandas`, `tqdm`, `scikit-learn`, `faiss`, `sentence-transformers`

Install dependencies:
```bash
pip install numpy pandas tqdm scikit-learn faiss-cpu sentence-transformers
```

## Usage

### Setting Up a Database

#### Initialize
```python
from sentence_transformers import SentenceTransformer
from vector_database import VectorDatabase

encoder = SentenceTransformer('all-MiniLM-L6-v2')
database = VectorDatabase(encoder)
```

#### Encode Documents
```python
from pathlib import Path

doc_paths = {1: Path("/path/to/doc1.txt"), 2: Path("/path/to/doc2.txt")}
database.encode_docs(doc_paths)
```

#### Save and Load
```python
database.save_database("/path/to/save")
database.load_database("/path/to/save")
```

### Perform Searches
```python
queries = ["Example query"]
k = 5
top_results = database.search_queries(queries, k)
```

### Specialized Databases

#### ClusterDatabase
```python
from vector_database import ClusterDatabase

cluster_db = ClusterDatabase(encoder, n_clusters=10)
cluster_db.encode_docs(doc_paths)
results = cluster_db.search_queries(queries, k=5, top_c=3)
```

#### FaissDatabase
```python
from vector_database import FaissDatabase

faiss_db = FaissDatabase(encoder, index_type="HNSW")
faiss_db.encode_docs(doc_paths)
results = faiss_db.search_queries(queries, k=5)
```

#### ChunkedDatabase
```python
from vector_database import ChunkedDatabase

chunked_db = ChunkedDatabase(encoder)
chunked_db.encode_docs(doc_paths)
results = chunked_db.search_queries(queries, k=5)
```

#### HnswDatabase
```python
from vector_database import HnswDatabase

hnsw_db = HnswDatabase(encoder, n_neighbors=5)
hnsw_db.encode_docs(doc_paths)
results = hnsw_db.search_queries(queries, k=5)
```

### Evaluation

#### Precision and Recall
```python
from evaluation import evaluate_database_queries

queries = {1: "Query 1", 2: "Query 2"}
query_results = {1: [1, 3], 2: [2, 4]}  # Ground truth
k_values = [1, 5, 10]

metrics = evaluate_database_queries(queries, query_results, k_values, database)
```

#### Chunked Documents
```python
from evaluation import evaluate_queries_in_chunked_docs

metrics = evaluate_queries_in_chunked_docs(queries, query_results, k_values, chunked_db)
```

#### F1 Score
```python
from evaluation import calculate_f1

precision = 0.8
recall = 0.6
f1_score = calculate_f1(precision, recall)
```

## Extending the Framework

### Adding a Database
1. Extend `VectorDatabase`.
2. Implement `encode_docs` and `search_queries`.

### Custom Metrics
Follow the structure of `precision_at_k` and `recall_at_k`.

## Contributing
Submit pull requests or open issues on GitHub.

## License
MIT License.

## Acknowledgements
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)

