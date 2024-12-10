import numpy as np
from typing import Dict, Union, Callable, Sequence
from tqdm import tqdm

DocID = Union[int, str]
QueryID = Union[int, str]

def precision_at_k(
    relevant_docs: Sequence[DocID], retrieved_docs: Sequence[DocID], k: int
) -> float:
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = len(set(relevant_docs).intersection(set(retrieved_k)))
    return relevant_retrieved / k


def recall_at_k(
    relevant_docs: Sequence[DocID], retrieved_docs: Sequence[DocID], k: int
) -> float:
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = len(set(relevant_docs).intersection(set(retrieved_k)))
    total_relevant = len(relevant_docs)
    if total_relevant == 0:
        return 0.0
    return relevant_retrieved / total_relevant


def map_at_k(
    queries: Sequence[str],
    query_results: Dict[QueryID, DocID],
    index_directory: str,
    k: int,
    query_function: Callable,
) -> float:
    """mean average precision at k search results"""
    precisions = []
    for q_id, q in tqdm(queries.items(), desc=f"Processing queries for MAP@K={k}"):
        relevant_docs = query_results[q_id]
        retrieved_docs = [
            doc_id for doc_id, _ in query_function(index_directory, q, top_k=k)
        ]

        precision = precision_at_k(relevant_docs, retrieved_docs, k)
        precisions.append(precision)

    return float(np.mean(precisions))


def mar_at_k(
    queries: Sequence[str],
    query_results: Dict[QueryID, DocID],
    index_directory: str,
    k: int,
    query_function: Callable,
) -> float:
    """mean average recall at k search results"""
    recalls = []
    for q_id, q in tqdm(queries.items(), desc=f"Processing queries for MAR@K={k}"):
        relevant_docs = query_results[q_id]
        retrieved_docs = [
            doc_id for doc_id, _ in query_function(index_directory, q, top_k=k)
        ]

        recall = recall_at_k(relevant_docs, retrieved_docs, k)
        recalls.append(recall)

    return float(np.mean(recalls))


def retrieve_top_k_docs(
    queries: Dict[int, str], index_directory: str, query_function: Callable, k: int
) -> Dict[int, Sequence[int]]:
    """
    Retrieve the top k documents per query from the Lucene index.

    Args:
        queries (Dict[int, str]): A dictionary of queries with their IDs.
        index_directory (str): The path to the directory of the Lucene index.
        query_function (Callable): A function to retrieve documents from the database.
        k (int): Number of top documents to retrieve per query.

    Returns:
        Dict[int, Sequence[int]]: A dictionary mapping each query ID to its top k retrieved document IDs.
    """
    top_k_docs = {}
    for query_id, query_str in tqdm(
        queries.items(), desc=f"Retrieving top-{k} documents"
    ):
        # Retrieve the top k documents for the query
        retrieved_docs = [
            doc_id for doc_id, _ in query_function(index_directory, query_str, k)
        ]
        top_k_docs[query_id] = retrieved_docs
    return top_k_docs
