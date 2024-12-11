from typing import Dict, Mapping, Sequence, Union

import numpy as np

from .vector_database import VectorDatabase

DocID = Union[int, str]
QueryID = Union[int, str]


def precision_at_k(
    relevant_docs: Sequence[DocID], retrieved_docs: Sequence[DocID], k: int
) -> float:
    """
    Calculate the precision at rank k for a set of retrieved documents.

    Precision at k is the proportion of relevant documents in the top-k \
        retrieved documents.

    Args:
        relevant_docs (Sequence[DocID]): A sequence of document IDs that are relevant.
        retrieved_docs (Sequence[DocID]): A sequence of document IDs that have \
            been retrieved.
        k (int): The rank position up to which precision is calculated.

    Returns:
        float: The precision at rank k, which is the number of relevant documents in \
            the top-k retrieved documents divided by k.
    """
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = len(set(relevant_docs).intersection(set(retrieved_k)))
    return relevant_retrieved / k


def recall_at_k(
    relevant_docs: Sequence[DocID], retrieved_docs: Sequence[DocID], k: int
) -> float:
    """
    Calculate the recall at rank k.

    Recall at k is the proportion of relevant documents that are retrieved in
    the top k results.

    Args:
        relevant_docs (Sequence[DocID]): A sequence of document IDs that are relevant.
        retrieved_docs (Sequence[DocID]): A sequence of document IDs that are retrieved.
        k (int): The rank position up to which to evaluate recall.

    Returns:
        float: The recall at rank k. If there are no relevant documents, returns 0.0.
    """
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = len(set(relevant_docs).intersection(set(retrieved_k)))
    total_relevant = len(relevant_docs)
    if total_relevant == 0:
        return 0.0
    return relevant_retrieved / total_relevant


def evaluate_queries(
    queries: Dict[QueryID, str],
    query_results: Mapping[QueryID, Sequence[DocID]],
    k_values: list[int],
    database: VectorDatabase,
    **search_kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the performance of search queries using precision and recall at different \
        k values.

    Args:
        queries (Dict[QueryID, str]): A dictionary mapping query IDs to query texts.
        query_results (Mapping[QueryID, Sequence[DocID]]): A mapping of query IDs to \
            sequences of relevant document IDs.
        k_values (list[int]): A list of k values to evaluate precision and recall at.
        database (VectorDatabase): An instance of the VectorDatabase to perform the \
            search queries.
        **search_kwargs: Additional keyword arguments to pass to the search function.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary where each key is a k value and the \
            value is another dictionary with keys "Precision" and "Recall" mapping to \
                their respective average values.
    """

    query_ids = list(queries.keys())
    query_texts = list(queries.values())

    top_k_indices = database.search_queries(
        query_texts, k=max(k_values), **search_kwargs
    )

    output = {}
    for k in k_values:
        precisions = []
        recalls = []
        for i, query_id in enumerate(query_ids):
            retrieved = top_k_indices[:k, i]
            relevant = query_results[query_id]
            precisions.append(precision_at_k(list(relevant), list(retrieved), k))
            recalls.append(recall_at_k(list(relevant), list(retrieved), k))
        precisions = np.mean(precisions)
        recalls = np.mean(recalls)

        output[k] = {"Precision": precisions, "Recall": recalls}

    return output


def calculate_f1(precision, recall):
    """
    Calculate the F1 score given precision and recall values.

    Parameters:
    precision (float): The precision value.
    recall (float): The recall value.

    Returns:
    float: The calculated F1 score.
    """
    if precision + recall == 0:
        return 0.0

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score
