from typing import Dict, Mapping, Sequence, Union

import numpy as np

from .helpers import get_top_k_unique
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
    query_ids: Sequence[QueryID],
    top_k_indices: np.ndarray,
    query_results: Mapping[QueryID, Sequence[DocID]],
    k_values: list[int],
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the performance of a retrieval process at various cutoffs (k-values).

    This function computes precision and recall metrics for a set of queries
    given the retrieved document indices. The evaluation is done for multiple k-values.

    Parameters
    ----------
    query_ids : Sequence[QueryID]
        A sequence of query identifiers. Each query ID corresponds to a unique query.
    top_k_indices : np.ndarray
        A 2D NumPy array where each column corresponds to a query and each row\
              corresponds to a retrieved document index. `top_k_indices[:k, i]` gives\
                  the top-k results for the i-th query.
    query_results : Mapping[QueryID, Sequence[DocID]]
        A mapping from a query ID to its list of *relevant* document IDs. These \
            represent the ground truth relevance judgments for evaluation.
    k_values : list[int]
        A list of integers specifying the different cutoffs (e.g., k=1, k=5, k=10) at\
              which the metrics (Precision, Recall) should be computed.

    Returns
    -------
    Dict[str, Dict[str, float]]
        A dictionary where each key is a k-value, and each value is another dictionary
        containing the computed "Precision" and "Recall" at that k.
        For example:
        {
            "1": {"Precision": ..., "Recall": ...},
            "5": {"Precision": ..., "Recall": ...},
            ...
        }

    Notes
    -----
    - Precision at k measures how many of the retrieved documents up to rank k are\
          relevant.
    - Recall at k measures how many of the total relevant documents were retrieved\
          within
      the top k results.
    - This function assumes that `precision_at_k` and `recall_at_k` are pre-defined\
          helper functions that compute these metrics for given relevant and retrieved\
              sets.
    """

    output = {}
    for k in k_values:
        precisions = []
        recalls = []
        # Loop through each query and compute metrics
        for i, query_id in enumerate(query_ids):
            # Extract the top-k retrieved documents for this query
            retrieved = top_k_indices[:k, i]
            # Get the ground truth relevant documents for this query
            relevant = query_results[query_id]

            # Compute precision and recall at k
            precisions.append(precision_at_k(list(relevant), list(retrieved), k))
            recalls.append(recall_at_k(list(relevant), list(retrieved), k))

        # Compute the mean precision and recall across all queries for the current k
        precisions = np.mean(precisions)
        recalls = np.mean(recalls)

        output[k] = {"Precision": precisions, "Recall": recalls}

    return output


def evaluate_database_queries(
    queries: Dict[QueryID, str],
    query_results: Mapping[QueryID, Sequence[DocID]],
    k_values: list[int],
    database: VectorDatabase,
    **search_kwargs,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate retrieval performance of a vector database on a set of queries.

    This function takes a set of queries (IDs and texts), retrieves results from a given
    vector database, and then calculates precision and recall at various k-values.

    Parameters
    ----------
    queries : Dict[QueryID, str]
        A dictionary mapping query IDs to their corresponding query text.
    query_results : Mapping[QueryID, Sequence[DocID]]
        Ground truth relevant documents for each query.
    k_values : list[int]
        The cutoff values at which the evaluation metrics will be computed.
    database : VectorDatabase
        An instance of a vector database that allows searching over a collection of\
              documents.
    **search_kwargs
        Additional keyword arguments to be passed to the database search function.

    Returns
    -------
    Dict[str, Dict[str, float]]
        A dictionary with k-values as keys and dictionaries of precision and recall as\
              values,
        similar to `evaluate_queries`.

    Notes
    -----
    - The `database.search_queries()` method is expected to return a 2D array of\
          retrieved document indices, where each column corresponds to a query.
    - This function simply orchestrates the retrieval and passes the results to\
          `evaluate_queries` for metric computation.
    """

    # Extract query IDs and texts separately for database searching
    query_ids = list(queries.keys())
    query_texts = list(queries.values())

    # Retrieve the top documents for each query up to the maximum required k
    top_k_indices = database.search_queries(
        query_texts, k=max(k_values), **search_kwargs
    )

    # Evaluate results at specified k-values
    return evaluate_queries(query_ids, top_k_indices, query_results, k_values)


def evaluate_queries_in_chunked_docs(
    queries: Dict[QueryID, str],
    query_results: Mapping[QueryID, Sequence[DocID]],
    k_values: list[int],
    database: VectorDatabase,
    k_multiple: int = 10,
    **search_kwargs,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate retrieval performance when documents may be chunked or split into \
        smaller segments.

    Some databases or retrieval systems might store documents in multiple chunks.
    In such cases, simply retrieving top-k chunks may not be enough to ensure that
    all relevant documents are represented among the top results. Thus, we may need
    to retrieve more documents and then filter or deduplicate.

    Parameters
    ----------
    queries : Dict[QueryID, str]
        A dictionary mapping query IDs to their query text.
    query_results : Mapping[QueryID, Sequence[DocID]]
        Ground truth relevant documents for each query.
    k_values : list[int]
        The cutoff values at which to compute evaluation metrics.
    database : VectorDatabase
        An instance of a vector database that supports search over documents/chunks.
    k_multiple : int, default=10
        A factor by which to multiply the maximum k-value to attempt retrieving\
              more documents.
        If not enough unique documents are found, this value is repeatedly multiplied.
    **search_kwargs
        Additional parameters passed to the search method of the database.

    Returns
    -------
    Dict[str, Dict[str, float]]
        A dictionary with k-values as keys and a dictionary of metrics\
             (Precision, Recall) as values.

    Notes
    -----
    - If the initial retrieval does not yield enough unique documents to compute top-k\
          unique documents, the retrieval size is increased (scaled by `k_multiple`).
    - The `get_top_k_unique` function is expected to filter the retrieved results down \
        to the top-k unique documents.
    """

    query_ids = list(queries.keys())
    query_texts = list(queries.values())

    # Start with an initial retrieval size and scale up
    # if not enough unique docs are available
    k_max = max(k_values) * k_multiple
    not_enough = True
    while not_enough:
        try:
            # Retrieve a large set of candidate documents
            top_k_indices = database.search_queries(
                query_texts, k=k_max, **search_kwargs
            )

            # Deduplicate and select top-k unique documents from the candidate set
            top_k = np.apply_along_axis(
                lambda x: get_top_k_unique(x, k=10), axis=0, arr=top_k_indices
            )
            not_enough = False
        except Exception:
            # If not enough unique docs, increase k_max and try again
            k_max *= k_multiple
            print(f"Multiple insufficient, trying with {k_max / max(k_values)}")

    # Evaluate the final retrieved unique documents
    return evaluate_queries(query_ids, top_k, query_results, k_values)


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
