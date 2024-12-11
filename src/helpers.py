from typing import Dict, List, Union

import numpy as np
import pandas as pd

DocID = Union[int, str]
QueryID = Union[int, str]


def process_query_results(
    query_results: pd.DataFrame,
    doc_col_name: str = "doc_number",
) -> Dict[QueryID, List[DocID]]:
    """
    Processes the query results to extract relevant document IDs for each query.

    Args:
        query_results (pd.DataFrame): DataFrame containing the query results with query IDs as the index.
        doc_col_name (str, optional): The column name in the DataFrame that contains the document IDs. Defaults to "doc_number".

    Returns:
        Dict[QueryID, List[DocID]]: A dictionary where the keys are query IDs and the values are lists of relevant document IDs.
    """
    new_query_results = {}
    for q_id in query_results.index:
        relevant_docs = [query_results.loc[q_id][doc_col_name]]
        # Handle multiple relevant docs scenario
        if not isinstance(relevant_docs[0], np.int64):
            relevant_docs = list(relevant_docs[0])  # If multiple relevant docs
        new_query_results[q_id] = relevant_docs
    return new_query_results


def set_default_probas(n_neighbours: int) -> np.ndarray:
    """
    Calculate and return the default probability distribution for assigning neighbors
    across different levels in a hierarchical structure.

    Parameters:
    n_neighbours (int): The number of neighbors to consider at each level.

    Returns:
    np.ndarray: An array of probabilities for each level, normalized to sum to 1.
    """
    m_L = 1 / np.log(n_neighbours)
    nn = 0  # set nearest neighbors count = 0
    cum_nneighbor_per_level = []
    level = 0  # we start at level 0
    assign_probas = []
    while True:
        # calculate probability for current level
        proba = np.exp(-level / m_L) * (1 - np.exp(-1 / m_L))
        # once we reach low prob threshold, we've created enough levels
        if proba < 1e-9:
            break
        assign_probas.append(proba)
        # neighbors is == M on every level except level 0 where == M*2
        nn += n_neighbours * 2 if level == 0 else n_neighbours
        cum_nneighbor_per_level.append(nn)
        level += 1
    return np.array(assign_probas) / np.sum(assign_probas)


def assign_levels(n_neighbours: int, n_docs: int):
    """
    Assigns levels to a given number of documents based on the probabilities
    derived from the number of neighbors.

    Parameters:
    n_neighbours (int): The number of neighbors to consider for determining
                        the assignment probabilities.
    n_docs (int): The number of documents to assign levels to.

    Returns:
    numpy.ndarray: An array of assigned levels for each document.
    """
    assign_probas = set_default_probas(n_neighbours)
    chosen_levels = np.random.choice(
        np.arange(len(assign_probas)), size=n_docs, p=assign_probas
    )
    return chosen_levels
