from typing import Dict, List, Union

import numpy as np
import pandas as pd

DocID = Union[int, str]
QueryID = Union[int, str]


def process_query_results(
    query_results: pd.DataFrame,
    doc_col_name: str = "doc_number",
) -> Dict[QueryID, List[DocID]]:
    new_query_results = {}
    for q_id in query_results.index:
        relevant_docs = [query_results.loc[q_id][doc_col_name]]
        # Handle multiple relevant docs scenario
        if not isinstance(relevant_docs[0], np.int64):
            relevant_docs = list(relevant_docs[0])  # If multiple relevant docs
        new_query_results[q_id] = relevant_docs
    return new_query_results


def set_default_probas(n_neighbours: int) -> np.ndarray:
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
    assign_probas = set_default_probas(n_neighbours)
    chosen_levels = np.random.choice(
        np.arange(len(assign_probas)), size=n_docs, p=assign_probas
    )
    return chosen_levels
