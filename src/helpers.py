import re
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
        query_results (pd.DataFrame): DataFrame containing the query results with query\
              IDs as the index.
        doc_col_name (str, optional): The column name in the DataFrame that contains\
              the document IDs. Defaults to "doc_number".

    Returns:
        Dict[QueryID, List[DocID]]: A dictionary where the keys are query IDs and the\
              values are lists of relevant document IDs.
    """
    new_query_results = {}
    for q_id in query_results.index:
        relevant_docs = [query_results.loc[q_id][doc_col_name]]
        # Handle multiple relevant docs scenario
        if not isinstance(relevant_docs[0], np.int64):
            relevant_docs = list(relevant_docs[0])  # If multiple relevant docs
        new_query_results[q_id] = relevant_docs
    return new_query_results


def documents_chunker(docs, tokenizer, max_length, stride):
    for doc_id, file_path in docs.items():
        with open(file_path, "r") as f:
            text = f.read()
            # cleaned = re.sub(r"[^a-zA-Z\s]", "", text)
            cleaned = text
            tokens = tokenizer.tokenize(cleaned, truncation=False, verbose=False)

            # Yield segments with proper stride handling
            for i in range(0, len(tokens) - stride, max_length - stride):
                yield doc_id, re.sub(" ##", "", " ".join(tokens[i : i + max_length]))


def get_top_k_unique(col_data: np.ndarray, k: int):
    seen = set()
    unique_vals = []

    for val in col_data:
        if val not in seen:
            seen.add(val)
            unique_vals.append(val)
            if len(unique_vals) == k:
                return unique_vals
