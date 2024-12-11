from typing import Tuple

import numpy as np
from tqdm import tqdm


def search(
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
    batch_size: int = 1000,
    load_bar: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a search over document vectors to find the top k most similar documents \
        for each query vector.

    Args:
        doc_vectors (np.ndarray): A 2D array where each row represents a document \
            vector.
        query_vectors (np.ndarray): A 2D array where each row represents a query vector.
        k (int): The number of top similar documents to retrieve for each query.
        batch_size (int, optional): The number of document vectors to process in each \
            batch. Defaults to 1000.
        load_bar (bool, optional): Whether to display a progress bar during the \
            search. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - final_indices (np.ndarray): Indices of the top k most similar documents \
                for each query.
            - final_sims (np.ndarray): Similarity scores of the top k most similar \
                documents for each query.
    """

    # Number of documents
    n_docs = doc_vectors.shape[0]
    # Split document indices into batches
    doc_split = np.array_split(np.arange(n_docs), np.ceil(n_docs / batch_size))
    top_indices = []
    top_sims = []
    start_index = 0

    # Initialize progress bar if load_bar is True
    if load_bar:
        iterator = tqdm(
            doc_split, total=len(doc_split), desc="Searching document batches"
        )
    else:
        iterator = doc_split

    # Iterate over each batch of document indices
    for doc_indices in iterator:
        # Get the document vectors for the current batch
        doc_batch = doc_vectors[doc_indices]

        # If the accumulated results exceed the batch size, aggregate them
        if len(top_indices) * k >= batch_size:
            top_indices = np.concatenate(top_indices, axis=0)
            top_sims = np.concatenate(top_sims, axis=0)
            top_k_index, top_sim = aggregate(top_indices, top_sims, k)
            top_indices = [top_k_index]
            top_sims = [top_sim]

        # Compute similarity scores between document batch and query vectors
        sims = np.dot(doc_batch, query_vectors.T)

        # Get the indices of the top k most similar documents
        if sims.shape[0] > k:
            intermed_index = np.argpartition(-sims, k, axis=0)[:k]
        else:
            intermed_index = np.argsort(-sims, axis=0)[:k]

        # Get the similarity scores for the top k documents
        top_sim = sims[intermed_index, np.arange(sims.shape[1])]
        # Adjust indices to account for the current batch's starting index
        top_k_index = intermed_index + start_index
        start_index += doc_batch.shape[0]

        # Append the results for the current batch
        top_indices.append(top_k_index)
        top_sims.append(top_sim)
    else:
        # Aggregate results from all batches
        top_indices = np.concatenate(top_indices, axis=0)
        top_sims = np.concatenate(top_sims, axis=0)
        final_indices, final_sims = aggregate(top_indices, top_sims, k, sort=True)

    return final_indices, final_sims


def aggregate(indices, sims, k, sort=False):
    """
    Aggregate the top-k indices and their corresponding similarity scores.

    Parameters:
    indices (ndarray): An array of indices.
    sims (ndarray): An array of similarity scores.
    k (int): The number of top elements to select.
    sort (bool, optional): If True, sort the similarity scores to get the top-k \
        elements.
                           If False, use partitioning to get the unsorted top-k \
                           elements. Default is False.

    Returns:
    tuple: A tuple containing:
        - top_indices (ndarray): The top-k indices.
        - top_sims (ndarray): The top-k similarity scores.
    """
    # If the number of similarity scores is greater than k and sorting is not required,
    # use argpartition to get the indices of the top k elements. This is more efficient
    # than sorting the entire array.
    if sims.shape[0] > k and not sort:
        intermed_indices = np.argpartition(-sims, k, axis=0)[:k]
    else:
        # Otherwise, use argsort to get the indices of the top k elements.
        intermed_indices = np.argsort(-sims, axis=0)[:k]

    # Retrieve the top k similarity scores using the intermediate indices.
    top_sims = sims[intermed_indices, np.arange(sims.shape[1])]

    # Retrieve the corresponding indices of the top k similarity scores.
    top_indices = indices[intermed_indices, np.arange(indices.shape[1])]

    return top_indices, top_sims
