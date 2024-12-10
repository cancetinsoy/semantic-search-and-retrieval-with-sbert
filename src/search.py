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

    n_docs = doc_vectors.shape[0]
    doc_split = np.array_split(np.arange(n_docs), np.ceil(n_docs / batch_size))
    top_indices = []
    top_sims = []
    start_index = 0

    if load_bar:
        iterator = tqdm(
            doc_split, total=len(doc_split), desc="Searching document batches"
        )
    else:
        iterator = doc_split

    for doc_indices in iterator:
        doc_batch = doc_vectors[doc_indices]
        if len(top_indices) * k >= batch_size:
            top_indices = np.concatenate(top_indices, axis=0)
            top_sims = np.concatenate(top_sims, axis=0)
            top_k_index, top_sim = aggregate(top_indices, top_sims, k)
            top_indices = [top_k_index]
            top_sims = [top_sim]
        sims = np.dot(doc_batch, query_vectors.T)

        if sims.shape[0] > k:
            intermed_index = np.argpartition(-sims, k, axis=0)[:k]
        else:
            intermed_index = np.argsort(-sims, axis=0)[:k]
        top_sim = sims[intermed_index, np.arange(sims.shape[1])]
        top_k_index = intermed_index + start_index
        start_index += doc_batch.shape[0]

        top_indices.append(top_k_index)
        top_sims.append(top_sim)
    else:
        top_indices = np.concatenate(top_indices, axis=0)
        top_sims = np.concatenate(top_sims, axis=0)
        final_indices, final_sims = aggregate(top_indices, top_sims, k)

    return final_indices, final_sims


def aggregate(indices, sims, k):
    if sims.shape[0] > k:
        intermed_indices = np.argpartition(-sims, k, axis=0)[:k]
    else:
        intermed_indices = np.argsort(-sims, axis=0)[:k]
    top_sims = sims[intermed_indices, np.arange(sims.shape[1])]
    top_indices = indices[intermed_indices, np.arange(indices.shape[1])]
    return top_indices, top_sims
