# src/evaluation.py
def evaluate_precision_recall(results, ground_truth, k_values):
    precision_at_k = {}
    recall_at_k = {}
    
    for k in k_values:
        precisions = []
        recalls = []
        for query_id, retrieved_docs in results.items():
            relevant_docs = set(ground_truth[query_id])
            retrieved_at_k = [doc[0] for doc in retrieved_docs[:k]]

            precision = len(set(retrieved_at_k) & relevant_docs) / k
            recall = len(set(retrieved_at_k) & relevant_docs) / len(relevant_docs)

            precisions.append(precision)
            recalls.append(recall)

        precision_at_k[k] = sum(precisions) / len(precisions)
        recall_at_k[k] = sum(recalls) / len(recalls)

    return precision_at_k, recall_at_k