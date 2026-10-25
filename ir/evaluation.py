"""
Evaluation (IR Module 4)
------------------------
Given a query, the set of recipes a human judged RELEVANT (ground truth),
and the ranked list a model returned:

  Precision@k = (relevant among the top k) / k           "how clean is the top of the list?"
  Recall@k    = (relevant among the top k) / |relevant|  "how much of what exists did we find?"
  Average Precision (AP) = mean of Precision@i taken at every rank i where a
                           relevant document appears (divided by |relevant|)
                           → rewards putting relevant documents EARLY
  MAP = mean of AP over all test queries → one number per model

Precision and recall need judgements, so data/eval_queries.json holds
~20 queries with their relevant doc_ids, labelled by the group.
"""


def precision_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    top = ranked[:k]
    return sum(1 for d in top if d in relevant) / k if k else 0.0


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def average_precision(ranked: list[int], relevant: set[int]) -> float:
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            hits += 1
            total += hits / i                  # Precision@i at this relevant hit
    return total / len(relevant)


def evaluate_run(runs: dict[str, list[int]], judgements: dict[str, set[int]], k: int = 10) -> dict:
    """
    runs: query -> ranked doc_ids from one model
    judgements: query -> relevant doc_ids
    Returns per-query rows and the averages (MAP etc.).
    """
    rows = []
    for q, relevant in judgements.items():
        ranked = runs.get(q, [])
        rows.append({
            "query": q,
            "relevant": len(relevant),
            "precision_at_k": round(precision_at_k(ranked, relevant, k), 3),
            "recall_at_k": round(recall_at_k(ranked, relevant, k), 3),
            "average_precision": round(average_precision(ranked, relevant), 3),
        })
    n = len(rows) or 1
    return {
        "k": k,
        "rows": rows,
        "mean_precision_at_k": round(sum(r["precision_at_k"] for r in rows) / n, 3),
        "mean_recall_at_k": round(sum(r["recall_at_k"] for r in rows) / n, 3),
        "map": round(sum(r["average_precision"] for r in rows) / n, 3),
    }
