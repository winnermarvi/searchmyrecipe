"""
Result Merging / Score Fusion
-----------------------------
Two rankers give two lists with scores on DIFFERENT scales:
  BM25   → unbounded, e.g. 0 … 20
  MiniLM → cosine, e.g. 0.2 … 0.8
Adding them raw would let BM25 drown the semantic signal, so first each list
is MIN-MAX NORMALISED to [0, 1]:

    s' = (s − min) / (max − min)

then the lists are combined by a weighted sum:

    fused(d) = w_lex · lex'(d) + (1 − w_lex) · sem'(d)

A document missing from one list gets 0 for that list. This is exactly the
score-normalisation approach to result merging (data fusion) that a
distributed-IR broker uses when merging lists from different collections.
"""


def min_max(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {d: 1.0 for d in scores}
    return {d: (s - lo) / (hi - lo) for d, s in scores.items()}


def fuse(lexical: dict[int, float], semantic: dict[int, float],
         w_lex: float = 0.6, top_k: int = 20) -> list[tuple[int, float, float, float]]:
    """Return [(doc_id, fused, lex_norm, sem_norm)] best first."""
    lex_n, sem_n = min_max(lexical), min_max(semantic)
    out = []
    for d in set(lex_n) | set(sem_n):
        l, s = lex_n.get(d, 0.0), sem_n.get(d, 0.0)
        out.append((d, w_lex * l + (1 - w_lex) * s, l, s))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_k]
