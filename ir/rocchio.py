"""
Rocchio Relevance Feedback (IR Module 3)
----------------------------------------
After seeing results, the user marks some as relevant (👍) and some as
non-relevant (👎). Rocchio moves the query vector TOWARDS the relevant
documents and AWAY from the non-relevant ones:

    q_new = α·q  +  (β / |Dr|) · Σ_{d ∈ Dr} d  −  (γ / |Dn|) · Σ_{d ∈ Dn} d

  α = 1.00  keep the original query
  β = 0.75  pull towards relevant documents (strong: positive evidence is reliable)
  γ = 0.15  push away from non-relevant ones (weak: there are many ways to be irrelevant)

All vectors are TF-IDF vectors in the same term space. Terms whose weight
turns negative are dropped (a query cannot "anti-contain" a term) and only
the strongest terms are kept so the query stays focused and fast.
This is BOTH query expansion (new terms arrive from relevant documents) and
term re-weighting (existing terms change weight).
"""

from collections import defaultdict

ALPHA, BETA, GAMMA = 1.0, 0.75, 0.15
MAX_TERMS = 30


def rocchio(query: dict[str, float], relevant: list[dict[str, float]],
            nonrelevant: list[dict[str, float]], alpha: float = ALPHA,
            beta: float = BETA, gamma: float = GAMMA, max_terms: int = MAX_TERMS) -> dict[str, float]:
    new: dict[str, float] = defaultdict(float)
    for t, w in query.items():
        new[t] += alpha * w
    for d in relevant:
        for t, w in d.items():
            new[t] += beta * w / len(relevant)
    for d in nonrelevant:
        for t, w in d.items():
            new[t] -= gamma * w / len(nonrelevant)
    positive = {t: w for t, w in new.items() if w > 0}
    top = sorted(positive.items(), key=lambda kv: kv[1], reverse=True)[:max_terms]
    return dict(top)


def diff(old: dict[str, float], new: dict[str, float], top: int = 12) -> dict:
    """What changed, for the UI: terms added, removed, strengthened, weakened."""
    added = sorted(((t, w) for t, w in new.items() if t not in old), key=lambda x: -x[1])[:top]
    removed = sorted(t for t in old if t not in new)[:top]
    changed = [(t, old[t], new[t]) for t in new if t in old]
    up = sorted((c for c in changed if c[2] > c[1] * 1.05), key=lambda x: -(x[2] - x[1]))[:top]
    down = sorted((c for c in changed if c[2] < c[1] * 0.95), key=lambda x: x[2] - x[1])[:top]
    r = lambda x: round(x, 3)
    return {
        "added": [{"term": t, "weight": r(w)} for t, w in added],
        "removed": removed,
        "strengthened": [{"term": t, "old": r(a), "new": r(b)} for t, a, b in up],
        "weakened": [{"term": t, "old": r(a), "new": r(b)} for t, a, b in down],
    }
