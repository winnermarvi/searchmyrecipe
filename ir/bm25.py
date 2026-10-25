"""
BM25 (IR Module 2 — probabilistic family)
-----------------------------------------
BM25 is the practical descendant of the probabilistic model. Think of it as
TF-IDF with two fixes:

  1. Term-frequency SATURATION: the 5th occurrence of "paneer" should add
     less than the 1st. Controlled by k1 (1.2–2.0; we use 1.5).
  2. Document LENGTH normalisation: a term in a short recipe is stronger
     evidence than the same term in a very long one. Controlled by b
     (0 = ignore length, 1 = full normalisation; we use 0.75).

    score(d,q) = Σ_t  idf(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1 − b + b·|d|/avgdl) )

    idf(t) = ln( (N − n_t + 0.5) / (n_t + 0.5) + 1 )      (Robertson–Spärck Jones form)

Scores are not bounded to [0,1] like cosine — they are only meaningful for
comparing documents against the SAME query.
"""

import math
from collections import Counter

from ir.inverted_index import InvertedIndex


class BM25:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1, self.b = k1, b
        N = index.N
        self.idf: dict[str, float] = {
            t: math.log((N - n + 0.5) / (n + 0.5) + 1) for t, n in index.df.items()
        }

    def term_score(self, term: str, doc_id: int, query_weight: float = 1.0) -> float:
        f = self.index.postings.get(term, {}).get(doc_id, 0)
        if f == 0 or term not in self.idf:
            return 0.0
        norm = 1 - self.b + self.b * self.index.doc_len[doc_id] / self.index.avg_doc_len
        return query_weight * self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * norm)

    def score(self, query_terms: dict[str, float], doc_id: int) -> float:
        """Full BM25 score of one document (used when fusing lists)."""
        return sum(self.term_score(t, doc_id, w) for t, w in query_terms.items())

    def rank(self, query_terms: dict[str, float], candidates: set[int] | None = None,
             top_k: int = 20) -> list[tuple[int, float]]:
        acc = Counter()
        for term, wq in query_terms.items():
            if term not in self.idf:
                continue
            for doc_id in self.index.postings[term]:
                if candidates is not None and doc_id not in candidates:
                    continue
                acc[doc_id] += self.term_score(term, doc_id, wq)
        scored = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]

    def explain(self, query_terms: dict[str, float], doc_id: int) -> dict:
        rows, total = [], 0.0
        dl = self.index.doc_len[doc_id]
        for term, wq in query_terms.items():
            if term not in self.idf:
                continue
            f = self.index.postings[term].get(doc_id, 0)
            s = self.term_score(term, doc_id, wq)
            total += s
            rows.append({"term": term, "tf": f, "df": self.index.df[term],
                         "idf": round(self.idf[term], 4), "w_query": round(wq, 2),
                         "contribution": round(s, 4)})
        rows.sort(key=lambda r: r["contribution"], reverse=True)
        return {
            "model": "BM25",
            "columns": ["term", "tf", "df", "idf", "w_query", "contribution"],
            "rows": rows,
            "footer": (f"|d| = {dl} tokens, avgdl = {self.index.avg_doc_len:.1f}, "
                       f"k1 = {self.k1}, b = {self.b} ⇒ score = Σ contributions = {total:.4f}"),
            "score": round(total, 4),
        }
