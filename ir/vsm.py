"""
Vector Space Model (IR Module 2)
--------------------------------
Every document and the query become vectors in "term space" (one axis per
vocabulary term). The weight on each axis is TF-IDF:

    tf(i,j)  = f(i,j) / max_l f(l,j)          normalised term frequency
    idf(i)   = log10( N / n_i )               rarity of the term in the collection
    w(i,j)   = tf(i,j) * idf(i)               document weight
    w(i,q)   = (0.5 + 0.5 * f(i,q)/max_l f(l,q)) * idf(i)     query weight

Ranking uses the cosine of the angle between the two vectors:

    sim(d,q) = (d . q) / ( |d| * |q| )

Cosine ignores vector length, so a long recipe is not favoured just for
having more words. Only documents that share at least one term with the
query can have a non-zero score, so we score just those (accumulator
pattern) — that is what the inverted index buys us.
"""

import math
from collections import Counter

from ir.inverted_index import InvertedIndex


class VectorSpaceModel:
    def __init__(self, index: InvertedIndex):
        self.index = index
        self.idf: dict[str, float] = {
            term: math.log10(index.N / df) for term, df in index.df.items()
        }
        self.doc_norm: dict[int, float] = self._compute_doc_norms()

    # ---------------------------------------------------------------- weights
    def doc_weight(self, term: str, doc_id: int) -> float:
        """w(i,j) = tf * idf for one term in one document."""
        f = self.index.postings.get(term, {}).get(doc_id, 0)
        if f == 0:
            return 0.0
        tf = f / self.index.doc_max_tf[doc_id]
        return tf * self.idf[term]

    def query_weights(self, query_terms: dict[str, float]) -> dict[str, float]:
        """
        query_terms: term -> raw weight (1.0 for a typed term; <1 for an
        expanded term). Returns the TF-IDF query vector, ignoring terms that
        are not in the vocabulary (they cannot match anything).
        """
        known = {t: f for t, f in query_terms.items() if t in self.idf}
        if not known:
            return {}
        max_f = max(known.values())
        return {t: (0.5 + 0.5 * f / max_f) * self.idf[t] for t, f in known.items()}

    def doc_vector(self, doc_id: int) -> dict[str, float]:
        """The full TF-IDF vector of one document (needed by Rocchio)."""
        max_tf = self.index.doc_max_tf[doc_id]
        return {t: (f / max_tf) * self.idf[t]
                for t, docs in self.index.postings.items() if (f := docs.get(doc_id, 0))}

    def _compute_doc_norms(self) -> dict[int, float]:
        sq = Counter()
        for term, docs in self.index.postings.items():
            idf = self.idf[term]
            for doc_id, f in docs.items():
                w = (f / self.index.doc_max_tf[doc_id]) * idf
                sq[doc_id] += w * w
        return {d: math.sqrt(s) for d, s in sq.items()}

    # ---------------------------------------------------------------- ranking
    def rank(self, query_terms: dict[str, float], candidates: set[int] | None = None,
             top_k: int = 20) -> list[tuple[int, float]]:
        """Return [(doc_id, cosine_score), ...] sorted best first."""
        return self.rank_vector(self.query_weights(query_terms), candidates, top_k)

    def rank_vector(self, q: dict[str, float], candidates: set[int] | None = None,
                    top_k: int = 20) -> list[tuple[int, float]]:
        """Rank against an already-weighted query vector (used after Rocchio)."""
        q = {t: w for t, w in q.items() if t in self.idf and w > 0}
        if not q:
            return []
        q_norm = math.sqrt(sum(w * w for w in q.values()))

        dot = Counter()                       # accumulator: doc_id -> d.q
        for term, wq in q.items():
            for doc_id, f in self.index.postings[term].items():
                if candidates is not None and doc_id not in candidates:
                    continue
                wd = (f / self.index.doc_max_tf[doc_id]) * self.idf[term]
                dot[doc_id] += wd * wq

        scored = [(d, s / (self.doc_norm[d] * q_norm)) for d, s in dot.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def explain(self, query_terms: dict[str, float], doc_id: int) -> dict:
        """Per-term breakdown used by the 'Why this result?' panel."""
        return self.explain_vector(self.query_weights(query_terms), doc_id)

    def explain_vector(self, q: dict[str, float], doc_id: int) -> dict:
        q = {t: w for t, w in q.items() if t in self.idf and w > 0}
        rows, dot = [], 0.0
        for term, wq in q.items():
            f = self.index.postings.get(term, {}).get(doc_id, 0)
            wd = self.doc_weight(term, doc_id)
            contribution = wd * wq
            dot += contribution
            rows.append({
                "term": term, "tf_raw": f, "df": self.index.df.get(term, 0),
                "idf": round(self.idf[term], 4), "w_doc": round(wd, 4),
                "w_query": round(wq, 4), "contribution": round(contribution, 4),
            })
        rows.sort(key=lambda r: r["contribution"], reverse=True)
        q_norm = math.sqrt(sum(w * w for w in q.values()))
        d_norm = self.doc_norm.get(doc_id, 0.0)
        cosine = dot / (d_norm * q_norm) if d_norm and q_norm else 0.0
        return {
            "model": "VSM",
            "columns": ["term", "tf_raw", "df", "idf", "w_doc", "w_query", "contribution"],
            "rows": rows,
            "footer": (f"dot product = Σ contributions = {dot:.4f}; |d| = {d_norm:.4f}, "
                       f"|q| = {q_norm:.4f} ⇒ cosine = {dot:.4f} / ({d_norm:.4f} × {q_norm:.4f}) = {cosine:.4f}"),
            "score": round(cosine, 4),
        }
