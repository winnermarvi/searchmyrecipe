"""
Query Expansion (IR Module 3 — query operations)
------------------------------------------------
A user types "biryani"; a great recipe may be indexed under "basmati",
"saffron", "dum". Pure keyword matching misses it (the VOCABULARY MISMATCH
problem). Query expansion adds related terms to the query, with a smaller
weight than the typed terms so they help but do not take over.

Two strategies, both explainable:

1. THESAURUS expansion (manual, controlled vocabulary)
   data/thesaurus.json → "expansion": {"paneer": ["cottage", "cheese"], ...}
   A human wrote these relations. Precise, but only covers what was written.

2. AUTOMATIC GLOBAL ANALYSIS (corpus-driven, computed once at index time)
   For every pair of terms (u, v) count how many recipes contain BOTH:
       c(u,v)
   and normalise it so that very common terms do not dominate:
       assoc(u,v) = c(u,v) / (n_u + n_v − c(u,v))          (Jaccard)
   i.e. "of all recipes that mention u or v, what fraction mention both?".
   For each term we keep its strongest associations. "Global" because the
   whole collection is analysed, not just the documents retrieved for one
   query (that would be local analysis).
"""

import json
import os
import pickle

import numpy as np

from ir.inverted_index import INDEX_DIR, InvertedIndex
from ir.preprocess import preprocess

_DATA_DIR = os.path.dirname(INDEX_DIR)
COOC_PATH = os.path.join(INDEX_DIR, "cooccurrence.pkl")
THESAURUS_PATH = os.path.join(_DATA_DIR, "thesaurus.json")

THESAURUS_WEIGHT = 0.5      # weight of an added thesaurus term (typed term = 1.0)
COOC_WEIGHT = 0.4           # weight of an added co-occurrence term
COOC_PER_TERM = 2           # how many associated terms to add per query term


# --------------------------------------------------------------------------
# Build step (called from scripts/build_index.py)
# --------------------------------------------------------------------------
def build_cooccurrence(index: InvertedIndex, min_df: int = 15, max_df_ratio: float = 0.25,
                       max_terms: int = 2500, keep: int = 8) -> dict[str, list[tuple[str, float]]]:
    """
    Returns {term: [(associated_term, jaccard), ...]} for the informative part
    of the vocabulary: terms that are neither too rare (noise) nor too common
    (they co-occur with everything — "salt", "oil").
    """
    N = index.N
    vocab = [t for t, df in index.df.items() if min_df <= df <= max_df_ratio * N]
    vocab = sorted(vocab, key=lambda t: index.df[t], reverse=True)[:max_terms]
    pos = {t: i for i, t in enumerate(vocab)}
    docs = sorted(index.doc_len)
    row = {d: i for i, d in enumerate(docs)}

    # binary document-term matrix M (N x |vocab|); M.T @ M gives c(u,v)
    M = np.zeros((N, len(vocab)), dtype=np.float32)
    for t, i in pos.items():
        for d in index.postings[t]:
            M[row[d], i] = 1.0
    C = M.T @ M                                  # co-occurrence counts
    n = np.diag(C).copy()                        # n_u on the diagonal
    denom = n[:, None] + n[None, :] - C
    J = np.where(denom > 0, C / np.maximum(denom, 1), 0.0)
    np.fill_diagonal(J, 0.0)

    table: dict[str, list[tuple[str, float]]] = {}
    for t, i in pos.items():
        best = np.argsort(-J[i])[:keep]
        table[t] = [(vocab[j], round(float(J[i, j]), 4)) for j in best if J[i, j] > 0]
    return table


def save_cooccurrence(table: dict, path: str = COOC_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(table, f)


# --------------------------------------------------------------------------
# Query-time expansion
# --------------------------------------------------------------------------
class QueryExpander:
    def __init__(self, cooc_path: str = COOC_PATH, thesaurus_path: str = THESAURUS_PATH):
        with open(thesaurus_path, encoding="utf-8") as f:
            data = json.load(f)
        self.thesaurus: dict[str, list[str]] = data.get("expansion", {})
        self.canonical: dict[str, str] = data.get("canonical", {})
        self.cooc: dict[str, list[tuple[str, float]]] = {}
        if os.path.exists(cooc_path):
            with open(cooc_path, "rb") as f:
                self.cooc = pickle.load(f)

    def expand(self, tokens: list[str], use_thesaurus: bool = True,
               use_cooc: bool = True) -> tuple[dict[str, float], list[dict]]:
        """
        tokens: the preprocessed query terms.
        Returns (weighted query {term: weight}, list of added terms for the UI).
        A term already typed by the user is never re-added; if two strategies
        propose the same term, the higher weight wins.
        """
        query: dict[str, float] = {t: 1.0 for t in tokens}
        added: dict[str, dict] = {}

        for t in tokens:
            if use_thesaurus:
                for raw in self.thesaurus.get(t, []):
                    for term in preprocess(raw) or [raw]:
                        self._propose(query, added, term, THESAURUS_WEIGHT, "thesaurus", t)
            if use_cooc:
                for term, j in self.cooc.get(t, [])[:COOC_PER_TERM]:
                    self._propose(query, added, term, COOC_WEIGHT, "co-occurrence", t, j)

        for term, info in added.items():
            query[term] = info["weight"]
        return query, sorted(added.values(), key=lambda x: -x["weight"])

    @staticmethod
    def _propose(query, added, term, weight, source, origin, assoc=None):
        if term in query and query[term] == 1.0:
            return
        if term not in added or added[term]["weight"] < weight:
            added[term] = {"term": term, "weight": weight, "source": source,
                           "from": origin, "assoc": assoc}

    def associations(self, term: str) -> list[dict]:
        """For the Index explorer: strongest global associations of a term."""
        return [{"term": t, "jaccard": j} for t, j in self.cooc.get(term, [])]
