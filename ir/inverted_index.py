"""
Inverted Index (IR Module 1)
----------------------------
Instead of scanning every recipe for every query (slow), we build a
dictionary that goes from TERM -> the documents that contain it:

    postings["paneer"] = {12: 3, 87: 1, 401: 2}   # doc_id -> term frequency

So answering "which recipes mention paneer?" is one dictionary lookup.
We also keep, for every term, the DOCUMENT FREQUENCY df (how many documents
contain it) because IDF needs it, and for every document its length and
maximum term frequency because TF normalisation and BM25 need them.
"""

import os
import pickle
from collections import Counter, defaultdict

from ir.preprocess import preprocess

INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "index")
INDEX_PATH = os.path.join(INDEX_DIR, "inverted_index.pkl")


class InvertedIndex:
    def __init__(self):
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.df: dict[str, int] = {}
        self.doc_len: dict[int, int] = {}      # number of tokens in each document
        self.doc_max_tf: dict[int, int] = {}   # highest term frequency in each document
        self.N: int = 0
        self.avg_doc_len: float = 0.0

    # ---------------------------------------------------------------- build
    def add_document(self, doc_id: int, text: str) -> None:
        tokens = preprocess(text)
        counts = Counter(tokens)
        for term, tf in counts.items():
            self.postings[term][doc_id] = tf
        self.doc_len[doc_id] = len(tokens)
        self.doc_max_tf[doc_id] = max(counts.values()) if counts else 1

    def finalize(self) -> None:
        self.N = len(self.doc_len)
        self.df = {term: len(docs) for term, docs in self.postings.items()}
        self.avg_doc_len = sum(self.doc_len.values()) / self.N if self.N else 0.0

    # ---------------------------------------------------------------- query
    def get_postings(self, term: str) -> dict[int, int]:
        return self.postings.get(term, {})

    def stats(self) -> dict:
        top = sorted(self.df.items(), key=lambda kv: kv[1], reverse=True)[:20]
        return {
            "documents": self.N,
            "vocabulary_size": len(self.postings),
            "avg_doc_length": round(self.avg_doc_len, 1),
            "total_postings": sum(self.df.values()),
            "most_common_terms": [{"term": t, "df": d} for t, d in top],
        }

    # ---------------------------------------------------------------- disk
    def save(self, path: str = INDEX_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str = INDEX_PATH) -> "InvertedIndex":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_index(recipes) -> InvertedIndex:
    index = InvertedIndex()
    for r in recipes:
        index.add_document(r.doc_id, r.indexed_text)
    index.finalize()
    return index
