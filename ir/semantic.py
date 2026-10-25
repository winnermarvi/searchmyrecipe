"""
Semantic Layer — sentence embeddings
------------------------------------
TF-IDF / BM25 can only match a query to a recipe if they SHARE WORDS.
Query expansion narrows that gap; embeddings close it.

MiniLM (all-MiniLM-L6-v2) is a small pre-trained neural network that turns
any piece of text into a vector of 384 numbers such that texts with similar
MEANING end up close together. "creamy rich dish for guests" lands near
"Shahi Paneer with cashew cream gravy" although they share no word.

We still rank with COSINE SIMILARITY — the same formula as the Vector Space
Model — only the vectors come from the neural model instead of term counts.
Because every vector is L2-normalised (length 1), cosine = dot product.

Recipes are encoded ONCE at index time and cached to disk (~9 MB); at query
time only the query is encoded (a few milliseconds on CPU).
"""

import json
import os

import numpy as np

from ir.inverted_index import INDEX_DIR

MODEL_NAME = "all-MiniLM-L6-v2"
EMB_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
IDS_PATH = os.path.join(INDEX_DIR, "doc_ids.json")


def recipe_text(r) -> str:
    """What the encoder sees: title, ingredients and the start of the method."""
    return f"{r.name}. Ingredients: {', '.join(r.ingredients)}. {r.instructions[:300]}"


def build_embeddings(recipes, incremental: bool = True) -> str:
    """Encode all recipes, or — when the cached ids are a prefix of the new
    list (recipes were only appended) — encode just the new ones."""
    from sentence_transformers import SentenceTransformer
    ids = [r.doc_id for r in recipes]
    old_emb, old_ids = None, []
    if incremental and os.path.exists(EMB_PATH) and os.path.exists(IDS_PATH):
        with open(IDS_PATH) as f:
            old_ids = json.load(f)
        if ids[:len(old_ids)] == old_ids and len(old_ids) <= len(ids):
            old_emb = np.load(EMB_PATH)
        else:
            old_ids = []
    todo = recipes[len(old_ids):]
    if not todo:
        return "unchanged"
    model = SentenceTransformer(MODEL_NAME)
    new = model.encode([recipe_text(r) for r in todo], batch_size=64,
                       normalize_embeddings=True, show_progress_bar=len(todo) > 50).astype(np.float32)
    emb = np.vstack([old_emb, new]) if old_emb is not None else new
    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(EMB_PATH, emb)
    with open(IDS_PATH, "w") as f:
        json.dump(ids, f)
    return f"encoded {len(todo)} new" if old_emb is not None else f"encoded all {len(todo)}"


class SemanticModel:
    def __init__(self):
        self.available = os.path.exists(EMB_PATH) and os.path.exists(IDS_PATH)
        if not self.available:
            return
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(MODEL_NAME)
        self.emb: np.ndarray = np.load(EMB_PATH)                 # (N, 384), unit length
        with open(IDS_PATH) as f:
            self.doc_ids: list[int] = json.load(f)
        self.row = {d: i for i, d in enumerate(self.doc_ids)}

    def encode(self, text: str) -> np.ndarray:
        return self.encoder.encode([text], normalize_embeddings=True)[0]

    def doc_embedding(self, doc_id: int) -> np.ndarray:
        return self.emb[self.row[doc_id]]

    def scores(self, q_vec: np.ndarray, candidates: set[int] | None = None,
               top_k: int = 100) -> dict[int, float]:
        """Cosine (= dot product) between the query vector and every recipe."""
        sims = self.emb @ q_vec                                  # one matrix-vector product
        order = np.argsort(-sims)
        out: dict[int, float] = {}
        for i in order:
            d = self.doc_ids[i]
            if candidates is not None and d not in candidates:
                continue
            out[d] = float(sims[i])
            if len(out) >= top_k:
                break
        return out

    def similarity(self, q_vec: np.ndarray, doc_id: int) -> float:
        return float(self.emb[self.row[doc_id]] @ q_vec)
