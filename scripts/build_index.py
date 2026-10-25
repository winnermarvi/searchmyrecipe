"""
Run this once (and again whenever recipes change):

    python scripts/build_index.py

It loads the corpus, builds the inverted index and saves both to data/index/.
"""

import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.corpus import load_corpus
from ir.inverted_index import INDEX_DIR, build_index
from ir.query_expansion import build_cooccurrence, save_cooccurrence
from ir.semantic import build_embeddings

CORPUS_PATH = os.path.join(INDEX_DIR, "corpus.pkl")


def build_all(force_embeddings: bool = False, log=print) -> None:
    """Rebuild everything in data/index/. Also called by the API after a recipe is added."""
    t0 = time.time()
    recipes = load_corpus()
    log(f"Loaded {len(recipes)} recipes ({sum(r.source == 'custom' for r in recipes)} custom)")

    index = build_index(recipes)
    index.save()
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(CORPUS_PATH, "wb") as f:
        pickle.dump(recipes, f)

    cooc = build_cooccurrence(index)
    save_cooccurrence(cooc)
    log(f"Co-occurrence table built for {len(cooc)} terms (global analysis)")

    try:
        status = build_embeddings(recipes, incremental=not force_embeddings)
        log(f"MiniLM embeddings: {status}")
    except ImportError:
        log("sentence-transformers not installed — semantic layer skipped (Hybrid model disabled)")

    s = index.stats()
    log(f"Indexed {s['documents']} documents, vocabulary = {s['vocabulary_size']} terms, "
        f"avg doc length = {s['avg_doc_length']} tokens, {time.time() - t0:.1f}s")


def main() -> None:
    build_all(force_embeddings="--force-embeddings" in sys.argv)


if __name__ == "__main__":
    main()
