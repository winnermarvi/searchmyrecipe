"""
SearchMyRecipe — FastAPI backend
Run:   uvicorn app.main:app --reload
Docs:  http://127.0.0.1:8000/docs
"""

import json
import os
import pickle
import sys
import threading
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.bm25 import BM25
from ir.boolean_model import BooleanModel
from ir.inverted_index import INDEX_DIR, InvertedIndex
from ir.preprocess import CANONICAL, preprocess
from ir.query_expansion import QueryExpander
from ir.rocchio import ALPHA, BETA, GAMMA, diff, rocchio
from ir.semantic import SemanticModel
from ir.fusion import fuse
from ir.evaluation import evaluate_run
from ir.pantry import PantrySearch
from ir.visuals import tile
from ir.corpus import add_custom_recipe, classify_diet
import numpy as np
from ir.vsm import VectorSpaceModel

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="SearchMyRecipe", version="0.6 (Phase 6)")


# --------------------------------------------------------------------------
# Engine: index + models, loaded once at start-up
# --------------------------------------------------------------------------
class Engine:
    def __init__(self):
        self.index = InvertedIndex.load()
        with open(os.path.join(INDEX_DIR, "corpus.pkl"), "rb") as f:
            recipes = pickle.load(f)
        self.recipes = {r.doc_id: r for r in recipes}
        self.vsm = VectorSpaceModel(self.index)
        self.bm25 = BM25(self.index)
        self.boolean = BooleanModel(self.index)
        self.expander = QueryExpander()
        self.semantic = SemanticModel()          # .available == False if embeddings were not built
        self.pantry = PantrySearch(recipes)
        # canonical term -> its raw spellings, so highlighting works on "yogurt" when the term is "curd"
        self.variants: dict[str, list[str]] = defaultdict(list)
        for raw, canon in CANONICAL.items():
            self.variants[canon].append(raw)
        # Metadata "postings": diet/cuisine -> set of doc_ids. Same idea as the
        # inverted index, so filters are plain set intersections.
        self.by_diet: dict[str, set[int]] = defaultdict(set)
        self.by_cuisine: dict[str, set[int]] = defaultdict(set)
        for r in recipes:
            self.by_diet[r.diet].add(r.doc_id)
            self.by_cuisine[r.cuisine].add(r.doc_id)
        self.cuisines = sorted(self.by_cuisine)


engine = Engine()

MODEL_LABELS = {
    "boolean": "Boolean model (unranked set)",
    "vsm": "Vector Space Model (TF-IDF · cosine)",
    "bm25": "BM25 (probabilistic)",
    "hybrid": "Hybrid: BM25 + MiniLM embeddings, fused",
}
if not engine.semantic.available:
    MODEL_LABELS.pop("hybrid")

FUSION_POOL = 100        # how many candidates each side contributes before fusing


def _hybrid(query_terms: dict[str, float], q_emb: np.ndarray, candidates, w_lex: float, top_k: int):
    lex = dict(engine.bm25.rank(query_terms, candidates=candidates, top_k=FUSION_POOL))
    sem = engine.semantic.scores(q_emb, candidates=candidates, top_k=FUSION_POOL)
    # score every pooled document under BOTH models, so min-max compares like with like
    for d in set(lex) | set(sem):
        lex.setdefault(d, engine.bm25.score(query_terms, d))
        sem.setdefault(d, engine.semantic.similarity(q_emb, d))
    fused = fuse(lex, sem, w_lex=w_lex, top_k=top_k)
    cards = []
    for d, f, ln, sn in fused:
        c = _card(d, f, set(query_terms))
        c["components"] = {"lexical": round(lex.get(d, 0.0), 4), "semantic": round(sem.get(d, engine.semantic.similarity(q_emb, d)), 4),
                           "lexical_norm": round(ln, 4), "semantic_norm": round(sn, 4), "w_lex": w_lex}
        cards.append(c)
    return cards


def _card(doc_id: int, score: float, query_terms: set[str]) -> dict:
    r = engine.recipes[doc_id]
    d = r.to_dict()
    d["score"] = round(score, 4)
    d["instructions"] = r.instructions[:400] + ("…" if len(r.instructions) > 400 else "")
    d["tile"] = tile(r.name, r.ingredients)
    d["matched_terms"] = sorted(t for t in query_terms if doc_id in engine.index.get_postings(t))
    d["highlight_terms"] = sorted({v for t in d["matched_terms"] for v in [t, *engine.variants.get(t, [])]})
    return d


def _candidate_set(diet: str, cuisine: str, exclude: str) -> tuple[set[int] | None, list[str]]:
    """
    Compile the filters into a Boolean expression and evaluate it.
    Returns (candidate doc_ids or None when no filter, human-readable clauses).
    """
    candidates: set[int] | None = None
    clauses: list[str] = []
    if diet:
        candidates = set(engine.by_diet.get(diet, set()))
        clauses.append(f"diet:{diet}")
    if cuisine:
        s = engine.by_cuisine.get(cuisine, set())
        candidates = s if candidates is None else candidates & s
        clauses.append(f"cuisine:{cuisine}")
    if exclude.strip():
        excl_expr = " OR ".join(exclude.replace(",", " ").split())
        excluded, postfix = engine.boolean.evaluate(excl_expr)
        base = candidates if candidates is not None else engine.boolean.universe
        candidates = base - excluded
        clauses.append("NOT (" + " OR ".join(t for t in postfix if t not in ("AND", "OR", "NOT")) + ")")
    return candidates, clauses


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/meta")
def meta():
    return {"cuisines": engine.cuisines, "documents": engine.index.N,
            "models": [{"id": k, "label": v} for k, v in MODEL_LABELS.items()]}


@app.get("/search")
def search(q: str = Query(..., min_length=1), model: str = "vsm", top_k: int = 20,
           diet: str = "", cuisine: str = "", exclude: str = "",
           expand_thesaurus: bool = False, expand_cooc: bool = False, w_lex: float = 0.6):
    if model not in MODEL_LABELS:
        raise HTTPException(400, f"unknown model '{model}'")

    candidates, clauses = _candidate_set(diet, cuisine, exclude)

    if model == "boolean":
        matched, postfix = engine.boolean.evaluate(q)
        if candidates is not None:
            matched &= candidates
        tokens = [t for t in postfix if t not in ("AND", "OR", "NOT")]
        ordered = sorted(matched)[:top_k]           # no ranking: just doc order
        results = [_card(d, 1.0, set(tokens)) for d in ordered]
        boolean_expr = " ".join(engine.boolean.tokenize(q))
        total = len(matched)
        added = []
    else:
        tokens = preprocess(q)
        query_terms, added = engine.expander.expand(tokens, expand_thesaurus, expand_cooc)
        if model == "hybrid":
            results = _hybrid(query_terms, engine.semantic.encode(q), candidates, w_lex, top_k)
        else:
            ranker = engine.vsm if model == "vsm" else engine.bm25
            ranked = ranker.rank(query_terms, candidates=candidates, top_k=top_k)
            results = [_card(d, s, set(query_terms)) for d, s in ranked]
        boolean_expr = ""                           # ranked models do not require all terms
        total = len(results)

    full_expr = " AND ".join([f"({boolean_expr})"] * bool(boolean_expr) + clauses) if clauses else boolean_expr
    return {
        "query": q,
        "model": MODEL_LABELS[model],
        "model_id": model,
        "processed_tokens": tokens,
        "expanded_terms": added,
        "boolean_expression": full_expr if model == "boolean" or clauses else "",
        "candidate_count": len(candidates) if candidates is not None else engine.index.N,
        "total_matched": total,
        "results": results,
    }


@app.get("/explain")
def explain(q: str, doc_id: int, model: str = "vsm",
            expand_thesaurus: bool = False, expand_cooc: bool = False):
    if doc_id not in engine.recipes:
        raise HTTPException(404, "unknown doc_id")
    query_terms, _ = engine.expander.expand(preprocess(q), expand_thesaurus, expand_cooc)
    if model in ("bm25", "hybrid"):
        out = engine.bm25.explain(query_terms, doc_id)
    elif model == "boolean":
        matched, postfix = engine.boolean.evaluate(q)
        out = {"model": "Boolean", "columns": ["term", "in_document"],
               "rows": [{"term": t, "in_document": doc_id in engine.index.get_postings(t)}
                        for t in postfix if t not in ("AND", "OR", "NOT")],
               "footer": f"postfix: {' '.join(postfix)} ⇒ document {'matches' if doc_id in matched else 'does not match'} (no score in the Boolean model)",
               "score": 1.0 if doc_id in matched else 0.0}
    else:
        out = engine.vsm.explain(query_terms, doc_id)
    out["recipe"] = engine.recipes[doc_id].name
    return out


class FeedbackRequest(BaseModel):
    q: str
    model: str = "vsm"
    relevant_ids: list[int] = []
    nonrelevant_ids: list[int] = []
    base_vector: dict[str, float] | None = None   # previous Rocchio query, for a 2nd round
    base_embedding: list[float] | None = None     # previous Rocchio query embedding (hybrid)
    w_lex: float = 0.6
    top_k: int = 20
    diet: str = ""
    cuisine: str = ""
    exclude: str = ""
    expand_thesaurus: bool = False
    expand_cooc: bool = False


def _bm25_weights(vec: dict[str, float]) -> dict[str, float]:
    """BM25 multiplies each term's score by a query weight; rescale the Rocchio
    vector so the strongest term has weight 1 (like a typed term)."""
    m = max(vec.values()) if vec else 1.0
    return {t: w / m for t, w in vec.items()}


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.model not in ("vsm", "bm25", "hybrid"):
        raise HTTPException(400, "relevance feedback needs a ranking model")
    if not req.relevant_ids and not req.nonrelevant_ids:
        raise HTTPException(400, "mark at least one result as relevant or non-relevant")

    tokens = preprocess(req.q)
    query_terms, _ = engine.expander.expand(tokens, req.expand_thesaurus, req.expand_cooc)
    old_vec = req.base_vector or engine.vsm.query_weights(query_terms)

    rel = [engine.vsm.doc_vector(d) for d in req.relevant_ids if d in engine.recipes]
    non = [engine.vsm.doc_vector(d) for d in req.nonrelevant_ids if d in engine.recipes]
    new_vec = rocchio(old_vec, rel, non)

    candidates, clauses = _candidate_set(req.diet, req.cuisine, req.exclude)
    new_emb = None
    if req.model == "vsm":
        ranked = engine.vsm.rank_vector(new_vec, candidates=candidates, top_k=req.top_k)
        results = [_card(d, s, set(new_vec)) for d, s in ranked]
    elif req.model == "bm25":
        ranked = engine.bm25.rank(_bm25_weights(new_vec), candidates=candidates, top_k=req.top_k)
        results = [_card(d, s, set(new_vec)) for d, s in ranked]
    else:
        # Rocchio in EMBEDDING space too: same formula, vectors are MiniLM embeddings
        q_emb = np.array(req.base_embedding, dtype=np.float32) if req.base_embedding else engine.semantic.encode(req.q)
        new_emb = ALPHA * q_emb
        if rel:
            new_emb = new_emb + BETA * np.mean([engine.semantic.doc_embedding(d) for d in req.relevant_ids if d in engine.recipes], axis=0)
        if non:
            new_emb = new_emb - GAMMA * np.mean([engine.semantic.doc_embedding(d) for d in req.nonrelevant_ids if d in engine.recipes], axis=0)
        new_emb = new_emb / np.linalg.norm(new_emb)
        results = _hybrid(_bm25_weights(new_vec), new_emb, candidates, req.w_lex, req.top_k)

    return {
        "query": req.q,
        "model": MODEL_LABELS[req.model] + " · after Rocchio feedback",
        "model_id": req.model,
        "processed_tokens": tokens,
        "expanded_terms": [],
        "boolean_expression": " AND ".join(clauses),
        "candidate_count": len(candidates) if candidates is not None else engine.index.N,
        "total_matched": len(results),
        "results": results,
        "rocchio": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA,
                    "relevant": len(rel), "nonrelevant": len(non),
                    "query_vector": {t: round(w, 4) for t, w in new_vec.items()},
                    "query_embedding": [round(float(x), 5) for x in new_emb] if new_emb is not None else None,
                    "diff": diff(old_vec, new_vec)},
    }


class ExplainVectorRequest(BaseModel):
    vector: dict[str, float]
    doc_id: int
    model: str = "vsm"


@app.post("/explain_vector")
def explain_vector(req: ExplainVectorRequest):
    """Explain a result ranked with a Rocchio-modified query vector."""
    if req.doc_id not in engine.recipes:
        raise HTTPException(404, "unknown doc_id")
    if req.model in ("bm25", "hybrid"):
        out = engine.bm25.explain(_bm25_weights(req.vector), req.doc_id)
    else:
        out = engine.vsm.explain_vector(req.vector, req.doc_id)
    out["recipe"] = engine.recipes[req.doc_id].name
    return out


@app.get("/index/stats")
def index_stats():
    return engine.index.stats()


@app.get("/index/term/{term}")
def index_term(term: str):
    cleaned = preprocess(term)
    if not cleaned:
        raise HTTPException(400, "term is a stop word or empty after cleaning")
    t = cleaned[0]
    postings = engine.index.get_postings(t)
    top = sorted(postings.items(), key=lambda kv: kv[1], reverse=True)[:15]
    return {
        "term": t, "df": len(postings),
        "idf": round(engine.vsm.idf.get(t, 0.0), 4),
        "idf_bm25": round(engine.bm25.idf.get(t, 0.0), 4),
        "associations": engine.expander.associations(t),
        "thesaurus": engine.expander.thesaurus.get(t, []),
        "postings": [{"doc_id": d, "tf": tf, "name": engine.recipes[d].name} for d, tf in top],
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
EVAL_PATH = os.path.join(os.path.dirname(INDEX_DIR), "eval_queries.json")
EVAL_CONFIGS = [
    ("boolean", "Boolean", dict(model="boolean")),
    ("vsm", "VSM", dict(model="vsm")),
    ("vsm_exp", "VSM + expansion", dict(model="vsm", expand_thesaurus=True, expand_cooc=True)),
    ("bm25", "BM25", dict(model="bm25")),
    ("bm25_exp", "BM25 + expansion", dict(model="bm25", expand_thesaurus=True, expand_cooc=True)),
    ("hybrid", "Hybrid", dict(model="hybrid")),
    ("hybrid_exp", "Hybrid + expansion", dict(model="hybrid", expand_thesaurus=True, expand_cooc=True)),
]


@app.get("/evaluate")
def evaluate(k: int = 10, depth: int = 50):
    with open(EVAL_PATH, encoding="utf-8") as f:
        data = json.load(f)
    judgements = {q["query"]: set(q["relevant_ids"]) for q in data["queries"]}
    reviewed = sum(1 for q in data["queries"] if q.get("reviewed"))
    out = []
    for cid, label, params in EVAL_CONFIGS:
        if params["model"] not in MODEL_LABELS:
            continue
        runs = {}
        for q in judgements:
            res = search(q=q, top_k=depth, **params)
            runs[q] = [r["doc_id"] for r in res["results"]]
        ev = evaluate_run(runs, judgements, k=k)
        ev.update({"id": cid, "label": label})
        out.append(ev)
    return {"k": k, "depth": depth, "queries": len(judgements), "reviewed": reviewed, "configs": out}


# --------------------------------------------------------------------------
# "What can I cook?"
# --------------------------------------------------------------------------
class PantryRequest(BaseModel):
    ingredients: list[str]
    min_coverage: float = 0.5
    top_k: int = 20
    diet: str = ""
    cuisine: str = ""
    assume_staples: bool = True


@app.post("/pantry")
def pantry(req: PantryRequest):
    candidates, clauses = _candidate_set(req.diet, req.cuisine, "")
    hits = engine.pantry.search(req.ingredients, req.min_coverage, req.top_k, candidates,
                                assume_staples=req.assume_staples)
    results = []
    for h in hits:
        c = _card(h["doc_id"], h["score"], set())
        c.update({"coverage": h["coverage"], "usage": h["usage"], "matched": h["matched"],
                  "assumed": h["assumed"], "missing": h["missing"]})
        results.append(c)
    return {"pantry": req.ingredients, "assume_staples": req.assume_staples,
            "boolean_expression": " AND ".join(clauses), "results": results}


# --------------------------------------------------------------------------
# Add a recipe (custom collection) + background re-index
# --------------------------------------------------------------------------
class NewRecipe(BaseModel):
    name: str
    ingredients: str            # comma-separated
    instructions: str = ""
    cuisine: str = "Custom"
    total_time: int = 0
    url: str = ""
    image_url: str = ""


rebuild_state = {"running": False, "log": [], "last_error": None}


def _rebuild():
    global engine
    from scripts.build_index import build_all
    rebuild_state.update(running=True, log=[], last_error=None)
    try:
        build_all(log=lambda m: rebuild_state["log"].append(m))
        engine = Engine()
        rebuild_state["log"].append("Engine reloaded — new recipe is searchable")
    except Exception as e:                       # noqa: BLE001
        rebuild_state["last_error"] = str(e)
    finally:
        rebuild_state["running"] = False


@app.post("/recipes")
def create_recipe(r: NewRecipe):
    if rebuild_state["running"]:
        raise HTTPException(409, "a re-index is already running; try again in a minute")
    ingredients = [i.strip() for i in r.ingredients.split(",") if i.strip()]
    if not r.name.strip() or not ingredients:
        raise HTTPException(400, "name and at least one ingredient are required")
    item = r.model_dump()
    item["ingredients"] = ingredients
    item["diet"] = classify_diet(ingredients)
    add_custom_recipe(item)
    threading.Thread(target=_rebuild, daemon=True).start()
    return {"saved": item, "message": "Recipe saved. Re-indexing in the background (index ≈ 6 s, embedding ≈ 1 s per recipe)."}


@app.get("/rebuild/status")
def rebuild_status():
    return rebuild_state


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
