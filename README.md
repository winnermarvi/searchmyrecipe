# SearchMyRecipe — an explainable recipe retrieval system

IR project, BE CMPN B, Group B6 (Salil Mahale, Winner Marvi, Sufiyan Mithawala), SFIT / Mumbai University, CSDC 7023.

## Run it

First time: `setup_and_run.bat` (Windows) or `./setup_and_run.sh` (macOS/Linux) — creates a venv, installs, builds the index, runs the smoke test and starts the server. Afterwards:

```bash
source .venv/bin/activate    # Windows: .venv\Scripts\activate
uvicorn app.main:app --port 8000     # open http://127.0.0.1:8000
```

Manual steps, if you prefer:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only PyTorch (smaller)
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ['stopwords','wordnet','omw-1.4']]"
python scripts/build_index.py     # ~6 s; MiniLM embeddings are shipped pre-built in data/index/
python scripts/smoke_test.py      # must print ALL GOOD
uvicorn app.main:app --port 8000
```

See **DEMO_GUIDE.md** for the presentation script and **VIVA.md** for the concept cheat-sheet.

API docs: http://127.0.0.1:8000/docs

## Where each IR concept lives

| Concept (syllabus) | File |
|---|---|
| Text operations: tokenise, stop words, lemmatise, ingredient normalisation | `ir/preprocess.py`, `data/thesaurus.json` |
| Document collection & logical view (field boosting) | `ir/corpus.py` |
| Inverted index, df, document length | `ir/inverted_index.py` |
| Vector Space Model: TF-IDF, cosine, explanation | `ir/vsm.py` |
| Boolean model: shunting-yard parser, set operations | `ir/boolean_model.py` |
| BM25 ranking (probabilistic family) | `ir/bm25.py` |
| Query expansion: thesaurus + global co-occurrence analysis | `ir/query_expansion.py`, `data/thesaurus.json` |
| Rocchio relevance feedback (query expansion + term re-weighting) | `ir/rocchio.py` |
| Semantic layer: MiniLM sentence embeddings, cosine | `ir/semantic.py` |
| Score fusion / result merging (min-max + weighted sum) | `ir/fusion.py` |
| Evaluation: Precision@k, Recall@k, Average Precision, MAP | `ir/evaluation.py`, `data/eval_queries.json` |
| Ingredient retrieval ("What can I cook?"): coverage × usage, F-measure | `ir/pantry.py` |
| Generated recipe tiles (photo host blocks hotlinking) | `ir/visuals.py` |
| Web API | `app/main.py` |
| Interface | `app/static/index.html` |

## Ground truth for evaluation
`data/eval_queries.json` holds 20 test queries with relevant doc_ids, pre-filled by name rules. The group should review each list (Index explorer → look up terms, or search) and set `reviewed: true`.

## Adding your own recipes
Use the **Add recipe** tab (re-indexes in the background), or:
Append to `data/custom_recipes.json` (see the field names in `ir/corpus.py::load_custom`) and re-run `python scripts/build_index.py` (embeddings are re-encoded automatically when the collection changes).

## Status
Complete (Phases 0–7). Group to-do: review `data/eval_queries.json` and set `reviewed: true`.