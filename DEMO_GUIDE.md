# DEMO guide — SearchMyRecipe

## A. First-time setup (do this the day BEFORE the demo, on the laptop you will present with)

1. Install Python 3.10–3.12 (3.12 recommended). Windows: tick "Add Python to PATH".
2. Unzip the project. Open a terminal **inside** the `searchmyrecipe` folder.
3. Run ONE of:
   - Windows: `setup_and_run.bat`
   - macOS/Linux: `./setup_and_run.sh`

   What it does, in order: creates a virtual environment → installs CPU PyTorch (~200 MB) → installs the rest → downloads NLTK data → **builds the index** (`scripts/build_index.py`, ~6 s; embeddings are already shipped in `data/index/`) → runs `scripts/smoke_test.py` (must print `ALL GOOD`) → starts the server.
4. Open http://127.0.0.1:8000 in Chrome/Edge. The first search in Hybrid mode downloads the MiniLM model (~80 MB) once — **do this while you have internet**. After that everything is offline except recipe images (they load from archanaskitchen.com; without internet the cards simply show a green placeholder).
5. Close the server with Ctrl+C.

## B. Every later run (demo day) — 2 commands

```
.venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (macOS/Linux)
uvicorn app.main:app --port 8000
```
Then open http://127.0.0.1:8000. If anything looks off, run `python scripts/smoke_test.py` first.

Only rebuild the index (`python scripts/build_index.py`) if you edited the dataset, `thesaurus.json`, `preprocess.py` stop words, or added recipes. Adding a recipe from the UI rebuilds automatically.

## C. The 8-minute demo script (in this order)

| # | Do | Say | Points to |
|---|---|---|---|
| 1 | Tab **How it works** | Walk the diagram left to right in 30 s. | Architecture |
| 2 | **Search** `dahi bhindi`, model VSM | Show the pipeline strip: typed → `curd, okra` after text operations. "Hindi and English land on the same index term." Open *Why this result?* on the first card and read the tf/idf row for `okra` vs `curd`. | Module 1 text ops, VSM |
| 3 | Tab **Index explorer**, look up `paneer` | "431 recipes, idf 1.139; here is its posting list and its strongest co-occurring terms." | Inverted index, global analysis |
| 4 | Back to Search: model **Boolean**, query `paneer AND (butter OR cream) NOT onion` | "75 recipes satisfy it — a set, no ranking." Then set Diet = Vegetarian and Exclude = `peanut` in VSM mode and show the compiled Boolean filter line. | Boolean model |
| 5 | Model **BM25**, query `chicken biryani` | Compare the list with VSM; open *Why this result?* and point at saturation/length in the footer. | BM25 |
| 6 | Query `biryani` with both expansion boxes OFF, then ON | "Without expansion the spice powder ranks first; with basmati/saffron/salan the real biryanis win." Hover a dashed chip to show its origin. | Query expansion |
| 7 | Query `paneer` (VSM): 👍 two masala gravies, 👎 a pulao → **Refine with feedback** | Read the Rocchio panel: strengthened / added terms; α β γ. | Relevance feedback |
| 8 | Model **Hybrid**, query `something creamy and rich to impress guests` | "No dish name in the query; BM25 alone can't do this." Drag the fusion slider to show the ranking shift. Open *Why this result?* → fusion line. | Embeddings, result merging |
| 9 | Tab **Evaluation** → Run evaluation (takes ~10 s) | "Hybrid > BM25 > VSM > Boolean on MAP. Expansion lowered MAP on exact-name queries — precision/recall trade-off." | Module 4 evaluation |
| 10 | Tab **What can I cook?** with `potato, onion, tomato, cumin, turmeric, salt, oil` | Coverage ranking, missing items. | Set-based retrieval |
| 11 | (Optional) **Add recipe**, then search for it | "Text ops → index → co-occurrence → only the new recipe embedded → live." | Retrieval process end to end |

Practice once with a timer. Each teammate should own 3–4 rows.

## D. What to expect / not to expect

- **Expect** Hybrid's first query after server start to take 3–10 s (model loads). Later queries are instant. Warm it up before the teacher arrives by running one Hybrid search.
- **Expect** the Evaluation tab to take ~10 s: it runs 7 configurations × 20 queries.
- **Expect** generated tiles instead of photos. If asked: the dataset's image host returns 403 to outside requests, so each card draws a category glyph (rule-based) with a colour hashed from the recipe name. Custom recipes you add can carry a real image URL.
- **Expect** some odd co-occurrence terms (e.g. `chocolate cake → cocoa, vanilla` is good; occasionally a method word slips in). Own it: "corpus statistics are noisy, which is why co-occurrence has the lowest weight."
- **Expect** MiniLM to occasionally confuse hot/cold or sweet/savoury. Own it: "this is why we fuse with BM25 rather than use embeddings alone."
- **Do not** type queries made only of stop words (`something nice`) — the strip will say nothing survived text operations. That is correct behaviour, but not a good demo.
- **Do not** use Rocchio on the Boolean model — buttons are hidden there by design.
- **Do not** add a recipe and immediately switch tabs during the rebuild; wait for "Engine reloaded" in the log (~6 s).
- **Do not** claim the diet label is ML — it is a keyword rule (`corpus.py`), and say so if asked.
- **Do not** edit `data/index/` by hand. If in doubt: `python scripts/build_index.py` then `python scripts/smoke_test.py`.
- If the projector laptop has no internet: everything works except Google Fonts and the first-time MiniLM download (so do step A.4 beforehand).

## E. If something breaks on the day

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | The venv is not active — run the activate command from section B. |
| "Hybrid" missing from the model list | Embeddings not found → `python scripts/build_index.py` (with internet the first time). |
| Port already in use | `uvicorn app.main:app --port 8001` and open :8001. |
| Search returns nothing | Check the pipeline strip: were all words stop words? Try a dish name. |
| Page looks unstyled | Tailwind and Chart.js are bundled in `app/static/`; only Google Fonts need internet (falls back to system fonts). Hard-refresh (Ctrl+F5). |
