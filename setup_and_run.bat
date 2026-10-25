@echo off
REM SearchMyRecipe — Windows one-shot setup + run
python -m venv .venv
call .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p, quiet=True) for p in ['stopwords','wordnet','omw-1.4']]"
python scripts\build_index.py
python scripts\smoke_test.py
uvicorn app.main:app --port 8000
