"""Quick health check: run after build_index.py, before a demo.  python scripts/smoke_test.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from app.main import app

c = TestClient(app)
ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)

meta = c.get("/meta").json()
check(f"index loaded ({meta['documents']} recipes)", meta["documents"] > 5000)
models = [m["id"] for m in meta["models"]]
check(f"models available: {', '.join(models)}", "hybrid" in models)
r = c.get("/search", params=dict(q="dahi bhindi", model="vsm")).json()
check("VSM finds Dahi Bhindi first", "dahi bhindi" in r["results"][0]["name"].lower())
r = c.get("/search", params=dict(q="biryani", model="bm25", expand_thesaurus=True, expand_cooc=True)).json()
check("query expansion adds terms", len(r["expanded_terms"]) >= 3)
r = c.get("/search", params=dict(q="paneer AND (butter OR cream) NOT onion", model="boolean")).json()
check(f"Boolean model ({r['total_matched']} matches)", r["total_matched"] > 10)
r = c.get("/search", params=dict(q="cold drink for a hot summer afternoon", model="hybrid")).json()
check("Hybrid model answers", len(r["results"]) == 20 and "components" in r["results"][0])
f = c.post("/feedback", json=dict(q="paneer", model="vsm", relevant_ids=[r0 := c.get("/search", params=dict(q="paneer masala", model="vsm")).json()["results"][0]["doc_id"]])).json()
check("Rocchio feedback re-ranks", len(f["rocchio"]["diff"]["added"]) > 0)
p = c.post("/pantry", json=dict(ingredients=["potato", "onion", "tomato", "cumin", "turmeric", "salt", "oil"])).json()
check("pantry search works", len(p["results"]) > 0)
print("\nALL GOOD — ready to demo" if ok else "\nSomething failed — re-run scripts/build_index.py and check the error above")
sys.exit(0 if ok else 1)
