# VIVA cheat-sheet — SearchMyRecipe

Answers the group should be able to give from memory. Every number below comes from the running system, so you can reproduce it live.

## One-paragraph pitch
"SearchMyRecipe is an explainable recipe retrieval system over 5,938 Indian recipes. A query passes through the same text operations as the documents, is optionally expanded (thesaurus and corpus co-occurrence), filtered by a Boolean model, ranked by the Vector Space Model or BM25 over a hand-built inverted index, and — in Hybrid mode — fused with MiniLM sentence-embedding similarity so recipes are found even when no keyword matches. Rocchio relevance feedback refines the query from 👍/👎, and every result has a 'Why this result?' panel that shows the score term by term. We evaluated all models with Precision@10, Recall@10 and MAP on 20 labelled queries."

## Where is the IR? (file → concept)
| Concept | File | What to say |
|---|---|---|
| Text operations | `ir/preprocess.py` | tokenise → stop words (English + recipe words like *cup, tbsp, chopped*) → lemmatise (*tomatoes→tomato*) → normalise (*dahi→curd, bhindi→okra*). Same pipeline for query and documents = "logical view". |
| Inverted index | `ir/inverted_index.py` | `term → {doc_id: tf}`, plus df, doc length, max tf. 9,875 terms. One lookup instead of scanning 5,938 recipes. |
| Boolean model | `ir/boolean_model.py` | Shunting-yard → postfix → set ops. AND=∩, OR=∪, NOT=complement. Unranked. Used for filters. |
| VSM | `ir/vsm.py` | tf = f/max f; idf = log10(N/n); w = tf·idf; cosine = d·q/(|d||q|). |
| BM25 | `ir/bm25.py` | idf·f(k1+1)/(f + k1(1−b+b·|d|/avgdl)); k1=1.5 saturation, b=0.75 length. |
| Query expansion | `ir/query_expansion.py` | Thesaurus (0.5) + global co-occurrence, Jaccard c/(nu+nv−c) (0.4). |
| Rocchio | `ir/rocchio.py` | q′ = 1·q + (0.75/|Dr|)Σd − (0.15/|Dn|)Σd. Applied in TF-IDF space and in embedding space. |
| Semantic | `ir/semantic.py` | MiniLM → 384-dim unit vectors; cosine = dot product; recipes encoded once, cached. |
| Fusion | `ir/fusion.py` | min-max each list to [0,1]; fused = 0.6·BM25′ + 0.4·MiniLM′. = result merging. |
| Evaluation | `ir/evaluation.py` | P@k, R@k, AP, MAP. |
| Pantry | `ir/pantry.py` | coverage = |recipe ∩ have| / |recipe|; usage = |typed ∩ recipe| / |typed|; score = harmonic mean (F-measure). Kitchen staples assumed present. |

## Worked numbers you can quote
- Query **dahi bhindi** → tokens `curd, okra`. Top hit *Dahi Bhindi* cosine **0.905**: `okra` df 116, idf 1.709, contribution 2.921; `curd` df 999, idf 0.774, contribution 0.424. Rare terms dominate — that is IDF.
- Query **paneer** → 431 postings; idf = log10(5938/431) = **1.139**.
- **BM25** *Chicken Donne Biryani* for "chicken biryani": `biryani` (df 127, idf 3.84) → 8.15; `chicken` (df 521, idf 2.43) → 5.32; total 13.47. |d| = 261 tokens vs avgdl 158.
- **Expansion** of *biryani*: thesaurus adds basmati, rice, saffron (0.5); co-occurrence adds salan (0.4, Jaccard 0.179). Without expansion *Biryani Masala Powder* ranks 1st; with expansion real biryanis win.
- **Rocchio**: query *paneer*, 👍 Tawa Paneer Masala + Kashmiri Paneer Masala, 👎 Paneer Tikka Pulao → `paneer` weight 1.139 → 1.823; added `kashmiri 0.299, masala 0.214, tawa 0.179, garam 0.109, saffron 0.102`. *Paneer Butter Masala* enters the top 6.
- **MiniLM** sanity check: "creamy rich paneer dish for guests" vs "Shahi Paneer with cashew cream gravy" = 0.644; vs "Bhindi fry" = 0.298. Zero shared words in either pair.
- **Evaluation (MAP, k=10, 20 queries)**: Hybrid **0.582** > BM25 0.557 > VSM 0.442 > Boolean 0.159. Expansion lowers MAP (BM25+exp 0.419) on exact-name queries — precision/recall trade-off.

## Likely questions → answers
1. **Why not scan every recipe?** Linear in collection size; the inverted index makes query cost proportional to posting-list lengths of the query terms only.
2. **Why cosine, not dot product?** Dot product grows with document length; cosine divides by the norms so a long recipe isn't favoured for being long.
3. **Can the Boolean model rank?** No — every match is equally "true". We use it for constraints and hand the set to a ranker.
4. **What does BM25 add?** Saturation (12th occurrence adds little) and length normalisation. It also has a probabilistic derivation (Robertson–Spärck Jones idf).
5. **How do you find recipes without the keyword?** Three layers: normalisation (dahi→curd), expansion (thesaurus/co-occurrence), embeddings (meaning).
6. **Local vs global analysis?** Global = association statistics from the whole corpus, computed once (ours). Local = from the documents retrieved for the current query. Rocchio is user feedback, not automatic.
7. **What is MiniLM?** A 6-layer, ~22M-parameter transformer trained so that semantically similar sentences have close vectors. We do not train anything; we only encode.
8. **Why min-max before fusing?** BM25 is unbounded (0–20), cosine is 0–1; unnormalised sums let BM25 drown the semantic score.
9. **Where did ground truth come from?** 20 queries; relevant lists pre-filled by name rules, then reviewed by the group.
10. **Why did expansion lower MAP?** Test queries are exact dish names; added neighbour terms admit related but non-relevant recipes. Expansion is for vague queries.
11. **What happens with 1 million recipes?** Compress postings, shard the index (distributed IR — brokers, result merging, which we already do for fusion), approximate nearest-neighbour search for embeddings.
12. **Why two measures in "What can I cook?"** Coverage alone favours 2-ingredient dishes (trivially 100%); usage alone favours huge dishes you cannot finish. They trade off like precision and recall, so we combine them with the harmonic mean — low unless both are high. Staples (salt, oil, onion, basic spices) are assumed present, otherwise a 4-item pantry could never pass 50% on a 15-ingredient recipe.
13. **Why no recipe photos?** The dataset stores URLs on archanaskitchen.com, which answers HTTP 403 to any request that does not come from its own pages (hotlink protection) — we verified this. Rather than show broken images we generate a tile per recipe: the glyph comes from a keyword rule over the name and ingredients (14 categories), the colour from a hash of the name, so it is deterministic and works offline.
14. **Limitations?** Rule-based diet classification; MiniLM can blur polarity (hot/cold); thesaurus is hand-written; evaluation set is small.
