"""
"What can I cook?" — ingredient-coverage retrieval
--------------------------------------------------
The query is not text but a SET of ingredients the user has. Each recipe is
also a set of ingredients, so this is set-based (Boolean-family) retrieval
with a graded score instead of a yes/no. TWO measures matter, and they pull
in opposite directions exactly like precision and recall:

    coverage(recipe) = |recipe ∩ have| / |recipe|     "can I make it?"
                       → a 2-ingredient dish scores 1.0 trivially
    usage(recipe)    = |typed ∩ recipe| / |typed|     "does it use what I have?"
                       → a huge dish using all my items scores 1.0 but I cannot cook it

Ranking on either one alone gives bad results, so we combine them with the
HARMONIC MEAN — the same F-measure used to combine precision and recall:

    score = 2 · coverage · usage / (coverage + usage)

The harmonic mean is low unless BOTH are high, so the top results are dishes
you can almost complete AND that actually use your ingredients.

An ingredient line like "cumin seeds (jeera)" counts as matched if any of
its cleaned tokens is in the cleaned pantry. Recipes are ranked by coverage,
ties broken by fewer missing items, and the missing items are shown so the
user knows what to buy.

KITCHEN STAPLES. Recipes list 10-20 ingredients, most of them things every
kitchen already has (salt, oil, water, turmeric, mustard seeds). If the user
had to type all of them, coverage would never pass 50% and useful recipes
would be filtered out — e.g. Vada Pav matched only 3 of 15 ingredients for
the pantry {pav, potato, coriander, masala}. So by default a fixed STAPLES
set is added to whatever the user typed; the staples are still listed under
"assumed" so the user can see what we took for granted. Turning the
assumption off makes the scoring strict again.
"""

from ir.preprocess import preprocess

# Assumed present in any kitchen unless the user turns the assumption off.
STAPLES = [
    "salt", "water", "sugar", "sunflower oil", "oil", "ghee", "butter",
    "turmeric powder", "red chilli powder", "green chillies", "cumin seeds",
    "mustard seeds", "coriander powder", "garam masala powder", "asafoetida",
    "curry leaves", "onion", "garlic", "ginger", "black pepper", "lemon",
]

# tokens too generic to prove an ingredient is present
_GENERIC = {"powder", "seed", "leaf", "oil", "fresh", "dry", "whole", "ground", "red", "green", "black", "white"}


def ingredient_tokens(ingredient: str) -> set[str]:
    return {t for t in preprocess(ingredient) if t not in _GENERIC}


class PantrySearch:
    def __init__(self, recipes):
        self.recipes = recipes
        self.tokens: dict[int, list[tuple[str, set[str]]]] = {
            r.doc_id: [(ing, ingredient_tokens(ing)) for ing in r.ingredients] for r in recipes
        }

    def search(self, pantry: list[str], min_coverage: float = 0.5, top_k: int = 20,
               candidates: set[int] | None = None, assume_staples: bool = True) -> list[dict]:
        have: set[str] = set()
        for item in pantry:
            have |= ingredient_tokens(item)
        typed = set(have)
        staple_tokens: set[str] = set()
        if assume_staples:
            for item in STAPLES:
                staple_tokens |= ingredient_tokens(item)
            have |= staple_tokens
        scored = []
        for doc_id, ings in self.tokens.items():
            if candidates is not None and doc_id not in candidates:
                continue
            if not ings:
                continue
            matched = [ing for ing, toks in ings if toks & have]
            missing = [ing for ing, toks in ings if not (toks & have)]
            assumed = [ing for ing, toks in ings if (toks & staple_tokens) and not (toks & typed)]
            yours = [ing for ing, toks in ings if toks & typed]
            if not yours:                        # nothing the user actually typed is used
                continue
            cov = len(matched) / len(ings)
            used = {t for ing, toks in ings for t in (toks & typed)}
            usage = len(used) / len(typed) if typed else 0.0
            score = 2 * cov * usage / (cov + usage) if (cov + usage) else 0.0
            if cov >= min_coverage:
                scored.append((score, cov, usage, -len(missing), doc_id, yours, assumed, missing))
        scored.sort(key=lambda x: (x[0], x[3]), reverse=True)
        return [{"doc_id": d, "score": round(sc, 3), "coverage": round(c, 3),
                 "usage": round(u, 3), "matched": y, "assumed": a, "missing": mi}
                for sc, c, u, _, d, y, a, mi in scored[:top_k]]
