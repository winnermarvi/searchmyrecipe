"""
Text Operations (IR Module 1)
-----------------------------
Every document AND every query passes through the SAME pipeline:

    raw text -> tokenize -> remove stop words -> lemmatize -> normalize ingredients

The output is the "logical view" of the text: a list of clean index terms.
Because documents and queries share this pipeline, a query term can only
match a document term if both were cleaned the same way.
"""

import json
import os
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# ---------------------------------------------------------------------------
# Stop words = words that appear in almost every recipe, so they carry no
# discriminating power (their IDF would be ~0 anyway). We use the standard
# English list PLUS a recipe-domain list.
# ---------------------------------------------------------------------------
_ENGLISH_STOPWORDS = set(stopwords.words("english"))

_RECIPE_STOPWORDS = {
    "cup", "cups", "tsp", "tbsp", "teaspoon", "teaspoons", "tablespoon",
    "tablespoons", "gram", "grams", "gm", "kg", "ml", "litre", "liter", "inch",
    "chopped", "finely", "sliced", "diced", "grated", "minced", "peeled",
    "add", "mix", "cook", "stir", "heat", "serve", "minute", "minutes",
    "required", "taste", "recipe", "make", "making", "begin", "well", "also",
    "little", "small", "medium", "large", "pinch", "piece", "pieces", "water",
    "need", "needed", "keep", "aside", "along", "till", "until", "once",
    "inserted", "toothpick", "transfer", "place", "next", "ready", "done", "mixing",
    "remove", "set", "side", "another", "turn", "bowl", "pan", "wok", "kadai",
    # query-intent words: they say how the user feels, not what the dish is
    "something", "anything", "want", "like", "good", "nice", "best", "great",
    "delicious", "tasty", "yummy", "dish", "food", "meal", "impress", "guest", "guests",
    "tonight", "today", "give", "show", "find", "please", "really",
}

STOPWORDS = _ENGLISH_STOPWORDS | _RECIPE_STOPWORDS

_lemmatizer = WordNetLemmatizer()

_TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(text: str) -> list[str]:
    """Lowercase the text and keep only alphabetic words."""
    return _TOKEN_RE.findall(text.lower())


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Drop stop words and very short tokens (1-2 letters)."""
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def lemmatize(tokens: list[str]) -> list[str]:
    """Reduce words to their dictionary form: 'tomatoes' -> 'tomato'."""
    return [_lemmatizer.lemmatize(t) for t in tokens]


def _load_canonical_map() -> dict[str, str]:
    """
    thesaurus.json has a "canonical" section mapping variants to one name,
    e.g. "capsicum" -> "bellpepper", "dahi" -> "curd".
    """
    path = os.path.join(_DATA_DIR, "thesaurus.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("canonical", {})


CANONICAL = _load_canonical_map()


def normalize_ingredients(tokens: list[str]) -> list[str]:
    """Map ingredient synonyms/Hindi names to one canonical term."""
    return [CANONICAL.get(t, t) for t in tokens]


def preprocess(text: str) -> list[str]:
    """Full pipeline. Used for documents at index time and queries at search time."""
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    tokens = normalize_ingredients(tokens)
    return tokens
