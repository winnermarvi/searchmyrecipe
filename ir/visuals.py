"""
Recipe tiles
------------
The dataset's image URLs point at archanaskitchen.com, which returns HTTP 403
for any request from outside its own site (hotlink protection), so the photos
cannot be displayed. Instead each recipe gets a generated tile:

  * a CATEGORY, derived by keyword rules over the recipe name and ingredients
    (the same kind of transparent rule used for the veg / non-veg label), which
    picks the glyph;
  * a COLOUR chosen by hashing the recipe name, so a given recipe always looks
    the same and neighbouring cards look different.

No network, no per-request cost, and the card reads as designed rather than
broken. A custom recipe added through the UI can still supply a real image URL.
"""

import hashlib

# Ordered: the first matching rule wins, so put specific categories first.
_CATEGORIES: list[tuple[str, str, tuple[str, ...]]] = [
    ("drink",     "🥤", ("lassi", "juice", "smoothie", "sharbat", "shake", "chaas", "buttermilk",
                         "tea", "coffee", "lemonade", "mojito", "cooler", "thandai", "jaljeera", "drink")),
    ("dessert",   "🍰", ("cake", "brownie", "cookie", "halwa", "kheer", "ladoo", "laddu", "barfi",
                         "burfi", "jamun", "rabri", "payasam", "pudding", "ice cream", "dessert",
                         "sweet", "mousse", "cheesecake", "tart", "pie", "custard", "modak", "peda")),
    ("soup",      "🍲", ("soup", "rasam", "sambar", "dal ", "daal", "lentil", "stew", "broth", "shorba")),
    ("bread",     "🫓", ("roti", "naan", "paratha", "chapati", "poori", "puri", "kulcha", "bhatura",
                         "thepla", "bread", "focaccia", "pizza", "bun", "dosa", "uttapam", "cheela",
                         "pancake", "waffle", "toast", "sandwich", "burger", "wrap", "roll")),
    ("rice",      "🍚", ("biryani", "pulao", "fried rice", "rice", "khichdi", "pongal", "risotto",
                         "poha", "upma", "sevai", "vermicelli", "noodle", "pasta", "spaghetti",
                         "macaroni", "idli")),
    ("salad",     "🥗", ("salad", "raita", "kachumber", "slaw", "sprout")),
    ("snack",     "🥟", ("pakora", "bhaji", "vada", "samosa", "kachori", "cutlet", "tikki", "chaat",
                         "bhel", "sev", "namkeen", "spring roll", "momo", "dumpling", "fritter",
                         "murukku", "chakli", "mathri", "snack", "starter")),
    ("chutney",   "🥣", ("chutney", "pickle", "achar", "thokku", "podi", "masala powder", "paste",
                         "sauce", "dip", "jam", "spice mix")),
    ("seafood",   "🐟", ("fish", "prawn", "shrimp", "crab", "squid", "pomfret", "surmai", "salmon",
                         "tuna", "mackerel", "clam", "seafood")),
    ("chicken",   "🍗", ("chicken", "murgh", "murg", "duck", "turkey")),
    ("mutton",    "🍖", ("mutton", "lamb", "gosht", "keema", "pork", "beef", "bacon", "ham")),
    ("egg",       "🥚", ("egg", "anda", "omelette", "frittata")),
    ("paneer",    "🧀", ("paneer", "cheese", "tofu")),
]
_DEFAULT = ("curry", "🍛")

# Warm, muted tones that sit well on the cream background.
_COLOURS = [
    ("#E8F0E9", "#2F5D45"), ("#FBEFD8", "#8A5A0B"), ("#F6E9E4", "#8C4A38"),
    ("#EDEBF6", "#4B4380"), ("#E7F0F3", "#2C5866"), ("#F5EEDC", "#6B5B2A"),
    ("#F3E9EF", "#7A3B5C"), ("#E9F1EC", "#3B6B54"),
]


def categorize(name: str, ingredients: list[str]) -> tuple[str, str]:
    """Return (category, glyph) for a recipe. First matching keyword wins."""
    haystack = (name + " " + " ".join(ingredients)).lower()
    title = name.lower()
    for category, glyph, keywords in _CATEGORIES:
        # the name is stronger evidence than the ingredient list
        if any(k in title for k in keywords):
            return category, glyph
    for category, glyph, keywords in _CATEGORIES:
        if any(k in haystack for k in keywords):
            return category, glyph
    return _DEFAULT


def tile(name: str, ingredients: list[str]) -> dict:
    """Everything the card needs to draw its tile."""
    category, glyph = categorize(name, ingredients)
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16)
    background, ink = _COLOURS[h % len(_COLOURS)]
    return {"category": category, "glyph": glyph, "background": background, "ink": ink}
