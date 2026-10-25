"""
Corpus (the document collection)
--------------------------------
Loads recipes from the cleaned Kaggle CSV plus any custom recipes the group
adds later (data/custom_recipes.json), and turns each into a Recipe object.

Each Recipe also gets an `indexed_text`: the string the inverted index will
see. We use simple FIELD BOOSTING: the recipe name is repeated 3 times and
the ingredient list 2 times, so a query term found in the title counts more
than the same term buried in the instructions.
"""

import csv
import json
import os
from dataclasses import dataclass, asdict

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CSV_PATH = os.path.join(_DATA_DIR, "raw", "Cleaned_Indian_Food_Dataset.csv")
CUSTOM_PATH = os.path.join(_DATA_DIR, "custom_recipes.json")

# Rule-based diet classification (explainable, no ML):
# if any of these words appears in the ingredient list -> Non-Vegetarian.
_NON_VEG_WORDS = {
    "chicken", "mutton", "lamb", "fish", "prawn", "prawns", "shrimp", "egg",
    "eggs", "beef", "pork", "crab", "squid", "meat", "keema", "bacon", "ham",
    "salmon", "tuna", "anchovy", "anchovies", "sausage", "turkey", "duck",
    "mackerel", "pomfret", "surmai", "bombil", "kingfish", "octopus", "clams",
}


@dataclass
class Recipe:
    doc_id: int
    name: str
    ingredients: list[str]      # cleaned ingredient names, e.g. ["onion", "cumin seeds"]
    ingredients_raw: str        # original with quantities, for display
    instructions: str
    cuisine: str
    diet: str                   # "Vegetarian" | "Non-Vegetarian"
    total_time: int
    url: str
    image_url: str
    source: str                 # "kaggle" | "custom"

    @property
    def indexed_text(self) -> str:
        name3 = (self.name + " ") * 3
        ingr2 = (", ".join(self.ingredients) + " ") * 2
        return name3 + ingr2 + self.instructions

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ingredients_count"] = len(self.ingredients)
        return d


def classify_diet(ingredients: list[str]) -> str:
    words = set()
    for ing in ingredients:
        words.update(ing.lower().replace("(", " ").replace(")", " ").split())
    return "Non-Vegetarian" if words & _NON_VEG_WORDS else "Vegetarian"


def _clean_name(name: str) -> str:
    name = name.strip()
    if name.lower().endswith(" recipe"):
        name = name[:-7]
    return name


def _split_ingredients(text: str) -> list[str]:
    return [i.strip() for i in text.split(",") if i.strip()]


def load_kaggle(path: str = CSV_PATH) -> list[Recipe]:
    recipes = []
    with open(path, encoding="utf-8") as f:
        for doc_id, row in enumerate(csv.DictReader(f)):
            ingredients = _split_ingredients(row["Cleaned-Ingredients"])
            recipes.append(Recipe(
                doc_id=doc_id,
                name=_clean_name(row["TranslatedRecipeName"]),
                ingredients=ingredients,
                ingredients_raw=row["TranslatedIngredients"],
                instructions=row["TranslatedInstructions"].strip(),
                cuisine=row["Cuisine"].strip().replace(" Recipes", "").replace("\ufeff", ""),
                diet=classify_diet(ingredients),
                total_time=int(float(row["TotalTimeInMins"] or 0)),
                url=row["URL"],
                image_url=row["image-url"],
                source="kaggle",
            ))
    return recipes


def load_custom(path: str = CUSTOM_PATH, start_id: int = 0) -> list[Recipe]:
    """Custom recipes: a JSON list of {name, ingredients (list or comma string),
    instructions, cuisine, total_time, url?, image_url?}."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    recipes = []
    for i, item in enumerate(items):
        ingr = item.get("ingredients", [])
        if isinstance(ingr, str):
            ingr = _split_ingredients(ingr)
        recipes.append(Recipe(
            doc_id=start_id + i,
            name=_clean_name(item["name"]),
            ingredients=ingr,
            ingredients_raw=item.get("ingredients_raw", ", ".join(ingr)),
            instructions=item.get("instructions", ""),
            cuisine=item.get("cuisine", "Custom"),
            diet=item.get("diet") or classify_diet(ingr),
            total_time=int(item.get("total_time", 0)),
            url=item.get("url", ""),
            image_url=item.get("image_url", ""),
            source="custom",
        ))
    return recipes


def load_corpus() -> list[Recipe]:
    """Kaggle recipes followed by custom ones, with continuous doc_ids."""
    kaggle = load_kaggle()
    custom = load_custom(start_id=len(kaggle))
    return kaggle + custom


def add_custom_recipe(item: dict, path: str = CUSTOM_PATH) -> None:
    items = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
    items.append(item)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
