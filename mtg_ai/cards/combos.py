import requests
import json
import pandas as pd
from pathlib import Path


class ComboData:
    def __init__(self, data: dict):
        self.id = data["d"]
        self.cards = data["c"]
        self.card_count = len(self.cards)
        self.color_identity = data["i"]
        self.prerequisites = [prereq["s"] for prereq in data["t"]]
        self.results = [r.strip() for r in data["r"].split(".")]
        self.steps = data["s"].splitlines()
        self.number_of_cards = len(self.cards)
        self.number_of_steps = len(self.steps)
        self.number_of_prerequisites = len(self.prerequisites)
        self.number_of_results = len(self.results)
        self.b = data["b"]
        self.o = data["o"]

    def __str__(self):
        return f"Combo[{self.id}]"

    def serialize(self):
        return {
            "id": self.id,
            "color": self.color_identity.replace(",", ""),
            "number_of_cards": self.card_count,
            "number_of_steps": self.number_of_steps,
            "number_of_prerequisites": self.number_of_prerequisites,
            "number_of_results": self.number_of_results,
            "cards": self.cards,
            "prerequisites": self.prerequisites,
            "results": self.results,
            "steps": self.steps,
        }

    @property
    def url(self):
        return f"https://commanderspellbook.com/combo/{self.id}"


class MTGCombos:
    def __init__(self):
        self.combos = self._init_combo_data()

    def _init_combo_data(self):
        cached_combo_path = Path("./data/combo-data.json")
        if cached_combo_path.is_file():
            data = json.loads(cached_combo_path.read_text())
            return [ComboData(combo_data) for combo_data in data]
        COMBO_URL = "https://commanderspellbook.com/api/combo-data.json"
        response = requests.get(COMBO_URL)
        response.raise_for_status()
        data = response.json()
        cached_combo_path.write_text(json.dumps(data))
        return [ComboData(combo_data) for combo_data in data]

    def __iter__(self):
        yield from self.combos

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame([c.serialize() for c in self])
