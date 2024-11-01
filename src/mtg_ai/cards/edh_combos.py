import json
import logging
from typing import Any, Dict, Iterator, Optional, TypedDict

import pandas as pd
import requests
from tqdm.auto import tqdm

from mtg_ai.cards.utils import MTG_CACHE_DIR
from mtg_ai.utils import is_tqdm_disabled

MTG_COMBOS_FILE = MTG_CACHE_DIR.joinpath("commander_spellbook_combos.json")

logger = logging.getLogger(__name__)


class Combo(TypedDict):
    combo: pd.Series | pd.DataFrame
    cards: pd.Series | pd.DataFrame
    features: pd.Series | pd.DataFrame
    requires: Optional[pd.Series | pd.DataFrame]


def _get_mtg_combos():
    logger.info("Fetching combos json")
    COMMANDER_SPELLBOOK_URL = "https://json.commanderspellbook.com/variants.json"

    if not MTG_CACHE_DIR.exists():
        MTG_CACHE_DIR.mkdir(parents=True)

    response = requests.get(COMMANDER_SPELLBOOK_URL, stream=True)
    data = response.json()
    MTG_COMBOS_FILE.write_text(json.dumps(data))


def build_combo_df(combo_data: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    main_data = []
    cards = []
    requires = []
    combo_features = []

    for data in tqdm(
        combo_data, desc="Building combo data", disable=is_tqdm_disabled()
    ):
        combo = {
            "combo_id": data["id"],
            "manaNeeded": data["manaNeeded"],
            "manaValueNeeded": data["manaValueNeeded"],
            "identity": data["identity"],
            "variantCount": data["variantCount"],
            "otherPrerequisites": data["otherPrerequisites"],
            "steps": data["description"],
            "notes": data["notes"],
            "of": ",".join([str(d["id"]) for d in data["of"]]),
            "includes": ",".join([str(d["id"]) for d in data["includes"]]),
        }
        main_data.append(combo)
        for use in data["uses"]:
            card_id = use["card"]["oracleId"]
            card_name = use["card"]["name"]
            zone_locations = use["zoneLocations"]
            battlefield_card_state = use["battlefieldCardState"]
            exile_card_state = use["exileCardState"]
            library_card_state = use["libraryCardState"]
            must_be_commander = use["mustBeCommander"]
            quantity = use["quantity"]
            card_data = {
                "combo_id": data["id"],
                "card_id": card_id,
                "card_name": card_name,
                "zone_locations": tuple(zone_locations),
                "battlefield_card_state": battlefield_card_state,
                "exile_card_state": exile_card_state,
                "library_card_state": library_card_state,
                "must_be_commander": must_be_commander,
                "quantity": quantity,
            }
            cards.append(card_data)

        for require in data["requires"]:
            require_data = {
                "combo_id": data["id"],
                "require_id": require["template"]["id"],
                "require_name": require["template"]["name"],
                "zone_locations": require["zoneLocations"],
                "battlefield_card_state": require["battlefieldCardState"],
                "exile_card_state": require["exileCardState"],
                "library_card_state": require["libraryCardState"],
                "graveyard_card_state": require["graveyardCardState"],
                "must_be_commander": require["mustBeCommander"],
                "quantity": require["quantity"],
            }
            requires.append(require_data)

        for produce in data["produces"]:
            feature_data = {
                "combo_id": data["id"],
                "feature_id": produce["feature"]["id"],
                "feature_name": produce["feature"]["name"],
                "uncountable": produce["feature"]["uncountable"],
                "quantity": produce["quantity"],
            }
            combo_features.append(feature_data)

    df = pd.DataFrame(main_data)
    df_features = pd.DataFrame(combo_features)
    df_requires = pd.DataFrame(requires)
    df_cards = pd.DataFrame(cards)
    df.set_index("combo_id", inplace=True)
    df_features.set_index(["combo_id", "feature_id"], inplace=True)
    df_requires.set_index(["combo_id", "require_id"], inplace=True)
    df_cards.set_index(["combo_id", "card_id"], inplace=True)
    return {
        "combos": df,
        "features": df_features,
        "requires": df_requires,
        "cards": df_cards,
    }


def load_combo_data() -> Dict[str, pd.DataFrame]:
    combo_data = json.loads(MTG_COMBOS_FILE.read_text())
    combo_data = combo_data["variants"]

    combo_dfs = build_combo_df(combo_data)

    return combo_dfs


class EDHComboDatabase:
    def __init__(self):
        if not MTG_COMBOS_FILE.exists():
            _get_mtg_combos()
        self._data = load_combo_data()

    @property
    def combos(self) -> pd.DataFrame:
        return self._data["combos"]

    @property
    def features(self) -> pd.DataFrame:
        return self._data["features"]

    @property
    def requires(self) -> pd.DataFrame:
        return self._data["requires"]

    @property
    def cards(self) -> pd.DataFrame:
        return self._data["cards"]

    @property
    def index(self) -> list[str]:
        return self.combos.index.to_list()

    def __getitem__(self, combo_id: str) -> Combo:
        return self.get_combo(combo_id)

    def __iter__(self) -> Iterator[Combo]:
        for combo_id in self.index:
            yield self[combo_id]

    def get_combo(self, combo_id: str) -> Combo:
        return {
            "combo": self.combos.loc[combo_id],
            "features": self.get_features_for_combo(combo_id),
            "requires": self.get_requires_for_combo(combo_id),
            "cards": self.get_cards_for_combo(combo_id),
        }

    def get_features_for_combo(self, combo_id: str) -> pd.Series | pd.DataFrame:
        return self.features.loc[combo_id]

    def get_requires_for_combo(
        self, combo_id: str
    ) -> Optional[pd.Series | pd.DataFrame]:
        try:
            return self.requires.loc[combo_id]
        except KeyError:
            return None

    def get_cards_for_combo(self, combo_id: str) -> pd.Series | pd.DataFrame:
        return self.cards.loc[combo_id]
