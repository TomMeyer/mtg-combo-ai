from enum import StrEnum
from typing import Final

MTG_STANDARD_COLOR_ORDER: Final = "WUBRGC"


class MTGColorCombo(StrEnum):
    COLORLESS = ""  # No color
    MONO_WHITE = "W"  # White
    MONO_BLUE = "U"  # Blue
    MONO_BLACK = "B"  # Black
    MONO_RED = "R"  # Red
    MONO_GREEN = "G"  # Green
    AZORIUS = "WU"  # White, Blue
    DIMIR = "UB"  # Blue, Black
    RAKDOS = "BR"  # Black, Red
    GRUUL = "RG"  # Red, Green
    SELESNYA = "WG"  # Green, White
    ORZHOV = "WB"  # White, Black
    IZZET = "UR"  # Blue, Red
    GOLGARI = "BG"  # Black, Green
    BOROS = "WR"  # White, Red
    SIMIC = "UG"  # Blue, Green
    ESPER = "WUB"  # White, Blue, Black
    GRIXIS = "UBR"  # Blue, Black, Red
    JUND = "BRG"  # Black, Red, Green
    NAYA = "WRG"  # White, Red, Green
    BANT = "WUG"  # White, Blue, Green
    ABZAN = "WBG"  # White, Black, Green
    JESKAI = "WUR"  # White, Blue, Red
    SULTAI = "UBG"  # Blue, Black, Green
    MARDU = "WBR"  # White, Black, Red
    TEMUR = "URG"  # Blue, Red, Green
    YIDRIS = "UBRG"  # Blue, Black, Red, Green
    SASKIA = "WBRG"  # White, Black, Red, Green
    KYNAIOS = "WURG"  # White, Blue, Red, Green
    ATRAXA = "WUBG"  # White, Blue, Black, Green
    BREYA = "WUBR"  # White, Blue, Black, Red
    FIVE_COLOR = "WUBRG"  # White, Blue, Black, Red, Green

    @staticmethod
    def _sort_multicolor_str(s: str):
        try:
            return "".join(
                sorted(s, key=lambda x: MTG_STANDARD_COLOR_ORDER.index(x.upper()))
            )
        except:
            return s

    @classmethod
    def from_str(cls, s: str):
        s = cls._sort_multicolor_str(s.upper())
        return cls(s)


drop_columns = [
    "life_modifier",
    "hand_modifier",
    "attraction_lights",
    "object",
    "multiverse_ids",
    "mtgo_id",
    "mtgo_foil_id",
    "tcgplayer_id",
    "cardmarket_id",
    "uri",
    "scryfall_uri",
    "highres_image",
    "image_status",
    "image_uris",
    "reserved",
    "foil",
    "nonfoil",
    "finishes",
    "oversized",
    "promo",
    "reprint",
    "variation",
    "set_uri",
    "set_search_uri",
    "scryfall_set_uri",
    "prints_search_uri",
    "flavor_text",
    "artist_ids",
    "illustration_id",
    "border_color",
    "frame",
    "textless",
    "booster",
    "story_spotlight",
    "related_uris",
    "purchase_uris",
    "security_stamp",
    "preview",
    "penny_rank",
    "frame_effects",
    "watermark",
    "card_faces",
    "tcgplayer_etched_id",
    "promo_types",
    "prices",
    "artist",
    "digital",
    "games",
    "set_id",
    "set",
    "set_name",
    "collector_number",
    "full_art",
    "all_parts",
    "arena_id",
    "released_at",
    "content_warning",
    "card_back_id",
    "lang",
    "id",
    "rulings_uri",
]

column_order = [
    "oracle_id",
    "card_name",
    "type_line",
    "rarity",
    "mana_cost",
    "cmc",
    "colors",
    "color_identity",
    "power",
    "toughness",
    "loyalty",
    "produced_mana",
    "keywords",
    "set_type",
    "oracle_text",
    "layout",
    "edhrec_rank",
    "color_indicator",
]

PAD_TOKEN = "[PAD]"

INPUT_IDS: Final = "input_ids"
ATTENTION_MASK: Final = "attention_mask"
CATEGORY: Final = "category"
LABELS: Final = "labels"

NAN_STRING: Final = "<NaN>"
NA_STRING: Final = "<N/A>"
