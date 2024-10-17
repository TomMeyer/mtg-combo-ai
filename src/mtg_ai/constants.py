import typing
from enum import StrEnum
from typing import Final, Literal

MTG_STANDARD_COLOR_ORDER: Final = "WUBRGC"

MTG_COLOR = Literal[
    "W", "U", "B", "R", "G", "C", "Green", "Red", "Blue", "White", "Black", "Colorless"
]


class Color(StrEnum):
    WHITE = "W"
    BLUE = "U"
    BLACK = "B"
    RED = "R"
    GREEN = "G"
    COLORLESS = "C"

    @classmethod
    def get(cls, name: MTG_COLOR | str) -> "Color":
        n = name.upper()

        try:
            return Color[n]
        except KeyError:
            pass

        try:
            return Color(n)
        except ValueError:
            pass

        raise ValueError(f"Invalid color: {name}")


mtg_colors: tuple[str, ...] = typing.get_args(MTG_COLOR)


COLOR_IDENTITY = Literal[
    "W",
    "U",
    "B",
    "R",
    "G",
    "C",
    "WU",
    "WB",
    "WR",
    "UB",
    "UR",
    "WG",
    "BR",
    "UG",
    "BG",
    "RG",
    "WUB",
    "WUR",
    "WUG",
    "WBR",
    "WBG",
    "UBR",
    "WRG",
    "UBG",
    "URG",
    "BRG",
    "WUBR",
    "WUBG",
    "WURG",
    "WBRG",
    "UBRG",
    "WUBRG",
    "Mono White",
    "Mono Blue",
    "Mono Black",
    "Mono Red",
    "Mono Green",
    "Azorius",
    "Orzhov",
    "Boros",
    "Dimir",
    "Izzet",
    "Selesnya",
    "Rakdos",
    "Simic",
    "Golgari",
    "Gruul",
    "Esper",
    "Jeskai",
    "Bant",
    "Mardu",
    "Abzan",
    "Grixis",
    "Naya",
    "Sultai",
    "Temur",
    "Jund",
    "Yore-Tiller",
    "Witch-Maw",
    "Ink-Treader",
    "Dune",
    "Glint-Eye",
    "Five-Color",
    "Colorless",
]


class ColorIdentity(StrEnum):
    # Mono-color
    MONO_WHITE = "W"
    MONO_BLUE = "U"
    MONO_BLACK = "B"
    MONO_RED = "R"
    MONO_GREEN = "G"

    # Two-color
    AZORIUS = "WU"
    ORZHOV = "WB"
    BOROS = "WR"
    DIMIR = "UB"
    IZZET = "UR"
    SELESNYA = "WG"
    RAKDOS = "BR"
    SIMIC = "UG"
    GOLGARI = "BG"
    GRUUL = "RG"
    ESPER = "WUB"
    JESKAI = "WUR"

    # Three-color
    BANT = "WUG"
    MARDU = "WBR"
    ABZAN = "WBG"
    GRIXIS = "UBR"
    NAYA = "WRG"
    SULTAI = "UBG"
    TEMUR = "URG"
    JUND = "BRG"

    # Four-color
    YORE_TILLER = "WUBR"
    WITCH_MAW = "WUBG"
    INK_TREADER = "WURG"
    DUNE = "WBRG"
    GLINT_EYE = "UBRG"

    FIVE_COLOR = "WUBRG"

    COLORLESS = "C"

    @classmethod
    def get(cls, name: COLOR_IDENTITY | str) -> "ColorIdentity":
        name = name.upper().replace("-", "_").replace(" ", "")

        try:
            return ColorIdentity[name]
        except KeyError:
            pass

        try:
            return ColorIdentity(name)
        except ValueError:
            pass

        raise ValueError(f"Invalid color identity: {name}")


MTG_CARD_TYPE = Literal[
    "Land",
    "Creature",
    "Enchantment",
    "Artifact",
    "Planeswalker",
    "Instant",
    "Sorcery",
    "Battle",
    "Tribal",
    "Legendary",
]


class CardType(StrEnum):
    LAND = "Land"
    CREATURE = "Creature"
    ENCHANTMENT = "Enchantment"
    ARTIFACT = "Artifact"
    PLANESWALKER = "Planeswalker"
    INSTANT = "Instant"
    SORCERY = "Sorcery"
    BATTLE = "Battle"
    TRIBAL = "Tribal"

    @classmethod
    def get(cls, name: MTG_CARD_TYPE | str) -> "CardType":
        name = name.capitalize()

        try:
            return CardType[name.upper()]
        except KeyError:
            pass

        try:
            return CardType(name)
        except ValueError:
            pass

        raise ValueError(f"Invalid card type: {name}")


CATEGORY: Final = "category"
LABELS: Final = "labels"

NAN_STRING: Final = "<NaN>"
NA_STRING: Final = "<N/A>"
