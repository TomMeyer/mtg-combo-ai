import mtg_ai.constants as constants
from mtg_ai.constants import MTGColorCombo, MTG_STANDARD_COLOR_ORDER
import mtg_ai.data as data
import mtg_ai.ai as ai

import logging as _logging

_logging.basicConfig(level=_logging.INFO)

__all__ = [
    "MTGColorCombo",
    "MTG_STANDARD_COLOR_ORDER",
    "data",
    "training",
    "constants",
    "ai",
]
