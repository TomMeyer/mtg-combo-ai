import logging as _logging

import mtg_ai.ai as ai
import mtg_ai.cards as cards
import mtg_ai.constants as constants

_logging.basicConfig()
_logging.getLogger("mtg-ai").addHandler(_logging.NullHandler())

__version__ = "0.1.0"

__all__ = [
    "cards",
    "constants",
    "ai",
]
