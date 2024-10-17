import logging as _logging


import mtg_ai.cards as cards
import mtg_ai.constants as constants
import mtg_ai.ai as ai


_logging.basicConfig()
_logging.getLogger("mtg-ai").addHandler(_logging.NullHandler())

__all__ = ["cards", "constants", "ai"]
