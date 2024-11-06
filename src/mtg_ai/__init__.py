import mtg_ai.ai as ai
import mtg_ai.cards as cards
import mtg_ai.constants as constants
from mtg_ai.cli import cli_main as _cli_main  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "cards",
    "constants",
    "ai",
]
