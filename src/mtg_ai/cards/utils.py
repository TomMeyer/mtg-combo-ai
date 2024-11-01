from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from mtg_ai.constants import ColorIdentity

MTG_CACHE_DIR = Path.home().joinpath(".cache", "mtg-ai")


def strip_brackets(x):
    if x is not None:
        x = x.replace("[", "").replace("]", "")
        x = x.replace("{", "").replace("}", "")
        x = x.replace("(", "").replace(")", "")
        return x
    return x


def strip_commas(x):
    if x is not None:
        return x.replace(",", "").replace(" ", "")
    return x


def sort_colors(x):
    if x is not None:
        wubrg_order = {"W": 0, "U": 1, "B": 2, "R": 3, "G": 4, "C": 5}
        sorted_mtg_colors = "".join(sorted(x, key=lambda v: wubrg_order[v]))
        return sorted_mtg_colors
    return x


def color_to_identity(x) -> str:
    if x is None or x == "":
        x = "C"
    return ColorIdentity(x).name.title().replace("_", " ")


T = TypeVar("T")
R = TypeVar("R")


class classproperty(Generic[T, R]):
    def __init__(self, func: Callable[[type[T]], R]) -> None:
        self.func = func

    def __get__(self, obj: Any, cls: type[T]) -> R:
        return self.func(cls)
