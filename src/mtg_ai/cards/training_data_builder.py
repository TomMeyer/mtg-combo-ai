import inspect
import json
import sys
import time
from pathlib import Path
from typing import Callable, Final

import datasets
import tqdm

from mtg_ai.cards.database import MTGDatabase

QUESTION_ANSWER_FILE = Path.home().joinpath(
    ".cache", "mtg-ai", "question_answer_dataset.json"
)

DELIMITER: Final = "|"
HEADER = "Question|Answer\n"


class DataFormat:
    def __init__(self, role: str, content: str):
        self.from_ = role
        self.value = content

    def to_dict(self) -> dict[str, str]:
        return {"role": self.from_, "content": self.value}


class DataEntry:
    def __init__(self, question: str, answer: str):
        self.human = DataFormat("user", question)
        self.gpt = DataFormat("assistant", answer)

    def to_json(self) -> list[dict[str, str]]:
        return [self.human.to_dict(), self.gpt.to_dict()]


def get_builder_funcs() -> list[Callable[[MTGDatabase], None]]:
    return [
        func
        for name, func in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
        if "build_" in name and "build_question_answer_datasets" != name
    ]


def build_question_answer_datasets(data: MTGDatabase, recreate: bool = False):
    if not QUESTION_ANSWER_FILE.exists() or recreate:
        QUESTION_ANSWER_FILE.write_text("Question|Answer\n")
    start_time = time.time()
    builder_functions: list[Callable[[MTGDatabase]]] = get_builder_funcs()

    entries: list[DataEntry] = []
    for func in tqdm.tqdm(builder_functions):
        entries.extend(func(data))

    output_data = json.dumps([entry.to_json() for entry in entries], indent=2)
    QUESTION_ANSWER_FILE.write_text(output_data)

    file_size = QUESTION_ANSWER_FILE.stat().st_size / 1024 / 1024

    print(
        (
            "Finished creating question answer dataset in "
            f"{time.time() - start_time} seconds | file size: {file_size:.2f} MB"
        )
    )


def build_card_name_to_cmc_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    for _, row in data.df[["name", "cmc"]].iterrows():
        question = f"What is the converted mana cost of {row['name']}?"

        answer = f"{row['cmc']}"
        entry = DataEntry(question=question, answer=answer)
        result.append(entry)
    return result


def build_card_name_to_color_identity_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    df = data.df.loc[data.df.colorIdentity != "", ["name", "colorIdentity"]]
    for _, row in df.iterrows():
        question = f"What is the color identity of {row['name']}?"
        answer = "{row['colorIdentity']}"
        result.append(DataEntry(question=question, answer=answer))
    return result


def build_card_name_to_ascii_name_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    df = data.df.loc[data.df.asciiName.notna(), ["name", "asciiName"]]
    for _, row in df[["name", "asciiName"]].iterrows():
        question = f"What is the ascii name of {row['name']}?"
        answer = f"{row['asciiName']}"
        result.append(DataEntry(question=question, answer=answer))
    return result


def build_card_name_to_type_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    for _, row in data.df[["name", "type"]].iterrows():
        question = f"What is the type of {row['name']}?"
        answer = f"{row['type']}"
        result.append(DataEntry(question=question, answer=answer))
    return result


def build_card_name_to_power_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    df = data.df.loc[data.df.power.notna(), ["name", "power"]]
    for _, row in df[["name", "power"]].iterrows():
        question = f"What is the power of {row['name']}?"
        answer = f"{row['power']}"
        result.append(DataEntry(question=question, answer=answer))
    for _, row in data.df.loc[data.df.power.isna(), ["name"]].iterrows():
        question = f"What is the power of {row['name']}?"
        answer = f"{row['name']} has no power"
        result.append(DataEntry(question=question, answer=answer))
    return result


def build_card_name_to_toughness_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    df = data.df.loc[data.df.toughness.notna(), ["name", "toughness"]]
    for _, row in df[["name", "toughness"]].iterrows():
        question = f"What is the toughness of {row['name']}?"
        answer = f"{row['toughness']}"
        result.append(DataEntry(question=question, answer=answer))
    for _, row in data.df.loc[data.df.toughness.isna(), ["name"]].iterrows():
        question = f"What is the toughness of {row['name']}?"
        answer = f"{row['name']} has no toughness"
        result.append(DataEntry(question=question, answer=answer))
    return result


def build_card_with_cmc_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    for cmc in data.cmcs:
        for row in data.get_cards_by_cmc(cmc)[["name", "cmc"]].iterrows():
            question = f"What is a card with converted mana cost (cmc) {row[1]['cmc']}?"
            answer = f"{row[1]['name']}"
            result.append(DataEntry(question=question, answer=answer))
    return result


def load_mtg_dataset() -> datasets.Dataset:
    data = json.loads(QUESTION_ANSWER_FILE.read_text())
    data = {"conversations": data}
    dataset = datasets.Dataset.from_dict(data)
    if not Path("./data/question_answer_dataset.json").exists():
        dataset.save_to_disk("./data/question_answer_dataset.json")
    return dataset


def read_mtg_dataset_from_disk() -> datasets.Dataset:
    return datasets.load_from_disk("./data/question_answer_dataset.json")  # type: ignore
