import logging
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Callable, Final, TypedDict

from datasets import Dataset
from tqdm.auto import tqdm

from mtg_ai.cards.database import MTGDatabase
from mtg_ai.cards.edh_combos import EDHComboDatabase
from mtg_ai.constants import PathLike
from mtg_ai.utils import is_tqdm_disabled

logger = logging.getLogger(__name__)

QUESTION_ANSWER_FOLDER = Path.home().joinpath(".cache", "mtg-ai", "training_data")

DELIMITER: Final = "|"
HEADER = "Question|Answer\n"


class DataFormat(TypedDict):
    role: str
    content: str


class DataEntry:
    def __init__(self, question: str, answer: str):
        self.gpt: DataFormat = {"role": "assistant", "content": answer}
        self.human: DataFormat = {"role": "user", "content": question}

    def to_json(self) -> list[DataFormat]:
        return [self.human, self.gpt]


class MTGDatasetBuilder:
    """
    A class used to build datasets for Magic: The Gathering (MTG) cards.

    Attributes
    ----------
    database : MTGDatabase
        An instance of MTGDatabase used to get card information.
    registered_functions : dict[str, list[Callable[[MTGDatabase], list[DataEntry]]]]
        A dictionary that groups functions that build question-answer datasets.


    Methods
    -------
    register(group: str)
        A class method that is used as a decorator to register functions to a group for building question-answer datasets.
    build_question_answer_datasets() -> None
        A class method that builds question-answer datasets.
        If the dataset folder does not exist it creates the folder and then generates the datasets.
    """

    database = MTGDatabase()
    registered_functions: dict[str, list[Callable[[MTGDatabase], list[DataEntry]]]] = (
        defaultdict(list)
    )
    group_order: dict[str, int] = {}

    def __init__(self) -> None:
        raise NotImplementedError(
            "MTGDatasetBuilder is a static class and should not be instantiated."
        )

    @classmethod
    def register(cls, group: str, train_order: int):
        """
        Used as a decorator to register a question-answer dataset building function to a group.

        Args:
            group (str): The group under which the function should be registered.
            train_order (int): The order in which the function should be run, must be the same for all functions in the group.

        Returns:
            Callable: A decorator that registers the given function and returns it.

        Decorator Args:
            func (Callable[[MTGDatabase], list[DataEntry]]): The function to be registered.
        """

        @wraps(cls.register)
        def decorator(func: Callable[[MTGDatabase], list[DataEntry]]):
            if group not in cls.group_order:
                cls.group_order[group] = train_order
            elif any(
                order == train_order
                for group_name, order in cls.group_order.items()
                if group_name != group
            ):
                raise ValueError(
                    f"Train order {train_order} is already used by another group"
                )
            elif cls.group_order[group] != train_order:
                raise ValueError(
                    f"Group {group} has conflicting train orders {cls.group_order[group]} and {train_order}"
                )
            cls.registered_functions[group].append(func)
            return func

        return decorator

    @classmethod
    def build_question_answer_datasets(
        cls, directory: PathLike = QUESTION_ANSWER_FOLDER
    ) -> None:
        """
        Builds question-answer datasets and saves them as JSON files.

        This method iterates over registered builder functions, generates question-answer
        data entries, and writes them to JSON files in the specified folder. If the folder
        does not exist it will be created.

        If the `recreate` flag is set to True, the files will be created.

        Args:
            recreate (bool): If True, the question-answer folder will be recreated. Defaults to False.

        Returns:
            None
        """
        directory = Path(directory)

        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
        for group, builder_functions in tqdm(
            cls.registered_functions.items(),
            desc="Building datasets",
        ):
            start_time = time.time()
            question_answer_file = directory.joinpath(f"{group}_question_answer")

            entries: list[DataEntry] = []
            for func in tqdm(builder_functions, desc=f"Running {group} builders"):
                entries.extend(func(cls.database))
            output = {"conversations": [entry.to_json() for entry in entries]}
            dataset = Dataset.from_dict(output)
            dataset.save_to_disk(question_answer_file)
            if dataset.size_in_bytes:
                file_size = dataset.size_in_bytes / 1024 / 1024
                logger.info(
                    (
                        "Finished creating question answer dataset in "
                        f"{time.time() - start_time} seconds | file size: {file_size:.2f} MB"
                    )
                )
            else:
                logger.info(
                    f"Finished creating question answer dataset in {time.time() - start_time} seconds"
                )

    # @classmethod
    # def build_question_answer_datasets_single(
    #     cls, directory: PathLike = QUESTION_ANSWER_FOLDER
    # ) -> None:
    #     start_time = time.time()
    #     directory = Path(directory)
    #     directory.mkdir(parents=True, exist_ok=True)
    #     question_answer_file = directory.joinpath("all_question_answer.json")
    #     output_data = []
    #     entries: list[DataEntry] = []
    #     for group, builder_functions in tqdm(
    #         cls.registered_functions.items(),
    #         desc="Building datasets",
    #     ):
    #         start_time = time.time()

    #         for func in tqdm(builder_functions, desc=f"Running {group} builders"):
    #             entries.extend(func(cls.database))

    #     output = {"conversations": [entry.to_json() for entry in entries]}
    #     output_data = json.dumps(output, indent=2)

    #     question_answer_file.write_text(output_data)

    #     file_size = question_answer_file.stat().st_size / 1024 / 1024
    #     logger.info(
    #         (
    #             "Finished creating question answer dataset in "
    #             f"{time.time() - start_time} seconds | file size: {file_size:.2f} MB"
    #         )
    #     )


def build_datasets(
    directory: PathLike = QUESTION_ANSWER_FOLDER,  # all_merged: bool = False
) -> None:
    """
    Helper function to run the question-answer datasets for Magic: The Gathering (MTG) cards.
    """
    # if all_merged:
    # MTGDatasetBuilder.build_question_answer_datasets_single(directory=directory)
    MTGDatasetBuilder.build_question_answer_datasets(directory=directory)


@MTGDatasetBuilder.register(group="cards", train_order=0)
def build_card_name_to_manacost_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result = []
    df = data.df.loc[data.df.manaCost.notna(), ["name", "manaCost", "side"]]
    for _, row in tqdm(
        df.iterrows(),
        desc="Building mana cost dataset",
        leave=False,
        total=len(df),
        disable=is_tqdm_disabled(),
    ):
        name = row["name"]
        if row["side"] == "a" and "//" in row["name"]:
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b" and "//" in row["name"]:
            name = row["name"].split(" // ")[1]
        questions = [
            f"What is the mana cost of {name}?",
            f"Can you tell me the mana cost of {name}?",
            f"What's the card mana cost for {name}?",
        ]
        answer = f"{row['manaCost']}"
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    for _, row in data.df.loc[
        data.df.manaCost.notna(), ["name", "type", "manaCost", "side"]
    ].iterrows():
        name = row["name"]
        if row["side"] == "a" and "//" in row["name"]:
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b" and "//" in row["name"]:
            name = row["name"].split(" // ")[1]
        questions = [
            f"What is the mana cost of {name}?",
            f"Can you tell me the mana cost of {name}?",
            f"What's the card mana cost for {name}?",
        ]
        answer = f"{name} has no mana cost"
        if "Land" in row["type"]:
            answer = f"{name} is a land and land cards have no mana cost"
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    return result


@MTGDatasetBuilder.register(group="cards", train_order=0)
def build_card_name_to_cmc_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    df = data.df.loc[data.df.cmc.notna(), ["name", "cmc", "side"]]
    for _, row in tqdm(
        df.iterrows(),
        desc="Building cmc dataset",
        leave=False,
        total=len(df),
        disable=is_tqdm_disabled(),
    ):
        name = row["name"]
        if row["side"] == "a" and "//" in row["name"]:
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b" and "//" in row["name"]:
            name = row["name"].split(" // ")[1]
        questions = [
            f"What is the converted mana cost of {name}?",
            f"Can you tell me the converted mana cost for {name}?",
            f"How much is the converted mana cost of {name}?",
        ]
        answer = f"{row['cmc']}"
        for question in questions:
            entry = DataEntry(question=question, answer=answer)
            result.append(entry)
    return result


@MTGDatasetBuilder.register(group="cards", train_order=0)
def build_card_name_to_keywords_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result = []
    df = data.df.loc[data.df.keywords.notna(), ["name", "keywords", "side"]]
    for _, row in tqdm(
        df.iterrows(),
        desc="Building keywords dataset",
        leave=False,
        total=len(df),
        disable=is_tqdm_disabled(),
    ):
        name = row["name"]
        if row["side"] == "a" and "//" in row["name"]:
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b" and "//" in row["name"]:
            name = row["name"].split(" // ")[1]
        questions = [
            f"What are the keywords for {name}?",
            f"Can you tell me the keywords of {name}?",
            f"What keywords are associated with {name}?",
        ]
        answer = f"{row['keywords']}"
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    return result


@MTGDatasetBuilder.register(group="cards", train_order=0)
def build_card_name_to_color_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    df = data.df.loc[data.df.colors.notna(), ["name", "colors", "side"]]
    for _, row in tqdm(
        df.iterrows(),
        desc="Building color dataset",
        leave=False,
        total=len(df),
        disable=is_tqdm_disabled(),
    ):
        name = row["name"]
        if row["side"] == "a" and "//" in row["name"]:
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b" and "//" in row["name"]:
            name = row["name"].split(" // ")[1]
        questions = [
            f"What is the color of {name}?",
            f"Can you tell me the color of {name}?",
            f"What color is {name}?",
        ]
        answer = DELIMITER.join(row["colors"])
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    return result


@MTGDatasetBuilder.register(group="cards", train_order=0)
def build_card_name_to_color_identity_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    df = data.df.loc[data.df.colorIdentity != "", ["name", "colorIdentity", "side"]]
    for _, row in tqdm(
        df.iterrows(),
        desc="Building color identity dataset",
        leave=False,
        total=len(df),
        disable=is_tqdm_disabled(),
    ):
        name = row["name"]
        if row["side"] == "a" and "//" in row["name"]:
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b" and "//" in row["name"]:
            name = row["name"].split(" // ")[1]
        questions = [
            f"What is the color identity of {name}?",
            f"Can you tell me the color identity of {name}?",
            f"What colors are included in {name}'s color identity?",
        ]
        answer = f"{row['colorIdentity']}"
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    return result


@MTGDatasetBuilder.register(group="cards", train_order=0)
def build_card_name_to_ascii_name_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    df = data.df.loc[data.df.asciiName.notna(), ["name", "asciiName", "side"]]
    for _, row in tqdm(
        df.iterrows(),
        desc="Building ascii name dataset",
        leave=False,
        total=len(df),
        disable=is_tqdm_disabled(),
    ):
        name = row["name"]
        if row["side"] == "a" and "//" in row["name"]:
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b" and "//" in row["name"]:
            name = row["name"].split(" // ")[1]
        questions = [
            f"What is the ascii name of {name}?",
            f"What is the plain text version of the name for {name}?",
            f"Provide a simplified name for {name}.",
        ]
        for question in questions:
            answer = f"{row['asciiName']}"
            result.append(DataEntry(question=question, answer=answer))
    return result


@MTGDatasetBuilder.register(group="cards", train_order=0)
def build_card_name_to_type_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    df = data.df.loc[data.df.type.notna(), ["name", "type", "side"]]
    for _, row in tqdm(
        df.iterrows(),
        desc="Building type dataset",
        leave=False,
        total=len(df),
        disable=is_tqdm_disabled(),
    ):
        name = row["name"]
        if row["side"] == "a" and "//" in row["name"]:
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b" and "//" in row["name"]:
            name = row["name"].split(" // ")[1]
        questions = [
            f"What is the type of {name}?",
            f"Can you tell me the type of {name}?",
            f"What type is {name}?",
        ]
        answer = f"{row['type']}"
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    return result


@MTGDatasetBuilder.register(group="cards", train_order=0)
def build_card_name_to_power_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    df = data.df.loc[data.df.power.notna(), ["name", "power", "side"]]
    for _, row in tqdm(
        df.iterrows(),
        desc="Building power dataset",
        leave=False,
        total=len(df),
    ):
        name = row["name"]
        if row["side"] == "a" and "//" in row["name"]:
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b" and "//" in row["name"]:
            name = row["name"].split(" // ")[1]
        questions = [
            f"What is the power of {name}?",
            f"Can you tell me the power of {name}?",
            f"How much power does {name} have?",
        ]
        answer = f"{row['power']}"
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    df = data.df.loc[data.df.power.isna(), ["name", "power", "side"]]
    for _, row in tqdm(
        df.iterrows(),
        desc="Building power dataset",
        leave=False,
        total=len(df),
        disable=is_tqdm_disabled(),
    ):
        name = row["name"]
        if row["side"] == "a" and "//" in row["name"]:
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b" and "//" in row["name"]:
            name = row["name"].split(" // ")[1]
        questions = [
            f"What is the power of {name}?",
            f"Can you tell me the power of {name}?",
            f"How much power does {name} have?",
        ]
        answer = f"{name} has no power"
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    return result


@MTGDatasetBuilder.register(group="cards", train_order=0)
def build_card_name_to_text_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    df = data.df.loc[data.df.text.notna(), ["name", "text", "side"]]
    for _, row in tqdm(
        df.iterrows(),
        desc="Building text dataset",
        leave=False,
        total=len(df),
        disable=is_tqdm_disabled(),
    ):
        name = row["name"]
        text = row["text"]
        if row["side"] == "a" and "//" in row["name"]:
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b" and "//" in row["name"]:
            name = row["name"].split(" // ")[1]
        questions = [
            f"What is the text of {name}?",
            f"What are the abilities of {name}?",
            f"Can you tell me the text of {name}?",
            f"What does {name} do?",
        ]
        answer = text
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    return result


@MTGDatasetBuilder.register(group="cards", train_order=0)
def build_card_name_to_loyalty_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result = []
    df = data.df.loc[data.df.loyalty.notna(), ["name", "loyalty", "side"]]
    for _, row in tqdm(
        df.iterrows(),
        desc="Building loyalty dataset",
        leave=False,
        total=len(df),
        disable=is_tqdm_disabled(),
    ):
        name = row["name"]
        if row["side"] == "a" and "//" in row["name"]:
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b" and "//" in row["name"]:
            name = row["name"].split(" // ")[1]
        questions = [
            f"What is the loyalty of {name}?",
            f"Can you tell me the loyalty of {name}?",
            f"What is {name}'s loyalty?",
        ]
        answer = f"{row['loyalty']}"
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    return result


@MTGDatasetBuilder.register(group="cards", train_order=0)
def build_card_name_to_rarity_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    df = data.df.loc[data.df.rarity.notna(), ["name", "rarity", "side"]]
    for _, row in tqdm(
        df.iterrows(),
        desc="Building rarity dataset",
        leave=False,
        total=len(df),
        disable=is_tqdm_disabled(),
    ):
        name = row["name"]
        if row["side"] == "a" and "//" in row["name"]:
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b" and "//" in row["name"]:
            name = row["name"].split(" // ")[1]
        questions = [
            f"What is the rarity of {name}?",
            f"Can you tell me the rarity of {name}?",
            f"What rarity is {name}?",
        ]
        answer = f"{row['rarity']}"
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    return result


@MTGDatasetBuilder.register(group="cards", train_order=0)
def build_card_name_to_toughness_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []

    df = data.df.loc[data.df.toughness.notna(), ["name", "toughness", "side"]]
    for _, row in tqdm(
        df.iterrows(),
        desc="Building toughness dataset",
        leave=False,
        total=len(df),
        disable=is_tqdm_disabled(),
    ):
        name = row["name"]
        if row["side"] == "a" and "//" in row["name"]:
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b" and "//" in row["name"]:
            name = row["name"].split(" // ")[1]
        questions = [
            f"What is the toughness of {name}?",
            f"Can you tell me the toughness of {name}?",
            f"What is {name}'s toughness?",
        ]
        answer = f"{row['toughness']}"
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    df = data.df.loc[data.df.toughness.isna(), ["name", "side"]]
    for _, row in tqdm(
        df.iterrows(),
        desc="Building toughness dataset",
        leave=False,
        total=len(df),
        disable=is_tqdm_disabled(),
    ):
        name = row["name"]
        if row["side"] == "a" and "//" in row["name"]:
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b" and "//" in row["name"]:
            name = row["name"].split(" // ")[1]
        questions = [
            f"What is the toughness of {name}?",
            f"Can you tell me the toughness of {name}?",
            f"What is {name}'s toughness?",
        ]
        for question in questions:
            answer = f"{name} has no toughness"
            result.append(DataEntry(question=question, answer=answer))
    return result


@MTGDatasetBuilder.register(group="cards", train_order=0)
def build_card_to_rulings_question_answer_dataset(data: MTGDatabase) -> list[DataEntry]:
    result: list[DataEntry] = []
    df = data.df.loc[data.df.rulings.notna(), ["name", "rulings"]]
    for _, row in tqdm(
        df.iterrows(),
        desc="Building rulings dataset",
        leave=False,
        total=len(df),
        disable=is_tqdm_disabled(),
    ):
        questions = [
            f"What are the rulings for {row['name']} card?",
            f"Can you tell me the rulings for {row['name']}?",
            f"What are the official rulings for {row['name']}?",
        ]
        rulings = []
        for i, ruling in enumerate(row["rulings"].splitlines()):
            ruling_text = f"{i}. {ruling}"
            rulings.append(ruling_text)
        answer = "\n".join(rulings)
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    return result


@MTGDatasetBuilder.register(group="cards", train_order=0)
def build_card_with_cmc_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    for cmc in data.cmcs:
        df = data.get_cards_by_cmc(cmc)[["name", "cmc", "side"]]
        for _, row in tqdm(
            df.iterrows(),
            desc="Building cmc dataset",
            leave=False,
            total=len(df),
            disable=is_tqdm_disabled(),
        ):
            name = row["name"]
            if row["side"] == "a" and "//" in row["name"]:
                name = row["name"].split(" // ")[0]
            elif row["side"] == "b" and "//" in row["name"]:
                name = row["name"].split(" // ")[1]
                questions = [
                    f"What is a card with converted mana cost (cmc) {row['cmc']}?",
                    f"Can you name a card that has a converted mana cost of {row['cmc']}?",
                    f"Give me a card with a cmc of {row['cmc']}.",
                ]
                answer = name
                for question in questions:
                    result.append(DataEntry(question=question, answer=answer))
    return result


@MTGDatasetBuilder.register(group="cards", train_order=0)
def build_card_name_to_side_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []

    df = data.df.loc[data.df.side.notna(), ["name", "side"]]
    for _, row in tqdm(
        df.iterrows(),
        desc="Building card side dataset",
        leave=False,
        total=len(df),
        disable=is_tqdm_disabled(),
    ):
        name = row["name"]
        if row["side"] == "a" and "//" in row["name"]:
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b" and " // " in row["name"]:
            name = row["name"].split(" // ")[1]
        answer = row["side"]
        questions = [
            f"What card side is {name} on?",
            f"Can you tell me what side {name} is on?",
            f"What side is {name} on?",
        ]
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
        answer = name
        questions = [
            f"What is the name of side {row['side']} of {answer}?",
            f"Can you tell me the name of side {row['side']} of {answer}?",
            f"What is the name of {row['side']} side of {answer}?",
        ]
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    return result


@MTGDatasetBuilder.register(group="rules", train_order=1)
def build_card_layout_to_rule_question_answer_dataset(
    _: MTGDatabase,
) -> list[DataEntry]:
    question_answers = [
        (
            "What is a Transform card?",
            "Double-sided cards with two unique faces. These cards can 'transform' during gameplay, flipping to reveal a different face.",
        ),
        (
            "What is a Modal DFC?",
            "Double-sided, but instead of transforming, players choose one face to play initially, often offering options like a creature or a land.",
        ),
        (
            "What is an Adventure card?",
            "Cards with a split 'Adventure' option in the lower left, allowing a spell to be cast first before the main creature is played.",
        ),
        (
            "What is a Split card?",
            "Single-sided cards divided horizontally, with two spells or effects. Players choose one to play.",
        ),
        (
            "What is a Saga card?",
            "Vertical storytelling format with sequential 'chapters' that trigger effects each turn.",
        ),
        (
            "What is a Reversible card?",
            "Cards with gameplay options on either side, like Modal DFCs, but not intended to transform.",
        ),
        (
            "What is an Aftermath card?",
            "Cards split horizontally, allowing one spell to be cast first and another 'aftermath' spell later from the graveyard.",
        ),
        (
            "What is a Flip card?",
            "Single-sided cards with an upside-down second face on the bottom half, which is turned 180 degrees to activate.",
        ),
        (
            "What is a Mutate card?",
            "Standard creature cards with 'mutate' options, allowing them to be merged with other creatures on the battlefield.",
        ),
        (
            "What is a Leveler card?",
            "Single-sided card with a leveling mechanic, where players pay to 'level up,' enhancing power and abilities.",
        ),
        (
            "What is a Class card?",
            "Enchantment cards representing D&D classes, where players pay to progress through 'levels' for added effects.",
        ),
        (
            "What is a Prototype card?",
            "Cards with an alternate casting cost and effect, offering a less powerful but cheaper version to be cast.",
        ),
        (
            "What is a Meld card?",
            "Two specific cards that combine to form a larger, single powerful card.",
        ),
        (
            "What is a Case card?",
            "No specific MTG card type currently; this may refer to packaging or card storage terminology.",
        ),
    ]
    results = []
    for question, answer in question_answers:
        results.append(DataEntry(question=question, answer=answer))
    return results


@MTGDatasetBuilder.register(group="rules", train_order=1)
def build_tricky_situations_question_answer_dataset(
    _: MTGDatabase,
) -> list[DataEntry]:
    question_answers = [
        (
            "If I declare attackers and my opponent taps them before blockers are declared, do they still attack?",
            "Yes, once attackers are declared, tapping them does not remove them from combat. They still attack, but if tapped, they can’t be used to activate abilities.",
        ),
        (
            "Can I cast an instant spell after my opponent draws a card but before they play a land?",
            "Yes, you can cast instants or activate abilities during their Draw Step or at the beginning of their Main Phase 1 before they play a land.",
        ),
        (
            "If I have a creature with 'First Strike' and my opponent's creature has 'Deathtouch' what happens?",
            "Your creature deals damage first in the First Strike Damage Step. If it destroys the opposing creature, that creature will not get to deal damage back.",
        ),
        (
            "My opponent attacks with two creatures. Can I tap one of them with an ability after they declare attackers?",
            "No, once attackers are declared, they are already tapped. You must tap potential attackers during the Beginning of Combat Step.",
        ),
        (
            "What happens if a creature with 'Trample' is blocked by a creature with less toughness than its power?",
            "The excess damage can be assigned to the defending player or planeswalker. The attacker chooses how much damage goes to the blocker and how much tramples over.",
        ),
        (
            "Can I play a land during my opponent’s turn using a flash ability?",
            "No, you can only play lands during your own Main Phase when the stack is empty, even if you have cards with flash.",
        ),
        (
            "If I activate an ability that draws me cards and I draw a card with Flash, can I play it right away?",
            "Yes, you can play a card with Flash at any time you could play an instant, even during your opponent’s turn.",
        ),
        (
            "If I have a creature with hexproof, can I target it with my own spells?",
            "Yes, hexproof only prevents opponents from targeting it. You can target your own hexproof creature with your spells or abilities.",
        ),
        (
            "My opponent activates an ability in my Upkeep Step. Can I cast a spell in response?",
            "Yes, you can respond to the activation by casting instants or activating abilities. The stack resolves from top to bottom.",
        ),
        (
            "Can I cast a creature during my Draw Step if I have a card with “Flash”?",
            "Yes, because “Flash” allows you to cast a creature as if it were an instant, including during your Draw Step.",
        ),
        (
            "My opponent attacks me, and I flash in a creature. Can it block?",
            "Yes, if you flash in the creature during the Beginning of Combat Step or Declare Attackers Step, it can block in the Declare Blockers Step.",
        ),
        (
            "If my opponent uses a spell to exile my creature and I have an ability to sacrifice it, what happens?",
            "You can respond by sacrificing the creature, and it will go to the graveyard instead of being exiled.",
        ),
        (
            "Can a creature that enters the battlefield tapped still be declared as a blocker?",
            "No, a creature must be untapped to be declared as a blocker.",
        ),
        (
            "If I have two abilities that trigger at the beginning of combat, which resolves first?",
            "You choose the order in which your triggered abilities go on the stack and resolve.",
        ),
        (
            "Can a creature with 'Defender' still attack if an ability removes 'Defender'?",
            "Yes, if the ability removes 'Defender', the creature can attack.",
        ),
        (
            "My opponent passes their turn without attacking. Can I use an instant during their End Step?",
            "Yes, you can cast instants or activate abilities during their End Step, after they pass priority.",
        ),
        (
            "Can I cast a spell before my opponent draws on their turn?",
            "No, the first opportunity to cast a spell on their turn is after their Draw Step starts, not before they draw.",
        ),
        (
            "If I declare an attacker and my opponent flashes in a blocker, what can I do?",
            "You can respond to the blocker entering the battlefield with instants or abilities before the Declare Blockers Step.",
        ),
        (
            "Can a creature with summoning sickness be tapped for mana if it has an ability to do so?",
            "Yes, summoning sickness only prevents attacking or using abilities with the tap symbol. Mana abilities can still be used.",
        ),
        (
            "If a card gives all my creatures 'Vigilance', can they still tap to activate abilities?",
            "Yes, Vigilance only prevents creatures from tapping when attacking. They can still tap for abilities as normal.",
        ),
    ]
    result = []
    for question, answer in tqdm(
        question_answers,
        desc="Building tricky situations dataset",
        leave=False,
        total=len(question_answers),
        disable=is_tqdm_disabled(),
    ):
        result.append(DataEntry(question=question, answer=answer))
    return result


@MTGDatasetBuilder.register(group="rules", train_order=1)
def build_phases_question_answer_dataset(data: MTGDatabase) -> list[DataEntry]:
    questions_answers = [
        (
            "What are the main phases in a turn of Magic: The Gathering?",
            "The phases are: Beginning Phase, Main Phase 1, Combat Phase, Main Phase 2, and Ending Phase.",
        ),
        (
            "What happens during the Beginning Phase?",
            "It consists of three steps: Untap Step (untap all your permanents), Upkeep Step (trigger abilities), and Draw Step (draw a card).",
        ),
        (
            "Can players cast spells during the Untap Step?",
            "No, players cannot cast spells or activate abilities during the Untap Step.",
        ),
        (
            "What is the Upkeep Step used for?",
            "It's a step where triggered abilities occur, and players can cast instants or activate abilities.",
        ),
        (
            "Can you draw multiple cards during the Draw Step?",
            "Only one card is drawn naturally during the Draw Step unless effects or abilities cause more draws.",
        ),
        (
            "When can you play lands during a turn?",
            "Lands can be played during Main Phase 1 or Main Phase 2, as long as the stack is empty.",
        ),
        (
            "What is the difference between Main Phase 1 and Main Phase 2?",
            "They function identically, but Main Phase 1 is before combat, and Main Phase 2 is after combat.",
        ),
        (
            "When can creatures attack?",
            "Creatures can attack during the Combat Phase, specifically during the Declare Attackers Step.",
        ),
        (
            "What are the steps in the Combat Phase?",
            "The Combat Phase has five steps: Beginning of Combat, Declare Attackers, Declare Blockers, Combat Damage, and End of Combat.",
        ),
        (
            "Can players cast spells during the Declare Attackers Step?",
            "Yes, after attackers are declared, both players get priority to cast instants or activate abilities.",
        ),
        (
            "How does the Declare Blockers Step work?",
            "The defending player assigns blockers. After blockers are declared, both players can cast instants or activate abilities.",
        ),
        (
            "Can blocked creatures still deal damage to the defending player?",
            "No, blocked creatures deal damage to the creatures blocking them unless they have trample or another ability allowing them to bypass blockers.",
        ),
        (
            "What happens during the Combat Damage Step?",
            "Creatures deal damage simultaneously. Damage assignment depends on whether creatures are blocked, unblocked, or have special abilities.",
        ),
        (
            "Can creatures be tapped for mana during combat?",
            "Yes, creatures can be tapped for mana or other abilities during combat, as long as the ability allows it.",
        ),
        (
            "What happens if you skip your Combat Phase?",
            "Nothing special; you move directly from Main Phase 1 to Main Phase 2. Some effects may trigger if combat is skipped intentionally or unintentionally.",
        ),
        (
            "When does the stack clear during a turn?",
            "The stack clears whenever all players pass priority in succession, allowing the top spell or ability to resolve.",
        ),
        (
            "What is the Ending Phase composed of?",
            "The Ending Phase consists of the End Step and the Cleanup Step.",
        ),
        (
            "Can players cast spells during the End Step?",
            "Yes, players can cast instants or activate abilities during the End Step.",
        ),
        (
            "What happens during the Cleanup Step?",
            "Players discard down to their maximum hand size (usually 7), and damage is removed from creatures. Players cannot cast spells unless an ability triggers.",
        ),
        (
            "Can you take actions during your opponent’s turn?",
            "Yes, players can cast instants or activate abilities when they have priority, usually after the opponent passes priority during phases or steps.",
        ),
    ]
    result = []
    for question, answer in tqdm(
        questions_answers,
        desc="Building phases dataset",
        leave=False,
        total=len(questions_answers),
        disable=is_tqdm_disabled(),
    ):
        result.append(DataEntry(question=question, answer=answer))
    return result


@MTGDatasetBuilder.register(group="combos", train_order=2)
def build_cards_to_combo_question_answer_dataset(database: MTGDatabase):
    zone_locations_to_text = {
        "B": "on the battlefield",
        "G": "in the graveyard",
        "H": "in your hand",
        "L": "in the library",
        "E": "exiled",
        "C": "in the command zone",
    }
    edh_combos = EDHComboDatabase()
    result: list[DataEntry] = []
    for combo in tqdm(
        edh_combos,
        desc="Building combo question-answer dataset",
        leave=False,
        disable=is_tqdm_disabled(),
    ):
        card_names_text = ", ".join(combo["cards"]["card_name"].to_list())

        features = []
        for _, feature_name in combo["features"]["feature_name"].items():
            if "LTB" in feature_name:
                feature_name = feature_name.replace("LTB", "leaves the battlefield")
            elif "ETB" in feature_name:
                feature_name = feature_name.replace("ETB", "enters the battlefield")
            features.append(f"  - {feature_name}")
        features_text = "\n".join(features)

        steps = []
        for i, step in enumerate(combo["combo"]["steps"].splitlines()):
            steps.append(f"  {i+1}. {step}")
        steps_text = "\n".join(steps)

        question = f"How can you create a combo with {card_names_text}?"
        answer = (
            "[START OF COMBO]"
            f"This combo can be formed with {card_names_text}\n\n"
            f"Color identity: {combo['combo']['identity']}\n"
            ""
            f"Mana cost: {combo['combo']['manaNeeded']}\n"
            ""
            "Steps:\n"
            f"{steps_text}"
            "[END OF COMBO STEPS]"
            "\n\n"
            "Result:\n"
            f"{features_text}"
            "[END OF COMBO RESULT]"
            "[END OF COMBO]"
        )

        additional_prerequisites = []
        if len(combo["cards"]["zone_locations"].unique()) == 1:
            zones = combo["cards"]["zone_locations"].unique().tolist()
            zone_text = zone_locations_to_text[zones[0][0]]
            text = f"  - All permanants must be {zone_text}"
            additional_prerequisites.append(text)
        else:
            for _, card in combo["cards"].iterrows():
                zones = card["zone_locations"]
                if len(zones) == 1:
                    zone_text = zone_locations_to_text[zones[0]]
                    text = f"  - {card['card_name']} must be {zone_text}"
                    additional_prerequisites.append(text)
                else:
                    zone_text = " or ".join(
                        [zone_locations_to_text[zone] for zone in zones]
                    )
                    text = f"  - {card['card_name']} must be {zone_text}"
                    additional_prerequisites.append(text)

        prerequisites = additional_prerequisites
        if other_prerequisites := combo["combo"]["otherPrerequisites"]:
            other_prerequisites = other_prerequisites or ""
            for prerequisite in other_prerequisites.splitlines():
                prerequisites.append(f"  - {prerequisite}")

        if prerequisites:
            prerequisites.append("[END OF COMBO PREREQUISITES]")
            other_prerequisites_text = "\n".join(prerequisites)
            answer += f"\n\nOther prerequisites:\n{other_prerequisites_text}"

        result.append(DataEntry(question=question, answer=answer))
    return result
