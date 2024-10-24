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


def build_card_name_to_manacost_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result = []
    for _, row in data.df.loc[
        data.df.manaCost.isna(), ["name", "manaCost", "side"]
    ].iterrows():
        name = row["name"]
        if row["side"] == "a":
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b":
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
        if row["side"] == "a":
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b":
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


def build_card_name_to_cmc_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    for _, row in data.df[["name", "cmc", "side"]].iterrows():
        name = row["name"]
        if row["side"] == "a":
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b":
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


def build_card_name_to_keywords_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result = []
    for _, row in data.df.loc[
        data.df.keywords.notna(), ["name", "keywords", "side"]
    ].iterrows():
        name = row["name"]
        if row["side"] == "a":
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b":
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


def build_card_name_to_color_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    for _, row in data.df[["name", "colors", "side"]].iterrows():
        name = row["name"]
        if row["side"] == "a":
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b":
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


def build_card_name_to_color_identity_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    df = data.df.loc[data.df.colorIdentity != "", ["name", "colorIdentity", "side"]]
    for _, row in df.iterrows():
        name = row["name"]
        if row["side"] == "a":
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b":
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


def build_card_name_to_ascii_name_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    df = data.df.loc[data.df.asciiName.notna(), ["name", "asciiName", "side"]]
    for _, row in df[["name", "asciiName"]].iterrows():
        name = row["name"]
        if row["side"] == "a":
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b":
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


def build_card_name_to_type_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    for _, row in data.df[["name", "type", "side"]].iterrows():
        name = row["name"]
        if row["side"] == "a":
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b":
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


def build_card_name_to_power_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    df = data.df.loc[data.df.power.notna(), ["name", "power", "side"]]

    for _, row in df[["name", "power", "side"]].iterrows():
        name = row["name"]
        if row["side"] == "a":
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b":
            name = row["name"].split(" // ")[1]
        questions = [
            f"What is the power of {name}?",
            f"Can you tell me the power of {name}?",
            f"How much power does {name} have?",
        ]
        answer = f"{row['power']}"
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    for _, row in data.df.loc[data.df.power.isna(), ["name"]].iterrows():
        questions = [
            f"What is the power of {row['name']}?",
            f"Can you tell me the power of {row['name']}?",
            f"How much power does {row['name']} have?",
        ]
        answer = f"{row['name']} has no power"
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    return result


def build_card_name_to_text_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    for _, row in data.df[["name", "text", "side"]].iterrows():
        name = row["name"]
        if row["side"] == "a":
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b":
            name = row["name"].split(" // ")[1]
        questions = [
            f"What is the text of {name}?",
            f"What are the abilities of {name}?",
            f"Can you tell me the text of {name}?",
            f"What does {name} do?",
        ]
        answer = f"{row['text']}"
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    return result


def build_card_name_to_loyalty_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result = []
    df = data.df.loc[data.df.loyalty.notna()]
    for _, row in df[["name", "loyalty", "side"]].iterrows():
        name = row["name"]
        if row["side"] == "a":
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b":
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


def build_card_name_to_rarity_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    for _, row in data.df.loc[
        data.df["rarity"].notna(), ["name", "rarity", "side"]
    ].iterrows():
        name = row["name"]
        if row["side"] == "a":
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b":
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


def build_card_name_to_toughness_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    df = data.df.loc[data.df.toughness.notna()]
    for _, row in df[["name", "toughness", "side"]].iterrows():
        name = row["name"]
        if row["side"] == "a":
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b":
            name = row["name"].split(" // ")[1]
        questions = [
            f"What is the toughness of {name}?",
            f"Can you tell me the toughness of {name}?",
            f"What is {name}'s toughness?",
        ]
        answer = f"{row['toughness']}"
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    for _, row in data.df.loc[data.df.toughness.isna(), ["name", "side"]].iterrows():
        name = row["name"]
        if row["side"] == "a":
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b":
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


def build_card_with_cmc_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []
    for cmc in data.cmcs:
        for _, row in data.get_cards_by_cmc(cmc)[["name", "cmc", "side"]].iterrows():
            name = row["name"]
            if row["side"] == "a":
                name = row["name"].split(" // ")[0]
            elif row["side"] == "b":
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


def build_card_name_to_side_question_answer_dataset(
    data: MTGDatabase,
) -> list[DataEntry]:
    result: list[DataEntry] = []

    for _, row in data.df.loc[data.df.side.notna(), ["name", "side"]].iterrows():
        name = ""
        if row["side"] == "a":
            name = row["name"].split(" // ")[0]
        elif row["side"] == "b":
            name = row["name"].split(" // ")
        else:
            raise ValueError(f"Invalid side value: {row['side']}")
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
            f"What is the name of side {row['side']} of {row['name']}?",
            f"Can you tell me the name of side {row[1]['side']} of {row['name']}?",
            f"What is the name of {row[1]['side']} side of {row['name']}?",
        ]
        for question in questions:
            result.append(DataEntry(question=question, answer=answer))
    return result


def build_tricky_situations_question_answer_dataset(
    data: MTGDatabase,
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
    for question, answer in question_answers:
        result.append(DataEntry(question=question, answer=answer))
    return result


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
    for question, answer in questions_answers:
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
