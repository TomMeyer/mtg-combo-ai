from pathlib import Path
from typing import Any
import pandas as pd
import mtg_ai.constants as constants


def remove_uneeded_columns(df: pd.DataFrame):
    df.rename(columns={"name": "card_name"}, inplace=True)
    df = df.drop(constants.drop_columns, axis=1)
    return df

def filter_for_modern(df: pd.DataFrame):
    legalities = pd.json_normalize(df.pop("legalities"))
    df = df.loc[legalities["modern"] == "legal"].reset_index(drop=True)
    return df

def fill_empty_values(df: pd.DataFrame):
    fill_values = dict.fromkeys(["mana_cost", "colors", "color_identity", "produced_mana", "color_indicator"], constants.NA_STRING)
    fill_values.update(dict.fromkeys(["power", "toughness", "loyalty", ], constants.NAN_STRING))
    fill_values["edhrec_rank"] = 0
    df.fillna(fill_values, inplace=True)
    return df

def merge_lists(df: pd.DataFrame):
    df.keywords = df.keywords.str.join(", ")
    columns = ["colors", "color_identity", "color_indicator", "produced_mana"]
    df[columns] = df[columns].map(lambda x: "".join(x))
    return df

def sort_color_strings(df: pd.DataFrame):
    columns = ["colors", "color_identity", "color_indicator", "produced_mana"]
    df[columns] = df[columns].map(constants.MTGColorCombo._sort_multicolor_str)
    return df

def convert_column_types(df: pd.DataFrame):
    data = {
        "oracle_id": str,
        "card_name": str,
        "rarity": constants.CATEGORY,
        "mana_cost": constants.CATEGORY,
        "cmc": float,
        "colors": constants.CATEGORY,
        "color_identity": constants.CATEGORY,
        "type_line": str,
        "power": str,
        "toughness": str,
        "loyalty": constants.CATEGORY,
        "produced_mana": constants.CATEGORY,
        "set_type": constants.CATEGORY,
        "oracle_text": str,
        "layout": constants.CATEGORY,
        "edhrec_rank": int,
        "color_indicator": constants.CATEGORY,
    }
    df = df.astype(data)
    return df

mtg_data_path = Path("./data/oracle-cards-20231121100139.json")

MTGCards: pd.DataFrame = (
    pd.read_json(mtg_data_path)
    .pipe(remove_uneeded_columns)
    .pipe(filter_for_modern)
    .pipe(fill_empty_values)
    .pipe(merge_lists)
    .pipe(sort_color_strings)
    .pipe(convert_column_types)
)


# def process() -> pd.DataFrame:
#     mtg_data_path = Path("./data/oracle-cards-20231121100139.json")
#     df: pd.DataFrame = pd.read_json(mtg_data_path)
#     df.rename(columns={"name": "card_name"}, inplace=True)
#     df.drop(constants.drop_columns, axis=1, inplace=True)
#     legalities = _expand_legalities_column(df)
#     df = _filter_legalities(df, legalities, "modern")
#     df = _clean_columns(df)
#     if len(constants.column_order) != len(df.columns):
#         raise Exception()
#     df = _calculate_columns(df)
#     df = df.reindex(columns=constants.column_order)
#     df.set_index("oracle_id", inplace=True)
#     mtg_ruling_path = Path("./data/wotc_rulings.json")
#     rulings_df = pd.read_json(mtg_ruling_path)
#     df = df.merge(rulings_df, left_index=True, right_index=True)
#     return df


