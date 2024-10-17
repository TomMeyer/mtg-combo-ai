import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd
import rapidfuzz

from mtg_ai.cards.utils import color_to_identity, sort_colors, strip_commas
from mtg_ai.constants import (
    COLOR_IDENTITY,
    MTG_CARD_TYPE,
    MTG_COLOR,
    CardType,
    Color,
    ColorIdentity,
)

MTG_CACHE_DIR = Path.home().joinpath(".cache", "mtg-ai")
MTG_SQLITE_FILE = MTG_CACHE_DIR.joinpath("AllPrintings.sqlite")
MTG_FILTERED_SQLITE_FILE = MTG_CACHE_DIR.joinpath("FilteredPrintings.sqlite")

BUILD_FILTER_SQL_COMMAND = """
WITH
    rulings AS (
        SELECT
            uuid,
            GROUP_CONCAT(text, ' ') as text
        FROM
            cardRulings
        GROUP BY
            uuid
    ),
    ranked_cards AS (
        SELECT
            c.*,
            ROW_NUMBER() OVER (
                PARTITION BY c.name, COALESCE(c.side, 'NULL_SIDE')
                ORDER BY c.uuid
            ) AS rn
        FROM
            cards c
        INNER JOIN
            cardLegalities cl ON cl.uuid = c.uuid
        WHERE
            cl.modern = 'Legal'
    )
SELECT
    rc.uuid,
    rc.name,
    rc.asciiName,
    rc.faceConvertedManaCost,
    rc.manaCost,
    rc.type,
    rc.power,
    rc.toughness,
    rc.loyalty,
    rc.defense,
    rc.colorIdentity,
    rc.colorIndicator,
    rc.colors,
    rc.edhrecRank,
    rc.edhrecSaltiness,
    rc.keywords,
    rc.language,
    rc.layout,
    rc.leadershipSkills,
    rc.manaValue,
    rc.rarity,
    rc.side,
    rc.types,
    rc.subtypes,
    rc.supertypes,
    rc.subsets,
    rc.text,
    rc.cardParts
FROM
    ranked_cards rc
    LEFT JOIN rulings r ON r.uuid = rc.uuid
WHERE
    rc.rn = 1;
"""


def _get_mtg_sqlite_file():
    import gzip
    import shutil

    import requests

    mtgjson_download_url = "https://mtgjson.com/api/v5/AllPrintings.sqlite.gz"

    if not MTG_CACHE_DIR.exists():
        MTG_CACHE_DIR.mkdir(parents=True)

    resposne = requests.get(mtgjson_download_url, stream=True)
    with gzip.GzipFile(fileobj=resposne.raw) as f_in:
        with MTG_SQLITE_FILE.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def _build_filtered_sqlite():
    if not MTG_SQLITE_FILE.exists():
        _get_mtg_sqlite_file()
    if MTG_FILTERED_SQLITE_FILE.exists():
        return
    # Read the AllPrintings.sql file and run the filtered_cards.sql script
    conn = sqlite3.connect(MTG_SQLITE_FILE)
    cursor = conn.execute(BUILD_FILTER_SQL_COMMAND)

    # Collect the filtered data
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]

    # Create the new sqlite database
    new_db_conn = sqlite3.connect(MTG_FILTERED_SQLITE_FILE)
    new_cursor = new_db_conn.cursor()

    # Create the new table
    create_table_query = f"""
    CREATE TABLE filtered_cards (
        {', '.join(f'{col} TEXT' for col in columns)}
    )
    """
    new_cursor.execute(create_table_query)

    # Insert the filtered data into the new table
    insert_query = f"""
    INSERT INTO filtered_cards ({', '.join(columns)})
    VALUES ({', '.join(['?' for _ in columns])})
    """
    new_cursor.executemany(insert_query, rows)

    # Commit the transaction and close the new database connection
    new_db_conn.commit()
    new_db_conn.close()


@dataclass
class MTGDatabase:
    TABLE_NAME: ClassVar[str] = "filtered_cards"

    def __post_init__(self):
        if not MTG_FILTERED_SQLITE_FILE.exists():
            _build_filtered_sqlite()
        conn = sqlite3.connect(MTG_FILTERED_SQLITE_FILE)
        df = pd.read_sql_query(
            f"SELECT * FROM {self.TABLE_NAME}",
            conn,
            dtype={
                "uuid": pd.StringDtype(),
                "name": pd.StringDtype(),
                "asciiName": pd.SparseDtype(pd.StringDtype(), fill_value=""),
                "faceConvertedManaCost": pd.CategoricalDtype(ordered=True),
                "manaCost": pd.CategoricalDtype(ordered=True),
                "power": pd.CategoricalDtype(ordered=True),
                "toughness": pd.CategoricalDtype(ordered=True),
                "loyalty": pd.CategoricalDtype(ordered=True),
                "defense": pd.CategoricalDtype(ordered=True),
                "type": pd.CategoricalDtype(),
                "colorIdentity": pd.CategoricalDtype(),
                "colorIndicator": pd.CategoricalDtype(),
                "colors": pd.CategoricalDtype(),
                "edhrecRank": pd.Int16Dtype(),
                "edhrecSaltiness": pd.Float32Dtype(),
                "keywords": pd.CategoricalDtype(),
                "language": pd.CategoricalDtype(),
                "layout": pd.CategoricalDtype(),
                "leadershipSkills": pd.CategoricalDtype(),
                "rarity": pd.CategoricalDtype(),
                "side": pd.CategoricalDtype(),
                "types": pd.CategoricalDtype(),
                "subtypes": pd.CategoricalDtype(),
                "supertypes": pd.CategoricalDtype(),
                "subsets": pd.CategoricalDtype(),
                "text": pd.StringDtype(),
                "cardParts": pd.CategoricalDtype(),
            },
        )

        def convert_float_to_int_to_category(column):
            try:
                df[column] = (
                    df[column]
                    .astype(float)
                    .astype(pd.Int8Dtype())
                    .astype(pd.CategoricalDtype(ordered=True))
                )
            except Exception:
                raise ValueError(f"Failed to convert {column} to int")

        convert_float_to_int_to_category("manaValue")
        convert_float_to_int_to_category("faceConvertedManaCost")

        conn.close()
        df["colorIndicator"] = df.colorIndicator.apply(strip_commas)
        df["colorIndicator"] = df.colorIndicator.apply(sort_colors)
        df["colorIdentity"] = df.colorIdentity.apply(strip_commas)
        df["colorIdentity"] = df.colorIdentity.apply(sort_colors)
        df["colorIdentityName"] = df.colorIdentity.apply(color_to_identity)
        df = df.rename(columns={"manaValue": "cmc"})
        self.df = df

    def get_card_by_id(self, uuid: str) -> pd.Series:
        return self.df.loc[self.df["uuid"] == uuid].iloc[0]

    def get_card_by_name(
        self, name: str, search_type: Literal["fuzzy", "exact"] = "fuzzy"
    ) -> pd.Series:
        if not name:
            raise ValueError("name cannot be empty")

        match search_type:
            case "fuzzy":
                similarity = self.df["name"].apply(
                    lambda x: rapidfuzz.fuzz.partial_ratio(name, x)
                )
                result = self.df.loc[similarity.idxmax()]
                if isinstance(result, pd.DataFrame):
                    result = result.iloc[0]
                return result
            case "exact":
                return self.df[self.df["name"] == name].iloc[0]
            case _:
                raise ValueError("search_type must be 'fuzzy' or 'exact'")

    def get_card_by_color_identity(
        self, color_identity: COLOR_IDENTITY
    ) -> pd.DataFrame:
        ci = ColorIdentity.get(color_identity)
        matches = self.df.colorIdentity.str.contains(ci, case=False, na=False)
        return self.df[matches]

    def get_card_by_mana_cost(self, mana_cost: str) -> pd.DataFrame:
        matches = self.df.manaCost.str.contains(mana_cost, case=False, na=False)
        return self.df[matches]

    def get_cards_by_color(self, color: MTG_COLOR) -> pd.DataFrame:
        c = Color.get(color)
        matches = self.df.colors.str.contains(c, case=False, na=False)
        return self.df[matches]

    def get_cards_by_type(self, card_type: MTG_CARD_TYPE) -> pd.DataFrame:
        ct = CardType.get(card_type)
        matches = self.df.type.str.contains(ct, case=False, na=False)
        return self.df[matches]

    def get_cards_by_cmc(self, mana_cost: int) -> pd.DataFrame:
        return self.df[self.df.cmc == mana_cost]

    @property
    def cmcs(self):
        cms = self.df.cmc.unique().tolist()
        cms.sort()
        return cms
