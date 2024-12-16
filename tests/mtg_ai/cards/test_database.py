import sqlite3
from pathlib import Path

from mtg_ai.cards.database import (
    _build_filtered_sqlite,
    _get_mtg_sqlite_file,
)


def test_get_mtg_sqlite_file_download(mocker, tmp_path):
    mocker.patch("mtg_ai.cards.database.MTG_CACHE_DIR", tmp_path)
    mocker.patch(
        "mtg_ai.cards.database.MTG_SQLITE_FILE", tmp_path / "AllPrintings.sqlite"
    )
    _get_mtg_sqlite_file()

    assert Path(tmp_path / "AllPrintings.sqlite").exists()


def test_get_mtg_sqlite_file_columns(mocker, tmp_path):
    mocker.patch("mtg_ai.cards.database.MTG_CACHE_DIR", tmp_path)
    mocker.patch(
        "mtg_ai.cards.database.MTG_SQLITE_FILE", tmp_path / "AllPrintings.sqlite"
    )
    mocker.patch(
        "mtg_ai.cards.database.MTG_FILTERED_SQLITE_FILE",
        tmp_path / "FilteredPrintings.sqlite",
    )
    mocker.patch("requests.get")
    mocker.patch("gzip.GzipFile")
    mocker.patch("shutil.copyfileobj")
    _build_filtered_sqlite()

    assert Path(tmp_path / "FilteredPrintings.sqlite").exists()

    conn = sqlite3.connect(tmp_path / "FilteredPrintings.sqlite")

    cursor = conn.execute("SELECT * FROM filtered_cards LIMIT 1")
    columns = [description[0] for description in cursor.description]

    expected_columns = [
        "uuid",
        "name",
        "asciiName",
        "faceConvertedManaCost",
        "manaCost",
        "type",
        "power",
        "toughness",
        "loyalty",
        "defense",
        "colorIdentity",
        "colorIndicator",
        "colors",
        "edhrecRank",
        "edhrecSaltiness",
        "keywords",
        "language",
        "layout",
        "leadershipSkills",
        "manaValue",
        "rarity",
        "side",
        "types",
        "subtypes",
        "supertypes",
        "subsets",
        "text",
        "originalText",
        "cardParts",
        "rulings",
    ]

    assert columns == expected_columns
    conn.close()
