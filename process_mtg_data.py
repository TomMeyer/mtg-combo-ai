import pandas as pd
from pathlib import Path

def process():
    mtg_data_path = Path("./oracle-cards-20231121100139.json")
    df: pd.DataFrame = pd.read_json(mtg_data_path)
    df = expand_legalities_column(df)
    
    return df


def expand_legalities_column(df: pd.DataFrame):
    legalities = pd.json_normalize(df["legalities"])
    df.drop("legalities", axis=1, inplace=True)
    df = df.merge(legalities, left_index=True, right_index=True)
    return df


if __name__ == "__main__":
    process()