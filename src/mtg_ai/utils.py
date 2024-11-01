import os


def enable_tqdm():
    os.environ["TQDM_DISABLE"] = ""


def disable_tqdm():
    os.environ["TQDM_DISABLE"] = "1"


def is_tqdm_disabled():
    return bool(os.getenv("TQDM_DISABLE", False))
