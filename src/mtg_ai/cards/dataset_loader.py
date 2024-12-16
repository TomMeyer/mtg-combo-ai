import logging
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from datasets.combine import concatenate_datasets

from mtg_ai.cards.training_data_builder import QUESTION_ANSWER_FOLDER
from mtg_ai.cards.utils import classproperty

logger = logging.getLogger(__name__)


class MTGDatasetLoader:
    _datasets: dict[str, Path] = {}
    _directory: Path = QUESTION_ANSWER_FOLDER
    """
    MTGDatasetLoader is a class for loading Magic: The Gathering (MTG) datasets from a specified directory.

    Attributes:
        directory (Path): The directory containing the dataset files.
        files (dict[str, tuple[Path, bool]]): A dictionary mapping dataset names to their file paths and a boolean indicating if they are Arrow datasets.

    Methods:
        __init__(directory: Path = QUESTION_ANSWER_FOLDER):
            Initializes the MTGDatasetLoader with the specified directory and populates the files attribute with dataset files.

        list_datasets() -> list[str]:
            Returns a list of dataset names available in the directory.

        load_dataset(name: str) -> datasets.Dataset:
            Loads the specified dataset by name. Raises a ValueError if the dataset is not found.
            If the dataset is an Arrow dataset, it is loaded from disk. Otherwise, it is loaded from a JSON file and saved to disk as an Arrow dataset.
    """

    def __init__(self):
        raise NotImplementedError(
            "MTGDatasetLoader is a static class and should not be instantiated."
        )

    @classproperty  # type: ignore
    def directory(cls) -> Path:
        return cls._directory

    @classproperty  # type: ignore
    def datasets(cls) -> dict[str, Path]:
        if not cls._datasets:
            cls._update_datasets()
        return cls._datasets

    @classmethod
    def set_directory(cls, directory: Path):
        cls._directory = directory
        cls._update_datasets()

    @classmethod
    def _update_datasets(cls):
        cls._datasets = {}
        if not cls.directory.exists():
            return
        for file in cls.directory.iterdir():
            name = file.stem.replace("_question_answer", "")
            cls._datasets[name] = file

    @classproperty  # type: ignore
    def dataset_names(cls) -> list[str]:
        return list(cls.datasets.keys())

    @classmethod
    def load_dataset(cls, *name: str) -> Dataset:
        if not name:
            raise ValueError("No dataset name specified.")
        elif len(name) == 1:
            dataset = cls._load_dataset(name[0])
            return dataset

        datasets = []
        for dataset_name in name:
            dataset = cls._load_dataset(dataset_name)
            datasets.append(dataset)
        return concatenate_datasets(datasets)

    @classmethod
    def _load_dataset(cls, name: str) -> Dataset:
        """
        Load a dataset by its name.

        Args:
            name (str): The name of the dataset to load.

        Returns:
            datasets.Dataset: The loaded dataset.

        Raises:
            ValueError: If the dataset name is not found in the available files.
            TypeError: If the loaded dataset is a DatasetDict instead of a Dataset.
        """

        if name in cls.datasets:
            file = cls.datasets[name]
            dataset = load_from_disk(file)
        else:
            dataset = load_dataset(name)
        if isinstance(dataset, DatasetDict):
            dataset = dataset["train"]
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Loaded dataset is not a Dataset: {type(dataset)}")
        return dataset
