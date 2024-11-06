import json
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict

from mtg_ai.cards.training_data_builder import QUESTION_ANSWER_FOLDER, MTGDatasetBuilder
from mtg_ai.cards.utils import classproperty

logger = logging.getLogger(__name__)


class MTGDatasetLoader:
    _datasets: dict[str, tuple[Path, bool]] = {}
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
    def datasets(cls) -> dict[str, tuple[Path, bool]]:
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
        for file in QUESTION_ANSWER_FOLDER.glob("*.json"):
            name = file.stem.replace("_question_answer", "")
            cls._datasets[name] = (file, False)
        for file in cls.directory.iterdir():
            if file.is_dir() and file.stem in cls._datasets:
                cls._datasets[file.stem] = (file, True)

    @classproperty  # type: ignore
    def dataset_names(cls) -> list[str]:
        return list(cls.datasets.keys())

    @classproperty  # type: ignore
    def dataset_order(cls) -> list[str]:
        return [
            key
            for key, _ in sorted(
                MTGDatasetBuilder.group_order.items(), key=lambda x: x[1]
            )
        ]

    @classmethod
    def load_dataset(cls, name: str) -> Dataset:
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
        if name not in cls.datasets:
            raise ValueError(f"Dataset {name} not found in {cls.datasets}")

        file, is_arrow_dataset = cls.datasets[name]
        if is_arrow_dataset:
            dataset = Dataset.load_from_disk(str(file))
            if isinstance(dataset, DatasetDict):
                print(dataset)
                raise TypeError("Dataset is a DatasetDict")
            return dataset
        else:
            logger.info(f"loading {file} to HuggingFace dataset")
            data = json.loads(file.read_text())
            dataset = Dataset.from_dict(data)
            dataset.save_to_disk(str(file.with_suffix("")))
            return dataset
