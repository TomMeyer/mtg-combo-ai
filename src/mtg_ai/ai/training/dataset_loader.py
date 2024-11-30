from logging import getLogger
from typing import Any, Optional, TypeVar

from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, Trainer

from mtg_ai.cards import MTGDatasetLoader

logger = getLogger(__name__)

TrainerT = TypeVar("TrainerT", bound=Trainer)


class MTGCardAITrainingDatasetLoader:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        datasets: str | list[str],
        num_procs: Optional[int] = None,
        accelerator: Optional[Accelerator] = None,
    ) -> None:
        """
        Initializes the MTGCardAITrainingDatasetLoader with the specified parameters.

        ### Args
        - **tokenizer** (PreTrainedTokenizer | PreTrainedTokenizerFast):
            The tokenizer to be used for processing the dataset.
        - **datasets** (str | list[str]):
            The dataset or list of datasets to be loaded.
        - **num_procs** (Optional[int], optional):
            The number of processes to use for tokenization. Defaults to `None`.

        ### Returns
        None

        ### Raises
        - **ValueError**: If the datasets parameter is not a string or list of strings.
        """
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        if isinstance(datasets, str):
            dataset: Dataset = MTGDatasetLoader.load_dataset(datasets)
        elif isinstance(datasets, list):
            dataset = MTGDatasetLoader.load_dataset(*datasets)
        else:
            raise ValueError("datasets must be a string or list of strings")
        logger.info(f"chat template: {self.tokenizer.get_chat_template()}")

        def format_prompt(examples) -> dict[str, Any]:
            texts = self.tokenizer.apply_chat_template(
                conversation=examples["conversations"],
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            return {"texts": texts}

        logger.info(f"Tokenizing dataset with {num_procs} processes")
        formatted_dataset = dataset.map(format_prompt, batched=True)
        logger.info("finished tokenizing dataset")

        self.dataset: DatasetDict = formatted_dataset.train_test_split(test_size=0.2)

    @property
    def train_dataset(self) -> Dataset:
        """
        Returns the training dataset.

        ### Args
        None

        ### Returns
        Dataset:
        The training dataset.

        ### Raises
        - **ValueError**: If the dataset is not loaded.
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded")
        return self.dataset["train"]  # type: ignore

    @property
    def test_dataset(self) -> Dataset:
        """
        Returns the test dataset.

        ### Args
        None

        ### Returns
        Dataset:
        The test dataset.

        ### Raises
        - **ValueError**: If the dataset is not loaded.
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded")
        return self.dataset["test"]  # type: ignore
