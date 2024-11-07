from logging import getLogger
from multiprocessing import cpu_count
from typing import Any, Optional, TypeVar

from datasets import Dataset, DatasetDict
from transformers.trainer import Trainer

from mtg_ai.cards import MTGDatasetLoader

logger = getLogger(__name__)

TrainerT = TypeVar("TrainerT", bound=Trainer)


class MTGCardAITrainingDatasetLoader:
    def __init__(
        self,
        tokenizer,
        datasets: str | list[str],
        num_procs: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        if isinstance(datasets, str):
            dataset: Dataset = MTGDatasetLoader.load_dataset(datasets)
        elif isinstance(datasets, list):
            dataset = MTGDatasetLoader.load_dataset(*datasets)
        else:
            raise ValueError("datasets must be a string or list of strings")

        def format_prompt(examples) -> dict[str, Any]:
            texts = self.tokenizer.apply_chat_template(
                conversation=examples["conversations"],
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": texts}

        num_procs = min(num_procs or cpu_count() - 1, 12)
        logger.info(f"Tokenizing dataset with {num_procs} processes")
        formatted_dataset = dataset.map(format_prompt, num_proc=num_procs, batched=True)  # type: ignore
        logger.info("finished tokenizing dataset")

        self.dataset: DatasetDict = formatted_dataset.train_test_split(test_size=0.2)

    @property
    def train_dataset(self) -> Dataset:
        if not self.dataset:
            raise ValueError("Dataset not loaded")
        return self.dataset["train"]  # type: ignore

    @property
    def test_dataset(self) -> Dataset:
        if not self.dataset:
            raise ValueError("Dataset not loaded")
        return self.dataset["test"]  # type: ignore
