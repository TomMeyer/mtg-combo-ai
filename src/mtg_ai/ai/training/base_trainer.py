import gc
from logging import getLogger
from typing import Optional

import torch
from peft.peft_model import PeftModel
from transformers import PreTrainedTokenizer
from transformers.trainer import Trainer

from mtg_ai.ai.training.dataset_loader import MTGCardAITrainingDatasetLoader

logger = getLogger(__name__)


class BaseTrainer:
    def __init__(
        self,
        model_name: str,
        datasets: list[str],
        output_name: str,
        max_seq_length: int = 300,
    ) -> None:
        # set properties
        self.output_name = output_name
        self._model: Optional[PeftModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self.tokenizer_name = model_name
        self.max_seq_length = max_seq_length
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._data_loader = None
        self.datasets = datasets
        self.load_model()

    @property
    def model(self) -> PeftModel:
        if not self._model:
            self.load_model()
            if self._model is None:
                raise RuntimeError("Model not loaded")
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        if not self._tokenizer:
            self.load_model()
            if self._tokenizer is None:
                raise RuntimeError("Tokenizer not loaded")
            self.tokenizer.add_tokens(
                [
                    "[END OF COMBO STEPS]",
                    "[END OF COMBO RESULT]",
                    "[END OF COMBO PREREQUISITES]",
                    "[START OF COMBO]",
                    "[END OF COMBO]",
                ]
            )
        return self._tokenizer

    @property
    def data_loader(self) -> MTGCardAITrainingDatasetLoader:
        if not self._data_loader:
            self._data_loader = MTGCardAITrainingDatasetLoader(
                tokenizer=self.tokenizer, datasets=self.datasets
            )
        return self._data_loader

    @classmethod
    def _get_model_and_tokenizer(
        cls, model_name: str, max_sequence_length: int = 500
    ) -> tuple[PeftModel, PreTrainedTokenizer]:
        raise NotImplementedError("Must be implemented in subclass")

    def load_model(self):
        logger.info(f"loading model {self.model_name}")
        model, tokenizer = self._get_model_and_tokenizer(
            self.model_name, self.max_seq_length
        )
        self._model = model
        self._tokenizer = tokenizer

    def build_trainer(
        self,
        learning_rate: float = 3e-5,
        weight_decay: float = 1e-6,
        train_batch_size: int = 16,
        eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        # trainer_args: dict[str, Any], # TODO: allow for specific args to be passed in
    ) -> Trainer:
        raise NotImplementedError("Must be implemented in subclass")

    def train(
        self,
        resume_from_checkpoint: bool = False,
        learning_rate: float = 3e-5,
        weight_decay: float = 1e-6,
        train_batch_size: int = 16,
        eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        raise NotImplementedError("Must be implemented in subclass")

    def evaluate(self) -> None:
        trainer = self.build_trainer()
        if trainer.eval_dataset is None:
            raise ValueError("No evaluation dataset loaded")
        logger.info("Evaluating model")
        eval_results = trainer.evaluate(ignore_keys=["predictions"])
        logger.info("Finished evaluating model")
        logger.info(f"Evaluation Results:\n{eval_results}")

    def print_gpu_stats(self):
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"Using GPU: {self.device}")
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        logger.info(f"{start_gpu_memory} GB of memory reserved.")

    def free_memory(self):
        self._model = None
        self._tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
