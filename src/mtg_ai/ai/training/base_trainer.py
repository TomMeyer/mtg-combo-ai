import gc
from logging import getLogger
from typing import Optional, TypeVar

import torch
from peft import PeftMixedModel, PeftModel
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.trainer import Trainer
from trl import SFTConfig

from mtg_ai.ai.training.config import MTGAITrainingConfig
from mtg_ai.ai.training.dataset_loader import MTGCardAITrainingDatasetLoader

logger = getLogger(__name__)


ModelT = TypeVar("ModelT", bound=torch.nn.Module)


class BaseTrainer:
    """
    Base class for training MTGCard AI models.

    ### Attributes
    - **output_name** (str):
        The name of the output file or directory.
    - **model** (Optional[PeftModel]):
        The model to be trained.
    - **tokenizer** (Optional[PreTrainedTokenizer | PreTrainedTokenizerFast]):
        The tokenizer for the model.
    - **tokenizer_name** (str):
        The name of the tokenizer.
    - **max_seq_length** (int):
        The maximum sequence length for the model.
    - **model_name** (str):
        The name of the model.
    - **device** (torch.device):
        The device to be used for training (CPU or GPU).
    - **data_loader** (Optional[MTGCardAITrainingDatasetLoader]):
        The data loader for training and evaluation datasets.
    - **datasets** (list[str]):
        A list of dataset names to be used for training.

    ### Methods
    - **_get_model_and_tokenizer**: Loads the model and tokenizer.
    - **model**: Returns the model instance.
    - **tokenizer**: Returns the tokenizer instance.
    - **data_loader**: Returns the data loader instance.
    - **load_model**: Loads the model and tokenizer.
    - **build_trainer**: Builds a trainer with specified hyperparameters.
    - **train**: Trains the model with specified parameters.
    - **evaluate**: Evaluates the model using the evaluation dataset.
    - **print_gpu_stats**: Prints the GPU statistics.
    - **free_memory**: Frees up memory by setting model and tokenizer to None and clearing cache.

    ### Example
    ```python
    trainer = BaseTrainer(
        model_name="model_name",
        datasets=[
            "dataset1",
            "dataset2",
        ],
        output_name="output_name",
        max_seq_length=300,
    )
    trainer.train()
    ```
    """

    def __init__(
        self,
        training_config: MTGAITrainingConfig,
        sft_config: SFTConfig,
    ) -> None:
        """
        Initializes the BaseTrainer with the specified parameters.

        ### Args
        - **model_name** (str):
            The name of the model to be used.
        - **datasets** (list[str]):
            A list of dataset names to be used for training.
        - **output_name** (str):
            The name of the output file or directory.
        - **max_seq_length** (int, optional):
            The maximum sequence length for the model. Defaults to `300`.

        ### Returns
        None

        ### Raises
        None
        """
        # set properties
        self.sft_config = sft_config
        if not training_config:
            training_config = MTGAITrainingConfig()
        self.training_config = training_config
        self._model: Optional[PeftModel | PeftMixedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._data_loader = None
        self.load_model()

    @classmethod
    def _get_model_and_tokenizer(
        cls, model_name: str, max_sequence_length: int = 500
    ) -> tuple[
        PeftModel | PeftMixedModel, PreTrainedTokenizer | PreTrainedTokenizerFast
    ]:
        """
        Loads the model and tokenizer based on the provided model name and maximum sequence length.

        ### Args
        - **model_name** (str, optional):
            The name of the model to load. Defaults to `None`.
        - **max_sequence_length** (int, optional):
            The maximum sequence length for the model. Defaults to `500`.

        ### Returns
        tuple[PeftModel, PreTrainedTokenizer | PreTrainedTokenizerFast]:
        A tuple containing the loaded model and tokenizer.

        ### Raises
        - **RuntimeError**: If the model or tokenizer could not be loaded.
        """
        raise NotImplementedError("Must be implemented in subclass")

    @property
    def model(self) -> PeftModel | PeftMixedModel:
        """
        Returns the model instance.

        ### Returns
        - **PeftModel**:
            The model instance.

        ### Raises
        - **RuntimeError**: If the model is not loaded.
        """
        if not self._model:
            self.load_model()
            if self._model is None:
                raise RuntimeError("Model not loaded")
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """
        Returns the tokenizer instance.

        ### Returns
        - **PreTrainedTokenizer | PreTrainedTokenizerFast**:
            The tokenizer instance.

        ### Raises
        - **RuntimeError**: If the tokenizer is not loaded.
        """
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
            self.tokenizer.padding_side = "right"
        return self._tokenizer

    @property
    def data_loader(self) -> MTGCardAITrainingDatasetLoader:
        """
        Returns the data loader instance.

        ### Returns
        - **MTGCardAITrainingDatasetLoader**:
            The data loader instance.

        ### Raises
        None
        """
        if not self._data_loader:
            self._data_loader = MTGCardAITrainingDatasetLoader(
                tokenizer=self.tokenizer,
                datasets=self.training_config.datasets,
                num_procs=self.sft_config.dataset_num_proc,
            )
        return self._data_loader

    def load_model(self):
        """
        Loads the model and tokenizer.

        ### Args
        None

        ### Returns
        None

        ### Raises
        None
        """
        logger.info(f"loading model {self.training_config.model_id}")
        if self.sft_config:
            max_seq_length = self.sft_config.max_seq_length or 1000
        else:
            max_seq_length = 1000
        model, tokenizer = self._get_model_and_tokenizer(
            self.training_config.model_id, max_sequence_length=max_seq_length
        )
        tokenizer.padding_side = "right"
        self._model = model
        self._tokenizer = tokenizer

    def build_trainer(
        self,
        # trainer_args: dict[str, Any], # TODO: allow for specific args to be passed in
    ) -> Trainer:
        """
        Build a trainer with the specified hyperparameters.

        ### Args
        - **learning_rate** (float, optional):
        The learning rate for training. Defaults to `3e-5`.
        - **weight_decay** (float, optional):
        The weight decay for regularization. Defaults to `1e-6`.
        - **train_batch_size** (int, optional):
        The batch size for training. Defaults to `16`.
        - **eval_batch_size** (int, optional):
        The batch size for evaluation. Defaults to `8`.
        - **gradient_accumulation_steps** (int, optional):
        The number of gradient accumulation steps. Defaults to `1`.

        ### Returns
        Trainer: An instance of a `Trainer`.

        ### Raises
        NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def train(
        self,
    ) -> None:
        """
        Train the model with the specified parameters.

        ### Args
        - **resume_from_checkpoint** (bool, optional):
            Whether to resume training from a checkpoint. Defaults to `False`.
        - **learning_rate** (float, optional):
            The learning rate for the optimizer. Defaults to `3e-5`.
        - **weight_decay** (float, optional):
            The weight decay for the optimizer. Defaults to `1e-6`.
        - **train_batch_size** (int, optional):
            The batch size for training. Defaults to `16`.
        - **eval_batch_size** (int, optional):
            The batch size for evaluation. Defaults to `8`.
        - **gradient_accumulation_steps** (int, optional):
            The number of gradient accumulation steps. Defaults to `1`.

        ### Returns
        None:
        This function does not return a value.

        ### Raises
        - **NotImplementedError**: Must be implemented in subclass.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def evaluate(self) -> None:
        """
        Evaluates the model using the evaluation dataset.

        ### Args
        None

        ### Returns
        None

        ### Raises
        - **ValueError**: If no evaluation dataset is loaded.
        """
        trainer = self.build_trainer()
        if trainer.eval_dataset is None:
            raise ValueError("No evaluation dataset loaded")
        logger.info("Evaluating model")
        eval_results = trainer.evaluate(ignore_keys=["predictions"])
        logger.info("Finished evaluating model")
        logger.info(f"Evaluation Results:\n{eval_results}")

    def print_gpu_stats(self):
        """
        Prints the GPU statistics including the name, total memory, and reserved memory.

        ### Args
        None

        ### Returns
        None

        ### Raises
        None
        """
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"Using GPU: {self.device}")
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        logger.info(f"{start_gpu_memory} GB of memory reserved.")

    def free_memory(self):
        """
        Frees up memory by setting model and tokenizer to None and clearing cache.

        ### Args
        None

        ### Returns
        None

        ### Raises
        None
        """
        self._model = None
        self._tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
