from logging import getLogger

import torch
from peft.peft_model import PeftModel
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from trl import SFTTrainer

from mtg_ai.ai.training.base_trainer import BaseTrainer
from mtg_ai.ai.utils import ProfilerCallback

logger = getLogger(__name__)


class MTGCardAITrainerUnsloth(BaseTrainer):
    """
    Trainer class for MTGCard AI using Unsloth.

    ### Attributes
    Inherits all attributes from BaseTrainer.

    ### Methods
    - **_get_model_and_tokenizer**: Loads the model and tokenizer.
    - **build_trainer**: Builds the SFTTrainer with specified parameters.
    - **train**: Trains the model with specified parameters.

    ### Example
    ```python
    trainer = MTGCardAITrainerUnsloth(
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

    @classmethod
    def _get_model_and_tokenizer(
        cls, model_name: str, max_sequence_length: int = 500
    ) -> tuple[PeftModel, PreTrainedTokenizer | PreTrainedTokenizerFast]:
        """
        Loads the model and tokenizer based on the provided model name and maximum sequence length.

        ### Args
        - **model_name** (str):
            The name of the model to load.
        - **max_sequence_length** (int, optional):
            The maximum sequence length for the model. Defaults to `500`.

        ### Returns
        tuple[PeftModel, PreTrainedTokenizer | PreTrainedTokenizerFast]:
            A tuple containing the loaded model and tokenizer.

        ### Raises
        - **RuntimeError**: If the model or tokenizer could not be loaded.

        ### Example
        ```python
        model, tokenizer = (
            MTGCardAITrainerUnsloth._get_model_and_tokenizer(
                "model_name"
            )
        )
        ```
        """
        # Unsloth import has side effects, import here to improve performance
        from unsloth import FastLanguageModel, get_chat_template

        logger.info(f"loading model and tokenizer {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            max_seq_length=max_sequence_length,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",  # type: ignore # this is a weird use by unsloth
            max_seq_length=max_sequence_length,
            use_rslora=True,
            loftq_config=None,
        )
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
        return model, tokenizer

    def build_trainer(
        self,
    ) -> SFTTrainer:
        """
        Builds the SFTTrainer with the specified training parameters.

        ### Args
        - **learning_rate** (float, optional):
            The learning rate for training. Defaults to `3e-5`.
        - **weight_decay** (float, optional):
            The weight decay for training. Defaults to `1e-6`.
        - **train_batch_size** (int, optional):
            The batch size for training. Defaults to `16`.
        - **eval_batch_size** (int, optional):
            The batch size for evaluation. Defaults to `8`.
        - **gradient_accumulation_steps** (int, optional):
            The number of gradient accumulation steps. Defaults to `1`.

        ### Returns
        SFTTrainer:
            The configured SFTTrainer instance.

        ### Raises
        None

        ### Example
        ```python
        trainer = trainer_instance.build_trainer()
        ```
        """
        # Unsloth import has side effects, import here to improve performance
        from unsloth import is_bfloat16_supported, train_on_responses_only

        logger.info(f"Can use bfloat16: {is_bfloat16_supported()}")
        model = self.model
        tokenizer = self.data_loader.tokenizer
        train_dataset = self.data_loader.train_dataset
        eval_dataset = self.data_loader.test_dataset

        trainer: SFTTrainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            args=self.sft_config,
        )

        if self.training_config.enable_profiling:
            profiler = ProfilerCallback(log_dir=self.training_config.log_dir)
            trainer.add_callback(profiler)

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
        return trainer

    def train(
        self,
    ) -> None:
        """
        Trains the model with the specified parameters.

        ### Args
        - **resume_from_checkpoint** (bool, optional):
            Whether to resume training from a checkpoint. Defaults to `False`.
        - **learning_rate** (float, optional):
            The learning rate for training. Defaults to `3e-5`.
        - **weight_decay** (float, optional):
            The weight decay for training. Defaults to `1e-6`.
        - **train_batch_size** (int, optional):
            The batch size for training. Defaults to `16`.
        - **eval_batch_size** (int, optional):
            The batch size for evaluation. Defaults to `8`.
        - **gradient_accumulation_steps** (int, optional):
            The number of gradient accumulation steps. Defaults to `1`.

        ### Returns
        None

        ### Raises
        - **ValueError**: If no training dataset is loaded.

        ### Example
        ```python
        trainer_instance.train(
            resume_from_checkpoint=False
        )
        ```
        """

        trainer: SFTTrainer = self.build_trainer()
        trainer.model.print_trainable_parameters()

        logger.info("EHRE")
        logger.info(len(trainer.train_dataset.data["input_ids"]))
        logger.info(
            trainer.tokenizer.decode(trainer.train_dataset.data["input_ids"][0].as_py())
        )
        logger.info(trainer.train_dataset[0])
        logger.info("NOT HERE")

        logger.info("Starting training")
        trainer.train(resume_from_checkpoint=self.sft_config.resume_from_checkpoint)

        logger.info("Saving model")
        trainer.save_model(self.training_config.model_output_dir)

        logger.info(f"Model saved to {self.training_config.model_output_dir}")
        logger.info("Training complete")
