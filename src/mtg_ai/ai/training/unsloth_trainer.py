import datetime
from logging import DEBUG, getLogger
from multiprocessing import cpu_count

import torch
from peft.peft_model import PeftModel
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from trl import SFTConfig, SFTTrainer

from mtg_ai.ai.training.base_trainer import BaseTrainer

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
        learning_rate: float = 3e-5,
        weight_decay: float = 1e-6,
        train_batch_size: int = 16,
        eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
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

        logging_steps = len(self.data_loader.train_dataset) // (10 * train_batch_size)
        logging_steps = max(logging_steps, 5)
        logger.info(f"setting logging steps to {logging_steps}")

        logger.info(f"Can use bfloat16: {is_bfloat16_supported()}")
        training_args = SFTConfig(
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=1,  # Set this for 1 full training run.
            max_steps=-1,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            weight_decay=weight_decay,
            lr_scheduler_type="linear",
            logging_steps=logging_steps,
            output_dir=f"outputs/{self.output_name}",
            log_level="info",
            logging_dir=f"./logs/{self.model_name}/{self.output_name}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            report_to="tensorboard",
            push_to_hub=False,
            disable_tqdm=False,
            overwrite_output_dir=True,
            include_tokens_per_second=True,
            do_eval=True,
            dataloader_num_workers=12,
            max_seq_length=self.max_seq_length,
            dataset_text_field="text",
        )

        model = self.model
        tokenizer = self.data_loader.tokenizer
        train_dataset = self.data_loader.train_dataset
        eval_dataset = self.data_loader.test_dataset
        trainer: SFTTrainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            dataset_num_proc=cpu_count() - 1,
            packing=False,
            args=training_args,  # type: ignore
        )
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
        return trainer

    def train(
        self,
        resume_from_checkpoint: bool = False,
        learning_rate: float = 3e-5,
        weight_decay: float = 1e-6,
        train_batch_size: int = 16,
        eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
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
        trainer_instance.train()
        ```
        """
        logger.info("Starting training")

        trainer = self.build_trainer(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        if trainer.train_dataset is None:
            raise ValueError("No training dataset loaded")

        if logger.level == DEBUG:
            logger.debug(
                (
                    "Check our dataset applied masking"
                    f"{self.tokenizer.decode(trainer.train_dataset[5]['input_ids'])}"
                )
            )
            space = self.tokenizer(" ", add_special_tokens=False).input_ids[0]
            d = [space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]
            logger.debug(self.tokenizer.decode(d))

        self.print_gpu_stats()

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)  # type: ignore
        logger.info("Saving model")
        self.model.save_pretrained(
            f"./results/{self.output_name}",
        )
        trainer.save_model(f"./results/{self.output_name}")

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(
            f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
        )
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(
            f"Peak reserved memory for training % of max memory = {lora_percentage} %."
        )

        logger.info("Model saved to ./results")
        logger.info("Training complete")
        logger.info("Model is now ready to be used for inference")
        # self.free_memory()

    # def save_model_gguf(self, quantization_type: QuantizationType = "q8_0"):
    #     logger.info(f"Saving model to GGUF with quantization {quantization_type}")
    #     self.model.save_pretrained_gguf(
    #         f"./results/{self.output_name}-gguf-{quantization_type}",
    #         self.tokenizer,
    #         quantization_method=quantization_type,
    #     )
