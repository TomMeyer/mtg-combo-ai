import datetime
from logging import DEBUG, getLogger

import torch
from peft import LoraConfig, PeftModel, PeftType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from mtg_ai.ai.training.base_trainer import BaseTrainer

logger = getLogger(__name__)


class MTGCardAITrainerFSDP(BaseTrainer):
    """
    Trainer class for MTGCard AI using Fully Sharded Data Parallel (FSDP).

    ### Attributes
    - **model** (PeftModel):
        The model to be trained.
    - **tokenizer** (PreTrainedTokenizer | PreTrainedTokenizerFast):
        The tokenizer for the model.

    ### Methods
    - **_get_model_and_tokenizer**: Loads the model and tokenizer.
    - **build_trainer**: Builds the SFTTrainer with specified parameters.
    - **train**: Trains the model with specified parameters.

    ### Example
    ```python
    trainer = MTGCardAITrainerFSDP(
        model,
        tokenizer,
    )
    trainer.train()
    ```
    """

    @classmethod
    def _get_model_and_tokenizer(
        cls, model_name: str, max_sequence_length: int = 500
    ) -> tuple[PeftModel, PreTrainedTokenizer | PreTrainedTokenizerFast]:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            AutoTokenizer.from_pretrained(model_name)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        lora_config = LoraConfig(
            peft_type=PeftType.ADALORA,
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
            use_rslora=True,
            loftq_config=None,
        )
        model = get_peft_model(
            model,
            peft_config=lora_config,
            adapter_name="default",
        )
        return model, tokenizer

    def build_trainer(
        self,
        learning_rate: float = 3e-5,
        weight_decay: float = 1e-6,
        train_batch_size: int = 16,
        eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
    ) -> SFTTrainer:
        # Unsloth import has side effects, import here to improve performance

        logging_steps = len(self.data_loader.train_dataset) // (10 * train_batch_size)
        logging_steps = max(logging_steps, 5)
        logger.info(f"setting logging steps to {logging_steps}")

        training_args = SFTConfig(
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=1,  # Set this for 1 full training run.
            max_steps=-1,
            learning_rate=learning_rate,
            fp16=False,
            bf16=True,
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
        )

        model = self.model
        tokenizer = self.data_loader.tokenizer
        train_dataset = self.data_loader.train_dataset
        eval_dataset = self.data_loader.test_dataset
        data_collator = DataCollatorForCompletionOnlyLM(
            instruction_template="<|start_header_id|>user<|end_header_id|>\n\n",
            response_template="<|start_header_id|>assistant<|end_header_id|>\n\n",
            tokenizer=tokenizer,
        )

        trainer: SFTTrainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            data_collator=data_collator,
            packing=False,
            args=training_args,
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
