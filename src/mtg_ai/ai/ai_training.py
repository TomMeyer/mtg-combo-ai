import datetime
import os
from logging import DEBUG, getLogger
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft.peft_model import PeftModel
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from mtg_ai.cards.training_data_builder import read_mtg_dataset_from_disk

logger = getLogger(__name__)

PathLike = str | Path | os.PathLike[str]


class MTGCardAITrainingDatasetLoader:
    def __init__(
        self,
        tokenizer,
        num_procs: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        dataset = read_mtg_dataset_from_disk()

        def format_prompt(examples):
            texts = self.tokenizer.apply_chat_template(
                conversation=examples["conversations"],
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": texts}

        logger.info(f"Tokenizing dataset with {num_procs} processes")
        formatted_dataset = dataset.map(
            format_prompt, num_proc=num_procs or cpu_count() - 1, batched=True
        )  # type: ignore
        logger.info("finished tokenizing dataset")

        self.dataset = formatted_dataset.train_test_split(test_size=0.2)  # type: ignore

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


class MTGCardAITrainer:
    def __init__(
        self,
        # dataset_loader: MTGCardAITrainingDatasetLoader,
        model_name: str,
        gguf_file: Optional[str] = None,
        num_epochs: int = 3,
        max_seq_length: int = 100,
    ) -> None:
        # set properties
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.max_seq_length = max_seq_length
        self.model_name = model_name
        self.tokenizer_name = model_name
        self.gguf_file = gguf_file
        # load model, tokenizer, and dataloader
        model, tokenizer = self._get_model_and_tokenizer(model_name, max_seq_length)
        self.model: PeftModel = model
        self.data_loader = MTGCardAITrainingDatasetLoader(tokenizer=tokenizer)

    @classmethod
    def _get_model_and_tokenizer(cls, model_name: str, max_sequence_length: int = 2048):
        logger.info(f"loading model and tokenizer {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            max_seq_length=max_sequence_length,
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
            use_gradient_checkpointing="unsloth",
            random_state=3047,
            max_seq_length=max_sequence_length,
            use_rslora=False,
            loftq_config=None,
        )
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
        return model, tokenizer

    @property
    def tokenizer(self):
        return self.data_loader.tokenizer

    def build_trainer(self) -> SFTTrainer:
        logger.info(f"Can use bfloat16: {is_bfloat16_supported()}")
        training_args = TrainingArguments(
            per_device_train_batch_size=32,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            warmup_steps=5,
            num_train_epochs=1,  # Set this for 1 full training run.
            max_steps=-1,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            logging_steps=25,
            output_dir="outputs",
            log_level="info",
            logging_dir=f"./logs/{self.model_name}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
            report_to="tensorboard",
            push_to_hub=False,
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
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            dataset_num_proc=9,
            packing=False,
            args=training_args,  # type: ignore
        )
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
        return trainer

    def print_gpu_stats(self):
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"Using GPU: {self.device}")
        logger.debug(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        logger.debug(f"{start_gpu_memory} GB of memory reserved.")

    def train(self, resume_from_checkpoint: bool = False) -> None:
        logger.info("Starting training")
        logger.info(f"Training {self.model} for {self.num_epochs} epochs")

        trainer = self.build_trainer()

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
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)  # type: ignore
        logger.info("Training complete")
        logger.info("Evaluating model")
        eval_results = trainer.evaluate(ignore_keys=["predictions"])
        logger.info(eval_results)
        logger.info("Saving model")
        self.model.save_pretrained(
            "./results",
        )
        logger.info("Model saved to ./results")
        logger.info("Finished saving model")
        logger.info("Model is now ready")


# def save_combined_model(
#     model_name: str,
#     model_dir: Path,
#     tokenizer_name: str,
#     gguf_file: Optional[str] = None,
# ) -> None:
#     merged_model_output = model_dir.joinpath("merged_model")

#     logger.info("Loading model")
#     # model = AutoModelForCausalLM.from_pretrained(model_name, gguf_file=gguf_file)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info("Loading tokenizer")
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

#     model: PeftModel = get_peft_model(model, str(model_dir))
#     model = model.to(device)

#     for name, param in model.named_parameters():
#         logger.debug(f"Layer: {name}, dtype: {param.dtype}, shape: {param.shape}")

#     logger.info("Merging model")
#     model = model.merge_and_unload()
#     logger.info("Saving model")
#     model.save_pretrained(str(merged_model_output))
#     logger.info("Saving tokenizer")
#     tokenizer.save_pretrained(str(merged_model_output))
#     logger.info("Finished saving model")
