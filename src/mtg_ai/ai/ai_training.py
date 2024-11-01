import datetime
import gc
from logging import DEBUG, getLogger
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import Dataset, DatasetDict
from peft.peft_model import PeftModel
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizer, TrainingArguments
from trl import SFTTrainer
from unsloth import (
    FastLanguageModel,
    is_bfloat16_supported,
)
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from mtg_ai.cards.training_data_builder import QUESTION_ANSWER_FOLDER, MTGDatasetLoader

logger = getLogger(__name__)


class MTGCardAITrainingDatasetLoader:
    def __init__(
        self,
        tokenizer,
        dataset_name: str,
        num_procs: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        dataset: Dataset = MTGDatasetLoader.load_dataset(dataset_name)

        def format_prompt(examples) -> dict[str, Any]:
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


class MTGCardAITrainer:
    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        max_seq_length: int = 300,
    ) -> None:
        # set properties
        self._model: Optional[PeftModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self.tokenizer_name = model_name
        self.max_seq_length = max_seq_length
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._data_loader = None
        self.dataset_name = dataset_name
        self.load_model()

    @property
    def data_loader(self) -> MTGCardAITrainingDatasetLoader:
        if not self._data_loader:
            self._data_loader = MTGCardAITrainingDatasetLoader(
                tokenizer=self.tokenizer, dataset_name=self.dataset_name
            )
        return self._data_loader

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
        return self._tokenizer

    def load_model(self):
        logger.info(f"loading model {self.model_name}")
        model, tokenizer = self._get_model_and_tokenizer(
            self.model_name, self.max_seq_length
        )
        self._model = model
        self._tokenizer = tokenizer

    @classmethod
    def _get_model_and_tokenizer(
        cls, model_name: str, max_sequence_length: int = 500
    ) -> tuple[PeftModel, PreTrainedTokenizer]:
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
        logging_steps = len(self.data_loader.train_dataset) // (10 * train_batch_size)
        logger.info(f"setting logging steps to {logging_steps}")

        logger.info(f"Can use bfloat16: {is_bfloat16_supported()}")
        training_args = TrainingArguments(
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
            output_dir=f"outputs/{self.dataset_name}_training",
            log_level="info",
            logging_dir=f"./logs/{self.model_name}/{self.dataset_name}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            report_to="tensorboard",
            push_to_hub=False,
            disable_tqdm=False,
            overwrite_output_dir=True,
            include_tokens_per_second=True,
            do_eval=True,
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

    def print_gpu_stats(self):
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"Using GPU: {self.device}")
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        logger.info(f"{start_gpu_memory} GB of memory reserved.")

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
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)  # type: ignore
        logger.info("Saving model")
        self.model.save_pretrained(
            f"./results/{self.dataset_name}",
        )
        logger.info("Model saved to ./results")
        logger.info("Training complete")
        logger.info("Model is now ready to be used for inference")
        self.free_memory()

    def evaluate(self) -> None:
        trainer = self.build_trainer()
        if trainer.eval_dataset is None:
            raise ValueError("No evaluation dataset loaded")
        logger.info("Evaluating model")
        eval_results = trainer.evaluate(ignore_keys=["predictions"])
        logger.info("Finished evaluating model")
        logger.info(f"Evaluation Results:\n{eval_results}")

    def free_memory(self):
        self._model = None
        self._tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()


class MTGCardAITrainerPipeline:
    def __init__(
        self, base_model_name: str, dataset_directory: Path = QUESTION_ANSWER_FOLDER
    ) -> None:
        MTGDatasetLoader.set_directory(dataset_directory)
        self.dataset_order = MTGDatasetLoader.dataset_order
        self.base_model_name = base_model_name

    def dataset_name_to_model_name(self, dataset_name: str) -> str:
        order = self.dataset_order.index(dataset_name)
        if order == 0:
            return self.base_model_name
        prev_dataset_name = self.dataset_order[order - 1]
        return f"./results/{prev_dataset_name}"

    def train(
        self,
        dataset_name: str,
        resume_from_checkpoint: bool = False,
        learning_rate: float = 3e-5,
        weight_decay: float = 1e-6,
        train_batch_size: int = 16,
        eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        if dataset_name not in self.dataset_order:
            raise ValueError(
                f"Dataset {dataset_name} is not a valid dataset name, must be one of {self.dataset_order}"
            )
        model_name = self.dataset_name_to_model_name(dataset_name)
        logger.info(f"Training {dataset_name}")
        MTGCardAITrainer(dataset_name=dataset_name, model_name=model_name).train(
            resume_from_checkpoint=resume_from_checkpoint,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        logger.info(f"Finished training {dataset_name}")

    def evaluate(self, dataset_name: str) -> None:
        logger.info(f"Evaluating {dataset_name}")
        model_name = self.dataset_name_to_model_name(dataset_name)
        MTGCardAITrainer(dataset_name=dataset_name, model_name=model_name).evaluate()
        logger.info(f"Finished evaluating {dataset_name}")

    def train_all(self) -> None:
        logger.info("Starting training all datasets")
        for dataset_name in self.dataset_order:
            self.train(dataset_name)
        logger.info("Finished training all datasets")
