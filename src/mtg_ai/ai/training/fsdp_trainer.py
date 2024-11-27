import datetime
from logging import DEBUG, getLogger

import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from peft import LoraConfig, PeftMixedModel, PeftModel, PeftType, get_peft_model
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
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

    python
        trainer = MTGCardAITrainerFSDP(
            model,
            tokenizer,
        )
        trainer.train()

    """

    def __init__(
        self,
        model_name: str,
        datasets: list[str],
        output_name: str,
        max_seq_length: int = 300,
    ) -> None:
        self.fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(
                offload_to_cpu=True, rank0_only=False
            ),
            optim_state_dict_config=FullOptimStateDictConfig(
                offload_to_cpu=True, rank0_only=False
            ),
            use_orig_params=True,
            cpu_offload=False,
            activation_checkpointing=True,
            cpu_ram_efficient_loading=True,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            auto_wrap_policy="transformer_based_wrap",
            forward_prefetch=False,
        )
        torch.distributed.init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=5400)
        )
        self.accelerator = Accelerator(
            mixed_precision="bf16",
            fsdp_plugin=self.fsdp_plugin,
            split_batches=True,
            gradient_accumulation_steps=32,
        )
        self.accelerator.print(
            "Cuda available", torch.cuda.is_available()
        )  # Should return True if CUDA is available
        self.accelerator.print(
            "device count", torch.cuda.device_count()
        )  # Number of GPUs available
        self.accelerator.print(
            "current device", torch.cuda.current_device()
        )  # Number of GPUs available
        self.accelerator.print("distributed type", self.accelerator.distributed_type)

        super().__init__(model_name, datasets, output_name, max_seq_length)

    @classmethod
    def _get_model_and_tokenizer(
        cls, model_name: str, max_sequence_length: int = 500
    ) -> tuple[
        PeftModel | PeftMixedModel, PreTrainedTokenizer | PreTrainedTokenizerFast
    ]:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            AutoTokenizer.from_pretrained(model_name)
        )
        bits_and_bytes_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        print("current_device", torch.cuda.current_device())
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bits_and_bytes_config,
            device_map={"": torch.cuda.current_device()},
            torch_dtype=torch.bfloat16,
            use_cache=True,
            attn_implementation="flash_attention_2",
        )
        if model is None:
            raise ValueError("Model not found")
        lora_config = LoraConfig(
            peft_type=PeftType.LORA,
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
            use_dora=False,
        )
        model = get_peft_model(
            model,
            peft_config=lora_config,
            adapter_name="default",
        )
        # model.gradient_checkpointing_enable()
        return model, tokenizer

    def build_trainer(
        self,
        learning_rate: float = 3e-5,
        weight_decay: float = 1e-6,
        train_batch_size: int = 16,
        eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
    ) -> Trainer:
        # Unsloth import has side effects, import here to improve performance

        logging_steps = len(self.data_loader.train_dataset) // (10 * train_batch_size)
        logging_steps = max(logging_steps, 5)
        if self.accelerator.is_main_process:
            logger.info(f"setting logging steps to {logging_steps}")
            logger.info(f"setting per_device_train_batch_size to {train_batch_size}")
            logger.info(
                f"setting gradient_accumulation_steps to {gradient_accumulation_steps}"
            )
        training_args = SFTConfig(
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=1,  # Set this for 1 full training run.
            max_steps=-1,
            learning_rate=learning_rate,
            fp16=False,
            bf16=True,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            output_dir=f"outputs/{self.output_name}",
            log_level="info",
            logging_dir=f"./logs/{self.model_name}/{self.output_name}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            report_to="tensorboard",
            push_to_hub=False,
            disable_tqdm=False,
            overwrite_output_dir=True,
            include_tokens_per_second=True,
            dataloader_num_workers=12,
            # gradient_checkpointing=True,
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
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            data_collator=data_collator,
            packing=False,
            args=training_args,
        )

        # trainer: Trainer = Trainer(
        #     model=model,
        #     args=training_args,
        #     train_dataset=train_dataset,
        #     eval_dataset=eval_dataset,
        #     data_collator=data_collator,
        # )
        return trainer

    def print_parameter_device(self) -> None:
        for name, param in self.model.named_parameters():
            print(name, param.device)

    def train(
        self,
        resume_from_checkpoint: bool = False,
        learning_rate: float = 3e-5,
        weight_decay: float = 1e-6,
        train_batch_size: int = 1,
        eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 64,
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
