import importlib.util
from logging import getLogger

import torch
from accelerate import FullyShardedDataParallelPlugin
from peft import LoraConfig, PeftMixedModel, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from trl import SFTTrainer

from mtg_ai.ai.training.base_trainer import BaseTrainer
from mtg_ai.ai.utils import ProfilerCallback

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

    # def __init__(
    #     self,
    #     sft_config: SFTConfig,
    #     training_config: MTGAITrainingConfig,
    # ) -> None:
    #     # self.fsdp_plugin = FullyShardedDataParallelPlugin(
    #     #     state_dict_config=FullStateDictConfig(
    #     #         offload_to_cpu=False, rank0_only=False
    #     #     ),
    #     #     optim_state_dict_config=FullOptimStateDictConfig(
    #     #         offload_to_cpu=False, rank0_only=False
    #     #     ),

    #     #     use_orig_params=True,
    #     #     cpu_offload=False,
    #     #     activation_checkpointing=True,
    #     #     cpu_ram_efficient_loading=True,
    #     #     sharding_strategy=ShardingStrategy.FULL_SHARD,
    #     #     state_dict_type=StateDictType.SHARDED_STATE_DICT,
    #     #     auto_wrap_policy="transformer_based_wrap",
    #     #     forward_prefetch=False,
    #     # )
    # super().__init__(sft_config=sft_config, training_config=training_config)

    # torch.distributed.init_process_group(
    #     backend="nccl", timeout=datetime.timedelta(seconds=5400)
    # )
    # self.accelerator = Accelerator(
    #     mixed_precision="bf16",
    #     fsdp_plugin=self.fsdp_plugin,
    #     split_batches=True,
    #     gradient_accumulation_steps=32,
    # )
    #     # self.accelerator.print(
    #     #     "Cuda available", torch.cuda.is_available()
    #     # )  # Should return True if CUDA is available
    #     # self.accelerator.print(
    #     #     "device count", torch.cuda.device_count()
    #     # )  # Number of GPUs available
    #     # self.accelerator.print(
    #     #     "current device", torch.cuda.current_device()
    #     # )  # Number of GPUs available
    #     # self.accelerator.print("distributed type", self.accelerator.distributed_type)

    @classmethod
    def _get_model_and_tokenizer(
        cls, model_name: str, max_sequence_length: int = 500
    ) -> tuple[
        PeftModel | PeftMixedModel, PreTrainedTokenizer | PreTrainedTokenizerFast
    ]:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            AutoTokenizer.from_pretrained(model_name, use_fast=True)
        )
        bits_and_bytes_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        print("current_device", torch.cuda.current_device())

        attn_implementation = (
            "flash_attention_2" if importlib.util.find_spec("flash-attn") else "sdpa"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bits_and_bytes_config,
            device_map={"": torch.cuda.current_device()},
            torch_dtype=torch.bfloat16,
            use_cache=True,
            attn_implementation=attn_implementation,
        )
        if model is None:
            raise ValueError("Model not found")
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=16,
            target_modules="all-linear",
            # target_modules=[
            #     "q_proj",
            #     "k_proj",
            #     "v_proj",
            #     "o_proj",
            #     "gate_proj",
            #     "up_proj",
            #     "down_proj",
            # ],
            lora_dropout=0,
            bias="none",
            use_rslora=True,
            use_dora=False,
        )
        model.enable_input_require_grads()
        model = get_peft_model(
            model,
            peft_config=lora_config,
            adapter_name="default",
        )
        # model.gradient_checkpointing_enable()
        return model, tokenizer

    def build_trainer(self) -> SFTTrainer:
        model = self.model
        tokenizer = self.data_loader.tokenizer
        train_dataset = self.data_loader.train_dataset
        eval_dataset = self.data_loader.test_dataset
        # data_collator = DataCollatorForCompletionOnlyLM(
        #     instruction_template="<|start_header_id|>user<|end_header_id|>\n\n",
        #     response_template="<|start_header_id|>assistant<|end_header_id|>\n\n",
        #     tokenizer=tokenizer,
        # )
        self.sft_config.packing = True
        self.sft_config.save_steps = 0.1
        self.sft_config.save_strategy = "steps"
        logger.info(f"SFTConfig:\n{self.sft_config.to_dict()}")
        trainer: SFTTrainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # data_collator=data_collator,
            args=self.sft_config,
        )
        logger.info(trainer.dataset_batch_size)
        trainer.dataset_num_proc = 24

        if self.training_config.enable_profiling:
            profiler = ProfilerCallback(log_dir=self.training_config.log_dir)
            trainer.add_callback(profiler)
        return trainer

    def train(self) -> None:
        logger.info("building trainer")

        trainer = self.build_trainer()

        if trainer.train_dataset is None:
            raise ValueError("No training dataset loaded")

        logger.info("Starting training")
        logger.info(trainer.accelerator.num_processes)
        if trainer.accelerator.is_main_process:
            trainer.model.print_trainable_parameters()
        trainer.train(resume_from_checkpoint=self.sft_config.resume_from_checkpoint)
        logger.info("Saving model")
        if trainer.is_fsdp_enabled and isinstance(
            trainer.accelerator.state.fsdp_plugin, FullyShardedDataParallelPlugin
        ):
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        trainer.save_model(self.training_config.model_output_dir)

        logger.info(f"Model saved to {self.training_config.model_output_dir}")
        logger.info("Training complete")
