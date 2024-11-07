from logging import getLogger

from peft import LoraConfig, PeftModel, PeftType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from trl import SFTTrainer

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
        model = AutoModelForCausalLM.from_pretrained(model_name)

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
        get_peft_model(
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
        raise NotImplementedError("TODO")

    def train(
        self,
        resume_from_checkpoint: bool = False,
        learning_rate: float = 3e-5,
        weight_decay: float = 1e-6,
        train_batch_size: int = 16,
        eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        raise NotImplementedError("TODO")
