from logging import getLogger

from peft.peft_model import PeftModel
from transformers import PreTrainedTokenizer
from trl import SFTTrainer

from mtg_ai.ai.training.base_trainer import BaseTrainer

logger = getLogger(__name__)


class MTGCardAITrainerFSDP(BaseTrainer):
    @classmethod
    def _get_model_and_tokenizer(
        cls, model_name: str, max_sequence_length: int = 500
    ) -> tuple[PeftModel, PreTrainedTokenizer]:
        raise NotImplementedError("TODO")

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
