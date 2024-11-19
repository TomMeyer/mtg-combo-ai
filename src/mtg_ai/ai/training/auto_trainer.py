import logging

import torch

from mtg_ai.ai.training.fsdp_trainer import MTGCardAITrainerFSDP
from mtg_ai.ai.training.unsloth_trainer import MTGCardAITrainerUnsloth

logger = logging.getLogger(__name__)


class AutoTrainer:
    def __init__(
        self,
        model_name: str,
        datasets: list[str],
        output_name: str,
        max_seq_length: int = 500,
    ) -> None:
        self.gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {self.gpu_count} GPUs")
        if self.gpu_count > 1:
            logger.info("Using FSDP Trainer")
            self._trainer = MTGCardAITrainerFSDP(
                model_name=model_name,
                datasets=datasets,
                output_name=output_name,
                max_seq_length=max_seq_length,
            )
        else:
            logger.info("Using UnSloth Trainer")
            self._trainer = MTGCardAITrainerUnsloth(
                model_name=model_name,
                datasets=datasets,
                output_name=output_name,
                max_seq_length=max_seq_length,
            )

    def train(
        self,
        resume_from_checkpoint: bool = False,
        learning_rate: float = 3e-5,
        weight_decay: float = 1e-6,
        train_batch_size: int = 16,
        eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        self._trainer.train(
            resume_from_checkpoint=resume_from_checkpoint,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    def evaluate(self) -> None:
        self._trainer.evaluate()
