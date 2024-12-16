import logging

import torch
from trl import SFTConfig

from mtg_ai.ai.training.config import MTGAITrainingConfig
from mtg_ai.ai.training.fsdp_trainer import MTGCardAITrainerFSDP
from mtg_ai.ai.training.unsloth_trainer import MTGCardAITrainerUnsloth

logger = logging.getLogger(__name__)


class AutoTrainer:
    def __init__(
        self, sft_config: SFTConfig, training_config: MTGAITrainingConfig
    ) -> None:
        self.sft_config = sft_config
        self.training_config = training_config
        self.gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {self.gpu_count} GPUs")
        if self.gpu_count > 1:
            logger.info("Using FSDP Trainer")
            self._trainer = MTGCardAITrainerFSDP(
                training_config=training_config, sft_config=sft_config
            )
        else:
            logger.info("Using UnSloth Trainer")
            self._trainer = MTGCardAITrainerUnsloth(
                training_config=training_config, sft_config=sft_config
            )

    def train(self) -> None:
        self._trainer.train()

    def evaluate(self) -> None:
        self._trainer.evaluate()
