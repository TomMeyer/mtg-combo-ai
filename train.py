import os
from typing import cast

if "NCCL_DEBUG" not in os.environ:
    os.environ["NCCL_DEBUG"] = "INFO"

import logging

from rich.console import Console
from rich.logging import RichHandler
from trl import SFTConfig
from trl.commands.cli_utils import TrlParser

from mtg_ai.ai.training import MTGAITrainingConfig
from mtg_ai.ai.training.auto_trainer import AutoTrainer


def build_parser():
    trl_parser = TrlParser((MTGAITrainingConfig, SFTConfig))
    mtg_ai_config, sft_config = trl_parser.parse_args_and_config()
    return mtg_ai_config, sft_config


def main():
    # fmt = "%(log_color)s%(levelname)s:%(name)s:%(message)s"
    mtg_ai_config, sft_config = build_parser()

    mtg_ai_config = cast(MTGAITrainingConfig, mtg_ai_config)

    sft_config = cast(SFTConfig, sft_config)
    sft_config.logging_dir = mtg_ai_config.log_dir
    sft_config.output_dir = mtg_ai_config.model_output_dir

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            RichHandler(
                level=logging.INFO,
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                show_path=False,
                console=Console(),
            )
        ],
    )
    logger = logging.getLogger("mtg-ai")
    logger.propagate = True

    trainer = AutoTrainer(
        training_config=mtg_ai_config,
        sft_config=sft_config,
    )
    trainer.train()


if __name__ == "__main__":
    main()
