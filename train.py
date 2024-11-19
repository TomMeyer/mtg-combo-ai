import logging

import colorlog

from mtg_ai.ai import AutoTrainer


def main():
    handler = colorlog.StreamHandler()
    fmt = "%(log_color)s%(levelname)s:%(name)s:%(message)s"
    formatter = colorlog.ColoredFormatter(
        fmt,
        log_colors={
            "DEBUG": "purple",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logger = logging.getLogger("mtg-ai")
    logger.propagate = True

    model_name = "unsloth/Llama-3.1-Nemotron-70B-Instruct"
    datasets = [
        "FringeFields/MTG-Rules-QA",
        "FringeFields/MTG-Cards-QA",
        "FringeFields/MTG-EDH-Combos-QA",
    ]
    trainer = AutoTrainer(
        model_name=model_name,
        datasets=datasets,
        max_seq_length=500,
        output_name="mtg-ai",
    )
    trainer.train(resume_from_checkpoint=False)


if __name__ == "__main__":
    main()
