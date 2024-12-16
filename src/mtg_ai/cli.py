# import argparse
# from pathlib import Path

# from mtg_ai.ai import ModelAndTokenizer, MTGCardAITrainer
# from mtg_ai.ai.ai_training import QUESTION_ANSWER_FOLDER
# from mtg_ai.cards.training_data_builder import build_datasets


# def build_datasets_command(all_merged):
#     """Build the datasets for training."""
#     build_datasets(directory=QUESTION_ANSWER_FOLDER, all_merged=all_merged)
#     print("Datasets built successfully.")


# def train_command(
#     dataset_name,
#     resume_from_checkpoint,
#     learning_rate,
#     weight_decay,
#     train_batch_size,
#     eval_batch_size,
#     gradient_accumulation_steps,
# ) -> None:
#     """Run the training process."""
#     model = ModelAndTokenizer.UNSLOTH_LLAMA_3_2_3B_INSTRUCT_Q8
#     training_pipeline = MTGCardAITrainer(
#         base_model_name=model.value, dataset_directory=QUESTION_ANSWER_FOLDER
#     )
#     training_pipeline.train(
#         dataset_name=dataset_name,
#         resume_from_checkpoint=resume_from_checkpoint,
#         learning_rate=learning_rate,
#         weight_decay=weight_decay,
#         train_batch_size=train_batch_size,
#         eval_batch_size=eval_batch_size,
#         gradient_accumulation_steps=gradient_accumulation_steps,
#     )
#     print(f"Training on {dataset_name} completed successfully.")


# def setup_logging(level: str):
#     import logging

#     logging.basicConfig(level=level)


# def cli_main():
#     parser = argparse.ArgumentParser(description="MTG AI CLI")
#     subparsers = parser.add_subparsers(dest="command")

#     # Build command
#     build_parser = subparsers.add_parser(
#         "build-dataset", help="Build the datasets for training."
#     )
#     build_parser.add_argument(
#         "--all-merged", action="store_true", help="Build all datasets merged into one."
#     )
#     build_parser.add_argument(
#         "--path",
#         "-p",
#         type=Path,
#         default=QUESTION_ANSWER_FOLDER,
#         help="Path to the dataset directory.",
#     )
#     build_parser.add_argument(
#         "-v", "--verbose", action="count", default=0, help="Increase output verbosity"
#     )

#     # Train command
#     train_parser = subparsers.add_parser("train", help="Run the training process.")
#     train_parser.add_argument(
#         "--dataset-name", required=True, help="Name of the dataset to train on."
#     )
#     train_parser.add_argument(
#         "--resume-from-checkpoint",
#         action="store_true",
#         help="Resume training from checkpoint.",
#     )
#     train_parser.add_argument(
#         "--learning-rate", type=float, default=3e-5, help="Learning rate for training."
#     )
#     train_parser.add_argument(
#         "--weight-decay", type=float, default=1e-6, help="Weight decay for training."
#     )
#     train_parser.add_argument(
#         "--train-batch-size", type=int, default=16, help="Training batch size."
#     )
#     train_parser.add_argument(
#         "--eval-batch-size", type=int, default=8, help="Evaluation batch size."
#     )
#     train_parser.add_argument(
#         "--gradient-accumulation-steps",
#         type=int,
#         default=1,
#         help="Gradient accumulation steps.",
#     )
#     train_parser.add_argument(
#         "-v", "--verbose", action="count", default=0, help="Increase output verbosity"
#     )

#     args = parser.parse_args()

#     if args.verbose:
#         level = "DEBUG" if args.verbose > 1 else "INFO"
#         setup_logging(level=level)

#     if args.command == "build-dataset":
#         build_datasets_command(args.all_merged)
#     elif args.command == "train":
#         train_command(
#             args.dataset_name,
#             args.resume_from_checkpoint,
#             args.learning_rate,
#             args.weight_decay,
#             args.train_batch_size,
#             args.eval_batch_size,
#             args.gradient_accumulation_steps,
#         )
#     else:
#         parser.print_help()


def cli_main():
    raise NotImplementedError("CLI is not implemented yet.")


if __name__ == "__main__":
    cli_main()
