import datetime
from dataclasses import dataclass, field


@dataclass
class MTGAITrainingConfig:
    enable_profiling: bool = False
    output_name: str = "mtg-ai"
    model_id: str = "unsloth/Llama-3.1-Nemotron-70B-Instruct"
    datasets: list[str] = field(
        default_factory=lambda: [
            "FringeFields/MTG-Rules-QA",
            "FringeFields/MTG-Cards-QA",
            "FringeFields/MTG-EDH-Combos-QA",
        ]
    )

    def __post_init__(self) -> None:
        self._timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    @property
    def output_dir(self):
        return f"./training_results/{self.output_name}/"

    @property
    def log_dir(self):
        return f"./logs/{self.output_name}/{self._timestamp}"

    @property
    def model_output_dir(self):
        return f"./results/{self.output_name}/"
