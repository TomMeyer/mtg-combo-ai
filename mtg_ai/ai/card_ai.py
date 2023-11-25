from logging import getLogger
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from mtg_ai import constants
import torch
import pandas as pd
from torch.utils.data import DataLoader
from mtg_ai.data import MTGCards
import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm, trange
from pathlib import Path
from tqdm.contrib.logging import logging_redirect_tqdm
from mtg_ai import constants
from mtg_ai.data import MTGCardDataset
from torch.optim.adamw import AdamW



logger = getLogger(__name__)


class MTGCardAI:
    def __init__(self, max_length: int = 50, train_mode: bool = True) -> None:
        self.max_length = max_length
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = constants.PAD_TOKEN
        self.model_path = Path("./data/mtg_card_model")
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            "Intel/neural-chat-7b-v3", device_map="auto", load_in_8bit=True
        )
        self.model.eval()

    def train(
        self,
        num_epochs: int = 3,
        batch_size: int = 1,
        use_cuda: bool = True,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        data_loader: DataLoader = DataLoader(
            MTGCardDataset(), batch_size=batch_size, shuffle=True
        )
        optimizer: AdamW = AdamW(self.model.parameters(), lr=5e-5)
        # self.model.to(device)
        self.model.train()
        with logging_redirect_tqdm():
            for epoch in (pbar := trange(num_epochs, desc="Training")):
                pbar.set_postfix_str(f"Epoch: {epoch}")
                self._train_epoch(data_loader, optimizer)
    
    def _train_epoch(self, data_loader: DataLoader, optimizer: AdamW):
        for batch in tqdm(data_loader, desc="training on batch"):
            # Move batch to device
            input_ids = batch[constants.INPUT_IDS]#.to(device)
            attention_mask = batch[constants.ATTENTION_MASK]#.to(device)
            labels = batch[constants.LABELS]#.to(device)
            # Forward pass
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def save(self):
        save_model_path = Path("./data/mtg_card_model")
        self.model.save_pretrained(save_model_path)

    def evaluate(self, text: str) -> tuple[str, str]:
        encoding = self.tokenizer(
            text=text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        tokenized_data = {
            constants.INPUT_IDS: encoding[constants.INPUT_IDS],
            constants.ATTENTION_MASK: encoding[constants.ATTENTION_MASK],
            constants.LABELS: encoding[constants.INPUT_IDS],
        }
        with torch.no_grad():
            outputs = self.model.generate(
                tokenized_data[constants.INPUT_IDS],
                attention_mask=tokenized_data[constants.ATTENTION_MASK],
                labels=tokenized_data[constants.LABELS],
                max_length=1024,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated_text: str = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            prompt, generated_text = generated_text.rsplit(" |", 1)[0].split(
                " | ", maxsplit=1
            )
            return prompt, generated_text

    def parse_output(self, generated_text: str) -> pd.Series:
        result = {}
        for field in generated_text.split(" | "):
            field_name, value = field.split(":")
            field_name = field_name.strip()
            value = value.strip()
            result[field_name] = value
        return pd.Series(result)

    def evaluate_to_pandas(self, text: str) -> tuple[str, pd.Series]:
        prompt, output = self.evaluate(text)
        parsed_output = self.parse_output(output)
        return prompt, parsed_output


def calculate_accuracy():
    mtg_ai = MTGCardAI()
    for row in MTGCards.iterrows():
        _, output = mtg_ai.evaluate_to_pandas(row.name)
