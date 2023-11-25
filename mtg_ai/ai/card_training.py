from mtg_ai.data.tokenize_cards import MTGCardDataset
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoModel, AutoModelForCausalLM
from torch.optim.adamw import AdamW
from tqdm import tqdm, trange
from pathlib import Path
from tqdm.contrib.logging import logging_redirect_tqdm
from logging import getLogger
from mtg_ai import constants

logger = getLogger(__name__)


class MTGCardTraining:
    def __init__(self, num_epochs: int = 3, batch_size: int = 2, use_cuda: bool = True):
        self.use_cuda: bool = use_cuda
        self.batch_size: int = batch_size
        self.num_epochs: int = num_epochs
        self.card_dataset: MTGCardDataset = MTGCardDataset()
        self.data_loader: DataLoader = DataLoader(
            self.card_dataset, batch_size=batch_size, shuffle=True
        )
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained("Intel/neural-chat-7b-v3", device_map="auto", load_in_4bit=True)
        self.optimizer: AdamW = AdamW(self.model.parameters(), lr=5e-5)

    def train(self):
        device = torch.device(
            "cuda" if torch.cuda.is_available() and self.use_cuda else "cpu"
        )
        # self.model.to(device)
        self.model.train()
        with logging_redirect_tqdm():
            for epoch in (pbar := trange(self.num_epochs, desc="Training")):
                pbar.set_postfix_str(f"Epoch: {epoch}")
                for batch in tqdm(self.data_loader, desc="training on batch"):
                    # Move batch to device
                    input_ids = batch[constants.INPUT_IDS].to(device)
                    attention_mask = batch[constants.ATTENTION_MASK].to(device)
                    labels = batch[constants.LABELS].to(device)
                    logger.debug(f"[Epoch {epoch}]: batch sent to {device}")

                    # Forward pass
                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                    logger.debug(f"[Epoch {epoch}]: forward pass complete")

                    # Backward pass and optimization
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    logger.debug(f"[Epoch {epoch}]: backward pass complete")
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item()}")

    def save(self):
        save_model_path = Path("./data/mtg_card_model")
        self.model.save_pretrained(save_model_path)
