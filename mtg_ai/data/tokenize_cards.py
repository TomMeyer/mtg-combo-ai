import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from mtg_ai.data.cards import MTGCards
from mtg_ai import constants


class MTGCardDataset(Dataset):
    
    def __init__(self):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("Intel/neural-chat-7b-v3")
        self.tokenized_data = tokenize(MTGCards, self.tokenizer)

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        tokenized_text = self.tokenized_data[idx]
        return tokenized_text


def tokenize(df: pd.DataFrame, tokenizer):
    tokenizer.pad_token = constants.PAD_TOKEN
    combined_str_data: dict[str, str] = {}
    for index, row in tqdm(df.iterrows(), desc="turning rows into strings"):
        combined_str: str = "| "
        for k, v in row.items():
            if isinstance(v, list):
                v = "".join(v)
            combined_str += f"{k}: {v} | "
        combined_str = combined_str.strip()
        combined_str_data[index] = combined_str
    result = []
    
    #340
    for k, v in tqdm(combined_str_data.items(), desc="tokenizing rows"):
        encoding = tokenizer(
            text=v,
            return_tensors="pt",
            max_length=350,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        # For a language modeling task, inputs and labels are typically the same
        tokenized_data = {
            constants.INPUT_IDS: encoding[constants.INPUT_IDS].flatten(),
            constants.ATTENTION_MASK: encoding[constants.ATTENTION_MASK].flatten(),
            constants.LABELS: encoding[constants.INPUT_IDS].flatten(),
        }
        result.append(tokenized_data)
    return result
