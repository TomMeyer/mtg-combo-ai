import json
import logging
from logging import getLogger
from typing import Any, Optional

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

logger = getLogger(__name__)


class MTGCardAI:
    def __init__(
        self,
        model_name: str,
        gguf_file: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        adapter: Optional[str] = None,
    ) -> None:
        if not tokenizer_name:
            tokenizer_name = model_name
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.gguf_file = gguf_file
        self.adapter_name = adapter
        self._model = None
        self._tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def tokenizer(self):
        if not self._tokenizer:
            logger.info(f"loading tokenizer {self.tokenizer_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
            )
            if not self._tokenizer.pad_token:
                self._tokenizer.pad_token = self._tokenizer.unk_token
        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel | PeftModel:
        if not self._model:
            logger.info(f"loading model {self.model_name}")
            config = BitsAndBytesConfig(load_in_8bit=True)

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                gguf_file=self.gguf_file,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                quantization_config=config,
            )
            if self.adapter_name:
                self._model = PeftModel.from_pretrained(
                    self._model, self.adapter_name, low_cpu_mem_usage=True
                )
            # self.model.load_adapter("./results/")
            logger.info(f"moving model to {self.device}")
            self._model = self.model.to(self.device)
        return self._model

    @property
    def _system_prompt(self) -> str:
        return """
        You are a Magic the Gathering information bot. 
        You are here to help users with their questions about Magic the Gathering.
        You know information about magic the gathering cards, rules, and other information.
        """  # noqa: E501

    def _build_prompt(self, prompt: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

    def run(self, prompt: str, max_new_tokens: int = 500):
        logger.debug(f"running model with prompt:\n{prompt}")
        messages = self._build_prompt(prompt)
        logger.debug(f"structered prompt:\n{json.dumps(messages, indent=2)}")
        formatted_prompt = self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        logger.debug(f"templated prompt:\n{formatted_prompt}")

        tokenized_prompt = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            # padding=True,
        )
        tokenized_prompt = tokenized_prompt.to(self.device)
        if logger.level == logging.DEBUG:
            detokenized_prompt = self.tokenizer.decode(tokenized_prompt)
            logger.debug(f"tokenized prompt:\n{detokenized_prompt}")

        attention_mask = tokenized_prompt["attention_mask"]
        response = self.model.generate(
            tokenized_prompt["input_ids"],
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
        )
        return self.tokenizer.decode(response[0], skip_special_tokens=True)
