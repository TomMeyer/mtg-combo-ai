import json
from logging import getLogger
from threading import Thread
from typing import Any, Generator, Optional

import torch
from peft import PeftModel
from transformers import BatchEncoding, TextIteratorStreamer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from unsloth import FastLanguageModel

from mtg_ai.cards import MTGDatabase

logger = getLogger(__name__)


class MTGCardAI:
    def __init__(
        self,
        model_name: str,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 2048,
        load_in_4bit: bool = False,
    ) -> None:
        model, tokenizer = self._get_model_and_tokenizer(
            model_name=model_name,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
            load_in_4bit=load_in_4bit,
        )
        self.database = MTGDatabase()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: PeftModel = model
        self.model.to(self.device)
        self.tokenizer: PreTrainedTokenizerFast = tokenizer

    @classmethod
    def _get_model_and_tokenizer(
        cls,
        model_name: str,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 2048,
        load_in_4bit: bool = False,
    ):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,  # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length=max_sequence_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
        return model, tokenizer

    @property
    def _system_prompt(self) -> str:
        return """
        You are a Magic the Gathering information bot. 
        You are here to help users with their questions about Magic the Gathering.
        You know information about magic the gathering cards, rules, and other information.
        """  # noqa: E501

    def _build_prompt(
        self, prompt: str, rag_data: Optional[str] = None
    ) -> list[dict[str, Any]]:
        system_prompt = self._system_prompt
        logger.debug(f"Rag data:\n{rag_data}")
        if rag_data:
            additional_prompt = """
            Higher Relevancy Scores means that information is more relevant.
            """
            system_prompt += (
                f".{additional_prompt}\n\relevant information:\n" + rag_data
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    def run(
        self,
        prompt: str,
        rag_data: Optional[str] = None,
        max_new_tokens: int = 500,
        temperature: float = 0.1,
        min_p: float = 0.1,
    ) -> Generator[str, None, None]:
        logger.debug(f"running model with prompt:\n{prompt}")
        messages = self._build_prompt(prompt=prompt, rag_data=rag_data)
        logger.debug(f"structered prompt:\n{json.dumps(messages, indent=2)}")
        tokenized_prompt = self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        if not isinstance(tokenized_prompt, (BatchEncoding, torch.Tensor)):
            raise ValueError(
                f"Tokenized prompt is not a transformers.BatchEncoding or torch.Tensor is {type(tokenized_prompt)} with value {tokenized_prompt}"
            )

        # tokenized_prompt.to("cuda")  # type: ignore

        text_streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )  # type: ignore
        # dict(inputs, streamer=streamer, max_new_tokens=20)
        generation_kwargs = dict(
            inputs=tokenized_prompt,
            streamer=text_streamer,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            min_p=min_p,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        yield from text_streamer
