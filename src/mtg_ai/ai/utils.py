from enum import Enum
from typing import Optional


class ModelAndTokenizer(Enum):
    LLAMA_3_1_8B_INSTRUCT = (
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        None,
    )
    LLAMA_3_2_3B_INSTRUCT = (
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        None,
    )
    BARTOWSKI_LLAMA_3_1_8B_INSTRUCT_Q4_K_S = (
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf",
    )
    BARTOWSKI_LLAMA_3_1_8B_INSTRUCT_Q4_K_M = (
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    )
    BARTOWSKI_LLAMA_3_1_8B_INSTRUCT_Q4_K_L = (
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf",
    )
    BARTOWSKI_LLAMA_3_2_8B_INSTRUCT_Q4_K_S = (
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "meta-llama/Llama-3.2-3B-Instruct",
        "Llama-3.2-3B-Instruct-Q4_K_S.gguf",
    )
    BARTOWSKI_LLAMA_3_2_8B_INSTRUCT_Q4_K_M = (
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "meta-llama/Llama-3.2-3B-Instruct",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    )
    BARTOWSKI_LLAMA_3_2_8B_INSTRUCT_Q4_K_L = (
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "meta-llama/Llama-3.2-3B-Instruct",
        "Llama-3.2-3B-Instruct-Q4_K_L.gguf",
    )
    UNSLOTH_LLAMA_3_2_3B_INSTRUCT_Q4_K_M = (
        "unsloth/Llama-3.2-3B-Instruct",
        "unsloth/Llama-3.2-3B-Instruct",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    )
    UNSLOTH_LLAMA_3_2_3B_INSTRUCT_Q8 = (
        "unsloth/Llama-3.2-3B-Instruct",
        "unsloth/Llama-3.2-3B-Instruct",
        "Llama-3.2-3B-Instruct-Q8_0.gguf",
    )
    UNSLOTH_LLAMA_3_2_3B_INSTRUCT_BNB_4BIT = (
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "Llama-3.2-3B-Instruct-Q8_0.gguf",
    )

    def __init__(
        self,
        value: str,
        tokenizer: Optional[str] = None,
        gguf_file: Optional[str] = None,
    ):
        self._value_ = value
        self.tokenizer = tokenizer or value
        self.gguf_file = gguf_file

    def __new__(
        cls,
        value: str,
        tokenizer: Optional[str] = None,
        gguf_file: Optional[str] = None,
    ):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.tokenizer: str = tokenizer or value  # type: ignore
        obj.gguf_file: Optional[str] = gguf_file  # type: ignore
        return obj
