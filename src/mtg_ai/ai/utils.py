import logging
from enum import Enum
from typing import Optional

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class ProfilerCallback(TrainerCallback):
    def __init__(self, log_dir):
        self.profiler = None
        self.log_dir = log_dir

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("Starting torch Profiler...")
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
            record_shapes=True,
            with_stack=True,
        )
        self.profiler.__enter__()

    def on_train_end(self, args, state, control, **kwargs):
        logger.info("Stopping torch Profiler...")
        if self.profiler:
            self.profiler.__exit__(None, None, None)

    def on_step_end(self, args, state, control, **kwargs):
        if self.profiler:
            self.profiler.step()


# def monitor_gpu_memory():
#     device_count = torch.cuda.device_count()
#     data = {i: {} for i in range(device_count)}

#     def build_table():
#         table = Table(title="GPU Memory Usage")
#         table.add_column("Device ID", justify="left")
#         table.add_column("Allocated (GB)", justify="right")
#         table.add_column("Reserved (GB)", justify="right")
#         for device_id in range(device_count):
#             allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
#             reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
#             torch.cuda.memory_
#             if allocated > 0 or reserved > 0:
#                 data[device_id]["allocated"] = allocated
#                 data[device_id]["reserved"] = reserved

#         for device_id, device_data in data.items():
#             table.add_row(
#                 str(device_id),
#                 f"{device_data.get('allocated', 0):.2f}",
#                 f"{device_data.get('reserved', 0):.2f}",
#             )


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
    UNSLOTH_LLAMA_3_1_STORM_8B = (
        "unsloth/Llama-3.1-Storm-8B",
        "unsloth/Llama-3.1-Storm-8B",
        None,
    )
    UNSLOTH_LLAMA_3_1_70B_INSTRUCT = (
        "unsloth/Meta-Llama-3.1-70B-Instruct",
        "unsloth/Meta-Llama-3.1-70B-Instruct",
        None,
    )
    UNSLOTH_LLAMA_3_1_NEMOTRON_70B_INSTRUCT = (
        "unsloth/Llama-3.1-Nemotron-70B-Instruct",
        "unsloth/Llama-3.1-Nemotron-70B-Instruct",
        None,
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
