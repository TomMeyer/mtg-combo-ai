MTG AI
------
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FTomMeyer%2Fmtg-combo-ai%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)


[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/TomMeyer/mtg-combo-ai/main.svg)](https://results.pre-commit.ci/latest/github/TomMeyer/mtg-combo-ai/main)
![license](https://img.shields.io/badge/license-MIT-blue)

Overview
========

Magic the Gathering LLM with RAG for general magic questions and creating card combos.

Trained using question and answer data generated from cards and popular combinations.

Installation
============

```bash
pixi install --all
```

Usage
=====

To start development shell
```bash
pixi shell -e dev
```

Training
========

```bash
# options can be set in the yaml file or passed as arguments
# --help to list all options

# for multi-gpu training
accelerate launch train.py --config ./llama-3.3-70B-Instruct-fsdp-qlora.yaml

# for single gpu training
accelerate launch train.py --config ./llama-3.3-70B-Instruct-unsloth.yaml
```

```python
from mtg_ai.ai import ModelAndTokenizer, MTGCardAITrainer
model = ModelAndTokenizer.UNSLOTH_LLAMA_3_2_3B_INSTRUCT_Q8
ai_trainer = MTGCardAITrainer(
    model_name=model.value, 
    gguf_file=gguf_file, 
    dataset_name="cards"
)
ai_trainer.train()

# Modify the config file to change the training parameters or resume from checkpoint
```

To use the model for inference
```python
from mtg_ai.ai import MTGAIRunner

ai_model_name: str = "./results"
rag_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

runner = MTGAIRunner(
    ai_model_name=ai_model_name,
    rag_embedding_model_name=rag_embedding_model_name
)

runner.run(
    "What is the converted mana cost of Acquisitions Expert?"
    max_new_tokens=100, 
    top_k=3
)
```

# Inference Server

The easiest way to use this model for model inference is to run it with the 
[huggingface/text-generation-inference](https://huggingface.co/docs/text-generation-inference/en/index) server.

```bash

model=unsloth/Llama-3.3-70B-Instruct
adapter=FringeFields/mtg-ai-llama-3.2-3b-adapter
docker run --gpus all --shm-size 1g -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id $model \
    --lora-adapters $adapter \
    --quantize bitsandbytes-nf4
```

# Acknowledgements

Card data is from [MTGJSON](https://mtgjson.com/)

Combo data is from [CommanderSpellbook](https://json.commanderspellbook.com/variants.json)

# Disclaimer

This is an unofficial project and is not affiliated with Wizards of the Coast.

Permitted under the [Wizards of the Coast Fan Content Policy](https://company.wizards.com/en/legal/fancontentpolicy) 
