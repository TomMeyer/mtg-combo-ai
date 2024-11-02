MTG AI
------
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FTomMeyer%2Fmtg-combo-ai%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)


[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/TomMeyer/mtg-combo-ai/main.svg)](https://results.pre-commit.ci/latest/github/TomMeyer/mtg-combo-ai/main)

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

To train the model
```python
from mtg_ai.ai import ModelAndTokenizer, MTGCardAITrainer
model = ModelAndTokenizer.UNSLOTH_LLAMA_3_2_3B_INSTRUCT_Q8
ai_trainer = MTGCardAITrainer(
    model_name=model.value, 
    gguf_file=gguf_file, 
    dataset_name="cards"
)
ai_trainer.train(resume_from_checkpoint=False)
```

**note**: Model is not quite ready to be uploaded to huggingface, so you will need to train it yourself.

To use the model for inference
```python
from mtg_ai.ai import MTGAIRunner

ai_model_name: str = "./results"
rag_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

runner = MTGAIRunner(
    ai_model_name=ai_model_name, rag_embedding_model_name=rag_embedding_model_name
)

runner.run(
    "What is the converted mana cost of Acquisitions Expert?"
    max_new_tokens=100, 
    filters=None, top_k=3
)

```

# Acknowledgements

Card data is from [MTGJSON](https://mtgjson.com/)

Combo data is from [CommanderSpellbook](https://json.commanderspellbook.com/variants.json)

# Disclaimer

This is an unofficial project and is not affiliated with Wizards of the Coast.

Permitted under the [Wizards of the Coast Fan Content Policy](https://company.wizards.com/en/legal/fancontentpolicy) 