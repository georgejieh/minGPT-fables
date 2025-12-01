# minGPT-Fables (Work in Progress)

minGPT-Fables is an educational project built to understand GPT-style transformer models by re-implementing Karpathy’s minGPT from scratch and adapting it for the generation of Aesop-style fables. The repository is currently under active development, and most modules are placeholders or partially implemented. The original minGPT codebase is included only as a reference during the rewrite process.

---

## Project Overview

The purpose of this project is to reconstruct the core ideas of minGPT in a modern, modular structure while customizing the implementation for a specialized text-generation task. The long-term objective is to build a small, interpretable GPT-style model trained specifically on Aesop's Fables, then extend the system with:

- A modular data processing and tokenization pipeline
- A from-scratch transformer architecture
- A reinforcement learning (RL) fine-tuning loop
- A coaching agent that evaluates narrative quality using an external LLM
- A lightweight Gradio interface for iterative generation and evaluation

At this stage, the repository represents an early scaffold for these components. The project will expand gradually as each module is rewritten and tested.

---

## Educational Purpose

This repository is intended for learning and exploration rather than for production deployment. All components are being rewritten with clarity, readability, and extensibility in mind. The original minGPT repository serves solely as conceptual guidance. No direct copying of source code is used; instead, each file is rethought and rebuilt to match the goals of this project.

---

## Repository Structure (Early State)

```
fables/
├── configs/          # YAML configuration templates
├── data/
│   ├── raw/          # Raw Aesop's Fables text
│   └── cleaned/      # Manually cleaned and script-standardized text
├── scripts/          # Data processing and future training scripts
├── src/
│   └── mingpt_fables/
│       ├── models/
│       ├── tokenization/
│       ├── training/
│       ├── evaluation/
│       ├── coaching/
│       ├── rl/
│       ├── ui/
│       └── utils/
└── tests/            # Unit test scaffolding

original_files/       # Unmodified fork of minGPT for reference only
```

Over time, the finalized implementation in `fables/` will be promoted to the project root, and `original_files/` will be removed.

---

## Data Pipeline (Partially Implemented)

The repository already includes the full dataset to be used for training small GPT models:

1. **Raw Dataset**  
   Located in `data/raw/aesop_fables_raw.txt`.  
   This file contains the unprocessed public-domain text.

2. **Manually Cleaned Dataset**  
   Located in `data/cleaned/aesop_fables_clean.txt`.  
   This version removes obvious formatting inconsistencies.

3. **Automated Standardization Script**  
   Located in `scripts/clean_aesop_fables.py`.  
   The script enforces consistent structural formatting by detecting title lines, removing commentary or indented lines, normalizing spacing, and preparing the text for ingestion by the tokenizer.

This forms the foundation of the training data pipeline, even though model training and evaluation have not yet been implemented.

---

## Planned Features (Not Yet Implemented)

The final vision for this project includes:

### Core Model
- A clean, modular GPT implementation
- Configurable “nano” and “small” variants for experimentation

### Tokenizer
- A rewritten BPE tokenizer inspired by minGPT
- Optional extensions for fable-specific tokens

### Training Pipeline
- Fully modular trainer
- Dataset and dataloader modules
- Logging, evaluation hooks, and checkpointing

### Reinforcement Learning
- Reward models for narrative structure, similarity, and moral clarity
- A coaching agent using an external LLM for qualitative evaluation
- Fine-tuning loops for improving story quality

### User Interface
- A simple Gradio application for generating and reviewing fables
- Integration of RL feedback and scoring into the UI

### Testing
- Unit tests for tokenizer, model components, trainer, RL loop, and evaluation metrics

These components will be developed incrementally as the project evolves.

---

## License

This project follows the licensing structure of the original minGPT repository. See the `LICENSE` file for details.

---

## Current Status

This repository is in the early stages of development. While the directory structure, data cleaning script, and initial module scaffolding are in place, the model code, RL components, training loop, and user interface have not yet been implemented.

The repository will continue to develop as each subsystem is rewritten and validated.

