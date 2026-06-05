# White Companion Model Case Study

White is a Korean LLM companion project focused on making a model learn a stable conversational style through SFT, DPO, and evaluation loops.

This public repository is a portfolio version. It keeps code and design notes that can be reviewed safely, while excluding model weights, raw training data, private logs, tokens, local databases, and generated artifacts.

## Start Here

- [Mind Map](docs/white-mindmap.md)
- [Case Study](docs/white-case-study.md)
- [White Runtime README](companions/white/README.md)

## Project Focus

- Runtime-aligned message-format SFT instead of simple prompt/answer pairs
- Calm Korean casual speech for a companion-style assistant
- Failure-driven evaluation for copying, repetition, wrapper leaks, broken Korean, and boundary mistakes
- Candidate-only training and reporting, with no automatic active promotion
- Low-load local experimentation on a memory-constrained Windows and WSL machine

## Current Snapshot

| Area | Status |
| --- | --- |
| Best baseline | v106 remains the strongest evaluated White candidate |
| Latest patch | v109 slightly improved boundary behavior over v106, but was not enough to promote |
| Main remaining weakness | Assistant-care and weather-boundary cases still need stronger preference training |
| Next direction | Accumulate real failed outputs as chosen/rejected DPO pairs and keep regression eval fixed |

## Public Scope

Included:

- White runtime code
- White test code
- Public-safe scripts and examples
- Portfolio documents explaining the training and evaluation decisions

Excluded:

- Model weights and adapters
- Raw SFT, DPO, and RL data
- Private Discord logs or user data
- Local `.env` files, databases, reports, caches, and generated training outputs

## Local Runtime Overview

The public White runtime code is under `companions/white`.

```powershell
cd companions\white
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
.\.venv\Scripts\python -m discord_lmstudio_bot
```

The model-learning work described in the case study was run with private training artifacts that are not included in this repository.
