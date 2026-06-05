# White Companion Runtime

White is a Discord companion runtime for a Korean LLM assistant. The runtime packages context, sends it to a local OpenAI-compatible model server, checks generated output, and keeps lightweight memory/state around the conversation.

This public folder contains runtime and test code only. Model weights, private datasets, logs, local databases, and training artifacts are intentionally excluded.

## Components

- `src/discord_lmstudio_bot/main.py`: Discord entrypoint and message flow
- `src/discord_lmstudio_bot/context_packer.py`: turns recent history and memory into the model input context
- `src/discord_lmstudio_bot/llm_client.py`: OpenAI-compatible local model client
- `src/discord_lmstudio_bot/output_guard.py`: catches repetition, malformed text, and unsafe output patterns
- `src/discord_lmstudio_bot/memory_store.py`: lightweight memory storage
- `src/discord_lmstudio_bot/runtime_state.py`: runtime status and state helpers
- `src/discord_lmstudio_bot/startup_lock.py`: prevents duplicate runtime startup
- `tests/`: unit tests for context packing, guards, runtime paths, client behavior, and speech helpers

## Local Setup

```powershell
cd companions\white
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` with local Discord and model-server settings, then run:

```powershell
.\.venv\Scripts\python -m discord_lmstudio_bot
```

## Model-Learning Notes

The current White model work is documented as a portfolio case study:

- [White Mind Map](../../docs/white-mindmap.md)
- [White Case Study](../../docs/white-case-study.md)

The training direction is candidate-based. New adapters are trained, evaluated, and reported, but not automatically promoted into active runtime use.
