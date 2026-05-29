from __future__ import annotations

from typing import Protocol


class TextGenerationClient(Protocol):
    async def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...
