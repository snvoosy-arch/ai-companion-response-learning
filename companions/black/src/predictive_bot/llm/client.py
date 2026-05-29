from __future__ import annotations

import httpx


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str | None,
        base_url: str,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.api_key = api_key or ""
        self.model = model or ""
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.4,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"].strip()
        if not content:
            raise RuntimeError("LLM returned an empty response.")
        return content
