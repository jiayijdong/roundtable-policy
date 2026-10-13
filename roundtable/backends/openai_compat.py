"""Backend for any OpenAI-compatible chat-completions endpoint.

Works with OpenAI, and with providers / servers that expose the same API
(OpenRouter, Together, DeepSeek, xAI, vLLM, SGLang, Ollama, ...) via ``base_url``.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .base import LLM


class OpenAICompatLLM(LLM):
    def __init__(
        self,
        model: str,
        name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        **request_kwargs: Any,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'roundtable-policy[openai]'") from e
        self.model = model
        self.name = name or model
        self.request_kwargs = request_kwargs  # e.g. temperature, max_tokens
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env))

    def complete(self, system: str, user: str, role: Optional[str] = None) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self.request_kwargs,
        )
        return resp.choices[0].message.content or ""
