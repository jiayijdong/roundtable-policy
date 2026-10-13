"""Backend for Claude models via the official Anthropic SDK."""

from __future__ import annotations

from typing import Optional

from .base import LLM

# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


class AnthropicLLM(LLM):
    def __init__(
        self,
        model: str = "claude-opus-5-5",
        name: Optional[str] = None,
        max_tokens: int = 64000,
        effort: Optional[str] = None,
        server_fallback: bool = True,
    ):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'roundtable-policy[anthropic]'") from e
        self.model = model
        self.name = name or model
        self.max_tokens = max_tokens
        self.effort = effort  # "low" | "medium" | "high" | "xhigh" | "max"
        self.server_fallback = server_fallback and model in _FALLBACK_MODELS
        # Credentials resolve from ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login`.
        self.client = anthropic.Anthropic()

    def complete(self, system: str, user: str, role: Optional[str] = None) -> str:
        kwargs = {}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.server_fallback:
            kwargs["extra_headers"] = {"anthropic-beta": "server-side-fallback-2026-07-01"}
            kwargs["extra_body"] = {"fallbacks": "default"}
        # Stream so long generations don't hit HTTP timeouts.
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **kwargs,
        ) as stream:
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            raise RuntimeError(f"{self.name} declined the request (stop_reason=refusal)")
        return "".join(b.text for b in message.content if b.type == "text")
