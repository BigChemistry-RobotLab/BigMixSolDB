from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import requests


class LLMError(RuntimeError):
    """Raised when an upstream model call fails."""


@dataclass(slots=True)
class ProviderConfig:
    provider: str
    model: str
    api_key: str
    api_base: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: int = 600


class LLMClient:
    def __init__(self, config: ProviderConfig):
        provider = config.provider.lower()
        if provider not in {"openai", "gemini"}:
            raise ValueError(f"Unsupported provider: {config.provider}")
        self.config = ProviderConfig(
            provider=provider,
            model=config.model,
            api_key=config.api_key,
            api_base=config.api_base,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )

    def generate_text(self, system_prompt: str, user_text: str) -> str:
        if self.config.provider == "openai":
            return self._generate_openai(system_prompt=system_prompt, user_text=user_text)
        return self._generate_gemini(system_prompt=system_prompt, user_text=user_text)

    def generate_from_image(
        self,
        system_prompt: str,
        user_text: str,
        image_bytes: bytes,
        mime_type: str,
    ) -> str:
        if self.config.provider == "openai":
            return self._generate_openai(
                system_prompt=system_prompt,
                user_text=user_text,
                image_bytes=image_bytes,
                mime_type=mime_type,
            )
        return self._generate_gemini(
            system_prompt=system_prompt,
            user_text=user_text,
            image_bytes=image_bytes,
            mime_type=mime_type,
        )

    def _generate_openai(
        self,
        system_prompt: str,
        user_text: str,
        image_bytes: bytes | None = None,
        mime_type: str | None = None,
    ) -> str:
        api_base = (self.config.api_base or "https://api.openai.com/v1").rstrip("/")
        url = f"{api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        if image_bytes is None:
            user_content: str | list[dict[str, Any]] = user_text
        else:
            if not mime_type:
                raise ValueError("mime_type is required when sending an image")
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            user_content = [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                },
            ]

        messages.append({"role": "user", "content": user_content})

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens

        response = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout)
        if response.status_code >= 400:
            raise LLMError(
                f"OpenAI-compatible request failed with status {response.status_code}: {response.text}"
            )

        data = response.json()
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected OpenAI-compatible response: {data}") from exc

        text = self._content_to_text(content).strip()
        if text:
            return text

        finish_reason = choice.get("finish_reason")
        if message.get("reasoning_content"):
            raise LLMError(
                "OpenAI-compatible response contained no assistant content. "
                f"finish_reason={finish_reason!r}. "
                "This backend returned reasoning content without a final answer; "
                "increase max_tokens or use a non-reasoning chat model."
            )
        raise LLMError(f"OpenAI-compatible response contained no assistant content: {data}")

    def _generate_gemini(
        self,
        system_prompt: str,
        user_text: str,
        image_bytes: bytes | None = None,
        mime_type: str | None = None,
    ) -> str:
        api_base = (self.config.api_base or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        url = f"{api_base}/models/{self.config.model}:generateContent?key={self.config.api_key}"

        parts: list[dict[str, Any]] = []
        if user_text.strip():
            parts.append({"text": user_text})
        if image_bytes is not None:
            if not mime_type:
                raise ValueError("mime_type is required when sending an image")
            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                    }
                }
            )

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": self.config.temperature},
        }
        if system_prompt.strip():
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
        if self.config.max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = self.config.max_tokens

        response = requests.post(url, json=payload, timeout=self.config.timeout)
        if response.status_code >= 400:
            raise LLMError(f"Gemini request failed with status {response.status_code}: {response.text}")

        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMError(f"Gemini returned no candidates: {data}")

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "\n".join(part.get("text", "") for part in parts if part.get("text"))
        if not text.strip():
            raise LLMError(f"Gemini returned no text content: {data}")
        return text

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part)
        return str(content)
