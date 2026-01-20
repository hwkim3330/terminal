#!/usr/bin/env python3
"""
TerminaI - LLM Providers

Supports multiple LLM backends:
- Gemini 3 (Flash, Pro, Deep Think)
- Local LLMs (Ollama, vLLM)
- OpenAI-compatible APIs
"""

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncIterator
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM configuration."""
    model: str = "gemini-2.0-flash"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.95
    timeout: float = 60.0
    thinking_mode: bool = False  # For Gemini Pro Deep Think


@dataclass
class LLMResponse:
    """LLM response."""
    content: str = ""
    thinking: Optional[str] = None  # Deep Think reasoning
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""


class LLMProvider(ABC):
    """Abstract LLM provider."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate completion."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate completion with streaming."""
        pass

    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """Chat completion."""
        # Convert messages to prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.insert(0, f"System: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"User: {content}")

        prompt = "\n\n".join(prompt_parts)
        return await self.generate(prompt, **kwargs)


class GeminiProvider(LLMProvider):
    """
    Google Gemini 3 provider.

    Supports:
    - Gemini 2.0 Flash (fast, 1M context)
    - Gemini 2.0 Pro (powerful, 2M context)
    - Gemini 3.0 Flash (faster, 2M context)
    - Gemini 3.0 Pro Deep Think (reasoning mode)
    """

    API_BASE = "https://generativelanguage.googleapis.com/v1beta"

    # Model mappings
    MODELS = {
        "gemini-flash": "gemini-2.0-flash",
        "gemini-pro": "gemini-2.0-pro",
        "gemini-3-flash": "gemini-3.0-flash",
        "gemini-3-pro": "gemini-3.0-pro",
        "gemini-2.0-flash": "gemini-2.0-flash",
        "gemini-2.0-pro": "gemini-2.0-pro",
    }

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            logger.warning("No Gemini API key found. Set GEMINI_API_KEY environment variable.")

        # Resolve model name
        self.model = self.MODELS.get(config.model, config.model)

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate completion using Gemini API."""
        if not self.api_key:
            return "[ERROR: No API key configured]"

        url = f"{self.API_BASE}/models/{self.model}:generateContent"
        params = {"key": self.api_key}

        # Build request
        contents = []

        if system:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System instructions: {system}"}]
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Understood. I will follow these instructions."}]
            })

        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "maxOutputTokens": kwargs.get("max_tokens", self.config.max_tokens),
                "topP": kwargs.get("top_p", self.config.top_p),
            },
        }

        # Enable thinking mode for supported models
        if self.config.thinking_mode and "pro" in self.model.lower():
            body["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": kwargs.get("thinking_budget", 1024)
            }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    params=params,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Gemini API error: {response.status} - {error_text}")
                        return f"[ERROR: {response.status}]"

                    data = await response.json()
                    return self._parse_response(data)

        except asyncio.TimeoutError:
            logger.error("Gemini API timeout")
            return "[ERROR: Timeout]"
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"[ERROR: {e}]"

    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate with streaming."""
        if not self.api_key:
            yield "[ERROR: No API key configured]"
            return

        url = f"{self.API_BASE}/models/{self.model}:streamGenerateContent"
        params = {"key": self.api_key, "alt": "sse"}

        contents = []
        if system:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System: {system}"}]
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Understood."}]
            })

        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "maxOutputTokens": kwargs.get("max_tokens", self.config.max_tokens),
            },
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    params=params,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                text = self._parse_response(data)
                                if text:
                                    yield text
                            except:
                                pass

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"[ERROR: {e}]"

    def _parse_response(self, data: Dict[str, Any]) -> str:
        """Parse Gemini API response."""
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                return ""

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])

            text_parts = []
            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])
                elif "thought" in part:
                    # Deep Think reasoning
                    text_parts.append(f"[THINKING]\n{part['thought']}\n[/THINKING]")

            return "\n".join(text_parts)

        except Exception as e:
            logger.error(f"Parse error: {e}")
            return ""


class OllamaProvider(LLMProvider):
    """
    Ollama local LLM provider.

    For completely local, sovereign execution.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_base = config.api_base or "http://localhost:11434"

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate using Ollama."""
        url = f"{self.api_base}/api/generate"

        body = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            },
        }

        if system:
            body["system"] = system

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    if response.status != 200:
                        return f"[ERROR: {response.status}]"

                    data = await response.json()
                    return data.get("response", "")

        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"[ERROR: {e}]"

    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate with streaming."""
        url = f"{self.api_base}/api/generate"

        body = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
            },
        }

        if system:
            body["system"] = system

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    async for line in response.content:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                        except:
                            pass

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"[ERROR: {e}]"


class OpenAICompatibleProvider(LLMProvider):
    """
    OpenAI-compatible API provider.

    Works with:
    - OpenAI
    - Azure OpenAI
    - vLLM
    - LM Studio
    - Any OpenAI-compatible server
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_base = config.api_base or "https://api.openai.com/v1"
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate using OpenAI API."""
        url = f"{self.api_base}/chat/completions"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    if response.status != 200:
                        return f"[ERROR: {response.status}]"

                    data = await response.json()
                    return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return f"[ERROR: {e}]"

    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate with streaming."""
        url = f"{self.api_base}/chat/completions"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                data = json.loads(line[6:])
                                delta = data["choices"][0]["delta"]
                                if "content" in delta:
                                    yield delta["content"]
                            except:
                                pass

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"[ERROR: {e}]"


def create_llm_provider(provider: str, model: str, **kwargs) -> LLMProvider:
    """
    Factory function to create LLM provider.

    Args:
        provider: Provider name (gemini, ollama, openai, local)
        model: Model name
        **kwargs: Additional configuration

    Returns:
        LLMProvider instance
    """
    config = LLMConfig(model=model, **kwargs)

    if provider.lower() in ["gemini", "google"]:
        return GeminiProvider(config)
    elif provider.lower() == "ollama":
        return OllamaProvider(config)
    elif provider.lower() in ["openai", "local", "vllm"]:
        return OpenAICompatibleProvider(config)
    else:
        # Default to Gemini
        return GeminiProvider(config)
