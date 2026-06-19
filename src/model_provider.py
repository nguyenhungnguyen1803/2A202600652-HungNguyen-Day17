from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic` or `google` -> `gemini`."""
    val = value.lower().strip()
    if val in ["openai", "custom"]:
        return val
    if val in ["gemini", "google"]:
        return "gemini"
    if val in ["anthropic", "claude"]:
        return "anthropic"
    if val in ["ollama"]:
        return "ollama"
    if val in ["openrouter"]:
        return "openrouter"
    return val


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate the real chat model for the selected provider."""
    provider = normalize_provider(config.provider)
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key
        )
    elif provider == "custom":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            openai_api_key=config.api_key,
            openai_api_base=config.base_url
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url
        )
    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            openai_api_key=config.api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")
