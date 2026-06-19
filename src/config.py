from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    # Resolve the repo root or default to the current file parent's parent
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    
    # Load environment variables from .env if present
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # Create directories if they do not exist
    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    # Read configuration knobs with sensible defaults
    provider = os.getenv("LLM_PROVIDER", "openai")
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", None)

    # Check for other providers if specified
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")

    model_config = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        api_key=api_key if api_key else None,
        base_url=base_url
    )

    judge_config = ProviderConfig(
        provider=os.getenv("JUDGE_PROVIDER", provider),
        model_name=os.getenv("JUDGE_MODEL", model_name),
        temperature=0.0,
        api_key=os.getenv("JUDGE_API_KEY", api_key) if os.getenv("JUDGE_API_KEY") else (api_key if api_key else None),
        base_url=os.getenv("JUDGE_BASE_URL", base_url)
    )

    # Sensible compaction configurations
    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "800"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model_config,
        judge_model=judge_config
    )
