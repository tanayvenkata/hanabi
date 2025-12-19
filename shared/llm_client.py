"""
LLM Client

Unified interface for calling LLMs from different providers:
- Local endpoints (LM Studio, vLLM, etc.)
- OpenAI API
- (Future: Anthropic, etc.)

Loads configuration from environment variables (.env file).
"""

import os
import requests
from typing import Optional, Dict, Any, List

# Try to import python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    # python-dotenv not installed, skip loading
    pass

# Try to import OpenAI client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMClient:
    """
    Unified client for calling LLMs.

    Supports:
    - Local endpoints (via requests)
    - OpenAI API (via openai package)

    Configuration priority:
    1. Config dict passed to __init__
    2. Environment variables (.env file)
    3. Hardcoded defaults
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLM client.

        Args:
            config: Optional configuration dict with:
                - llm_provider: "local" or "openai" (default: from env or "local")
                - llm_url: For local providers (default: LM Studio)
                - model: Model name (default: depends on provider)
                - temperature: Sampling temperature (default: 0.3)
                - openai_api_key: OpenAI API key (default: from env)
                - openai_org_id: OpenAI org ID (optional, from env)
        """
        config = config or {}

        # Provider selection
        self.provider = config.get(
            "llm_provider",
            os.getenv("LLM_PROVIDER", "local")
        ).lower()

        # Local endpoint configuration
        self.llm_url = config.get(
            "llm_url",
            os.getenv("LLM_URL", "http://127.0.0.1:1234/v1/chat/completions")
        )

        # Model selection (depends on provider)
        default_model = "gpt-4o-mini" if self.provider == "openai" else "nvidia/nemotron-3-nano"
        self.model = config.get(
            "model",
            os.getenv("LLM_MODEL", default_model)
        )

        # Temperature
        self.temperature = config.get(
            "temperature",
            float(os.getenv("LLM_TEMPERATURE", "0.3"))
        )

        # OpenAI-specific configuration
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI provider selected but 'openai' package not installed. "
                    "Install with: pip install openai"
                )

            api_key = config.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
            if not api_key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY in .env file or config."
                )

            org_id = config.get("openai_org_id", os.getenv("OPENAI_ORG_ID"))

            self.openai_client = OpenAI(
                api_key=api_key,
                organization=org_id if org_id else None
            )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Call LLM with chat completion format.

        Args:
            messages: List of message dicts with "role" and "content"
                Example: [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"}
                ]
            temperature: Optional temperature override
            model: Optional model override

        Returns:
            LLM response text
        """
        temperature = temperature if temperature is not None else self.temperature
        model = model if model is not None else self.model

        if self.provider == "openai":
            return self._call_openai(messages, temperature, model)
        else:
            return self._call_local(messages, temperature, model)

    def _call_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        model: str
    ) -> str:
        """Call OpenAI API."""
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"ERROR: OpenAI API request failed: {e}"

    def _call_local(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        model: str
    ) -> str:
        """Call local LLM endpoint (LM Studio, vLLM, etc.)."""
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"ERROR: Local LLM request failed: {e}"


def create_llm_client(config: Optional[Dict[str, Any]] = None) -> LLMClient:
    """
    Factory function to create an LLM client.

    Args:
        config: Optional configuration dict

    Returns:
        Configured LLMClient instance
    """
    return LLMClient(config)
