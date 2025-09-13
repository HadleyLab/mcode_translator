"""
Token tracking utilities for standardized token usage reporting across all LLM calls
"""

import json
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any, Dict, Optional


@dataclass
class TokenUsage:
    """Standardized token usage data structure"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_name: str = ""
    provider_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenUsage":
        """Create TokenUsage from dictionary"""
        return cls(**data)

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage instances together"""
        if not isinstance(other, TokenUsage):
            return self

        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            model_name=self.model_name or other.model_name,
            provider_name=self.provider_name or other.provider_name,
        )


class TokenTracker:
    """
    Thread-safe token usage tracker for aggregating token usage across all LLM calls
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TokenTracker, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._usage = TokenUsage()
            self._component_usage = {}
            self._initialized = True

    def add_usage(self, usage: TokenUsage, component: str = "default") -> None:
        """
        Add token usage to the tracker

        Args:
            usage: TokenUsage instance to add
            component: Component name for tracking (e.g., "nlp_extraction", "mcode_mapping")
        """
        with self._lock:
            # Add to total usage
            self._usage = self._usage + usage

            # Track by component
            if component not in self._component_usage:
                self._component_usage[component] = TokenUsage()
            self._component_usage[component] = self._component_usage[component] + usage

    def get_total_usage(self) -> TokenUsage:
        """Get total token usage across all components"""
        with self._lock:
            return TokenUsage(
                prompt_tokens=self._usage.prompt_tokens,
                completion_tokens=self._usage.completion_tokens,
                total_tokens=self._usage.total_tokens,
                model_name=self._usage.model_name,
                provider_name=self._usage.provider_name,
            )

    def get_component_usage(self, component: str) -> Optional[TokenUsage]:
        """Get token usage for a specific component"""
        with self._lock:
            return self._component_usage.get(component)

    def get_all_component_usage(self) -> Dict[str, TokenUsage]:
        """Get token usage for all components"""
        with self._lock:
            return {
                k: TokenUsage(**v.to_dict()) for k, v in self._component_usage.items()
            }

    def reset(self) -> None:
        """Reset all token usage tracking"""
        with self._lock:
            self._usage = TokenUsage()
            self._component_usage = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert tracker state to dictionary for serialization"""
        with self._lock:
            return {
                "total": self._usage.to_dict(),
                "components": {
                    k: v.to_dict() for k, v in self._component_usage.items()
                },
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenTracker":
        """Create TokenTracker from dictionary"""
        tracker = cls()
        tracker._usage = TokenUsage.from_dict(data.get("total", {}))
        tracker._component_usage = {
            k: TokenUsage.from_dict(v) for k, v in data.get("components", {}).items()
        }
        return tracker


def extract_token_usage_from_response(
    response: Any, model_name: str = "", provider_name: str = ""
) -> TokenUsage:
    """
    Extract token usage information from various LLM API response formats

    Args:
        response: LLM API response object
        model_name: Name of the model used
        provider_name: Name of the provider used

    Returns:
        TokenUsage instance with extracted token information
    """
    usage = TokenUsage(model_name=model_name, provider_name=provider_name)

    # Handle OpenAI-style responses
    if hasattr(response, "usage"):
        api_usage = response.usage
        if hasattr(api_usage, "prompt_tokens"):
            usage.prompt_tokens = getattr(api_usage, "prompt_tokens", 0)
        if hasattr(api_usage, "completion_tokens"):
            usage.completion_tokens = getattr(api_usage, "completion_tokens", 0)
        if hasattr(api_usage, "total_tokens"):
            usage.total_tokens = getattr(api_usage, "total_tokens", 0)

    # Handle DeepSeek-style responses (if different)
    elif isinstance(response, dict) and "usage" in response:
        api_usage = response["usage"]
        if isinstance(api_usage, dict):
            usage.prompt_tokens = api_usage.get("prompt_tokens", 0)
            usage.completion_tokens = api_usage.get("completion_tokens", 0)
            usage.total_tokens = api_usage.get("total_tokens", 0)

    # Try to infer total tokens if not provided
    if (
        usage.total_tokens == 0
        and usage.prompt_tokens > 0
        and usage.completion_tokens > 0
    ):
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

    return usage


# Global token tracker instance
global_token_tracker = TokenTracker()
