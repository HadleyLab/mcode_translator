"""
HeySol API Client Configuration Integration for mCODE Translator.

This module provides a typed configuration system that integrates
heysol_api_client's HeySolConfig with mCODE Translator's specific needs.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from heysol.config import HeySolConfig as BaseHeySolConfig
except ImportError:
    # Fallback for when heysol_api_client is not available
    BaseHeySolConfig = None

# Add heysol_api_client to path for imports
heysol_client_path = Path(__file__).parent.parent.parent.parent / "heysol_api_client" / "src"
if str(heysol_client_path) not in sys.path:
    sys.path.insert(0, str(heysol_client_path))

# Import existing mCODE config for integration
try:
    from utils.config import Config as McodeConfig
except ImportError:
    McodeConfig = None


@dataclass
class McodeHeySolConfig:
    """
    Extended HeySol configuration for mCODE Translator.

    Integrates heysol_api_client's HeySolConfig with mCODE-specific settings
    and supports both API key and user registry authentication.
    """

    # Core HeySol configuration (when available)
    heysol: Optional[BaseHeySolConfig] = field(default=None)

    # Authentication settings
    api_key: Optional[str] = field(default=None)
    user: Optional[str] = field(default=None)

    # mCODE-specific settings
    mcode_cache_enabled: bool = field(
        default_factory=lambda: os.getenv("MCODE_CACHE_ENABLED", "true").lower() == "true"
    )
    mcode_cache_directory: str = field(
        default_factory=lambda: os.getenv("MCODE_CACHE_DIRECTORY", "./data/cache")
    )
    mcode_batch_size: int = field(default_factory=lambda: int(os.getenv("MCODE_BATCH_SIZE", "10")))
    mcode_workers: int = field(default_factory=lambda: int(os.getenv("MCODE_WORKERS", "0")))

    # LLM settings
    mcode_default_model: str = field(
        default_factory=lambda: os.getenv("MCODE_DEFAULT_MODEL", "gpt-4")
    )
    mcode_default_prompt: str = field(
        default_factory=lambda: os.getenv(
            "MCODE_DEFAULT_PROMPT", "direct_mcode_evidence_based_concise"
        )
    )

    # API settings
    mcode_clinical_trials_base_url: str = field(
        default_factory=lambda: os.getenv(
            "MCODE_CLINICAL_TRIALS_BASE_URL", "https://clinicaltrials.gov/api/v2"
        )
    )

    # Storage settings
    mcode_storage_format: str = field(
        default_factory=lambda: os.getenv("MCODE_STORAGE_FORMAT", "ndjson")
    )
    mcode_compression_enabled: bool = field(
        default_factory=lambda: os.getenv("MCODE_COMPRESSION_ENABLED", "false").lower() == "true"
    )

    # Validation settings
    mcode_strict_mode: bool = field(
        default_factory=lambda: os.getenv("MCODE_STRICT_MODE", "true").lower() == "true"
    )
    mcode_require_api_keys: bool = field(
        default_factory=lambda: os.getenv("MCODE_REQUIRE_API_KEYS", "true").lower() == "true"
    )

    @classmethod
    def from_env(cls) -> "McodeHeySolConfig":
        """Create configuration from environment variables and command line args."""
        # Get authentication from environment or use provided values
        api_key = os.getenv("HEYSOL_API_KEY")
        base_url = os.getenv("HEYSOL_BASE_URL", "https://core.heysol.ai/api/v1")

        # Initialize HeySol config if available
        heysol_config = None
        if BaseHeySolConfig and api_key:
            try:
                heysol_config = BaseHeySolConfig(api_key=api_key, base_url=base_url)
            except Exception:
                # Fallback if HeySol config creation fails
                pass

        return cls(
            heysol=heysol_config, api_key=api_key, user=None  # User would be set from CLI args
        )

    @classmethod
    def with_authentication(
        cls,
        api_key: Optional[str] = None,
        user: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> "McodeHeySolConfig":
        """Create configuration with explicit authentication."""
        # Validate authentication
        if not api_key and not user:
            raise ValueError("Either api_key or user must be provided")

        if api_key and user:
            raise ValueError("Only one authentication method allowed (api_key or user)")

        # Resolve authentication
        resolved_api_key = api_key
        resolved_base_url = base_url or "https://core.heysol.ai/api/v1"

        # Initialize HeySol config if available
        heysol_config = None
        if BaseHeySolConfig and resolved_api_key:
            try:
                heysol_config = BaseHeySolConfig(
                    api_key=resolved_api_key, base_url=resolved_base_url
                )
            except Exception:
                # Fallback if HeySol config creation fails
                pass

        return cls(heysol=heysol_config, api_key=resolved_api_key, user=user)

    @classmethod
    def with_mcode_integration(
        cls, mcode_config: Optional["McodeConfig"] = None
    ) -> "McodeHeySolConfig":
        """Create configuration with mCODE-specific settings integration."""
        config = cls.from_env()

        # Integrate with existing mCODE config if available
        if mcode_config:
            try:
                # Override with mCODE-specific settings
                config.mcode_cache_enabled = mcode_config.is_cache_enabled()
                config.mcode_cache_directory = mcode_config.get_api_cache_directory()
                config.mcode_clinical_trials_base_url = mcode_config.get_clinical_trials_base_url()

                # Get LLM settings
                try:
                    config.mcode_default_model = next(
                        iter(mcode_config.get_all_llm_configs().keys())
                    )
                except (StopIteration, AttributeError):
                    pass  # Keep default

            except Exception:
                # If integration fails, continue with defaults
                pass

        return config

    def get_heysol_config(self) -> BaseHeySolConfig:
        """Get the underlying HeySol configuration."""
        return self.heysol

    def get_api_key(self, model_name: Optional[str] = None) -> Optional[str]:
        """Get HeySol API key."""
        if self.api_key:
            return self.api_key
        if self.heysol and hasattr(self.heysol, "api_key"):
            api_key = self.heysol.api_key
            return api_key if isinstance(api_key, str) else None
        return None

    def resolve_user_authentication(self) -> Optional[str]:
        """Resolve user registry authentication to API key.

        This method would implement the actual registry resolution logic.
        For now, returns None to indicate the feature structure is ready.
        """
        if not self.user:
            return None

        # TODO: Implement actual registry resolution
        # This would involve:
        # 1. Checking local registry files
        # 2. Querying HeySol registry API
        # 3. Caching resolved credentials securely

        return None

    def get_base_url_old(self) -> str:
        """Get HeySol base URL."""
        if self.heysol:
            base_url = self.heysol.base_url
            return base_url if isinstance(base_url, str) else ""
        return "https://core.heysol.ai/api/v1"  # Default fallback

    def get_timeout(self, model_name: Optional[str] = None) -> int:
        """Get HeySol timeout."""
        if self.heysol and hasattr(self.heysol, "timeout"):
            timeout = self.heysol.timeout
            return timeout if isinstance(timeout, int) else 60
        return 60  # Default fallback

    def get_temperature(self, model_name: Optional[str] = None) -> float:
        """Get temperature setting."""
        return 0.7  # Default temperature

    def get_max_tokens(self, model_name: Optional[str] = None) -> int:
        """Get max tokens setting."""
        return 1000  # Default max tokens

    def get_base_url(self, model_name: Optional[str] = None) -> str:
        """Get base URL."""
        if self.heysol:
            base_url = self.heysol.base_url
            return base_url if isinstance(base_url, str) else ""
        return "https://core.heysol.ai/api/v1"  # Default fallback

    def is_cache_enabled(self) -> bool:
        """Check if mCODE caching is enabled."""
        return self.mcode_cache_enabled

    def get_cache_directory(self) -> str:
        """Get mCODE cache directory."""
        return self.mcode_cache_directory

    def get_batch_size(self) -> int:
        """Get mCODE batch size."""
        return self.mcode_batch_size

    def get_workers(self) -> int:
        """Get mCODE worker count."""
        return self.mcode_workers

    def get_default_model(self) -> str:
        """Get default LLM model."""
        return self.mcode_default_model

    def get_default_prompt(self) -> str:
        """Get default prompt template."""
        return self.mcode_default_prompt

    def get_clinical_trials_base_url(self) -> str:
        """Get clinical trials API base URL."""
        return self.mcode_clinical_trials_base_url

    def is_strict_mode(self) -> bool:
        """Check if strict validation mode is enabled."""
        return self.mcode_strict_mode

    def require_api_keys(self) -> bool:
        """Check if API keys are required."""
        return self.mcode_require_api_keys


# Global configuration instance
_config_instance: Optional[McodeHeySolConfig] = None


def get_config() -> McodeHeySolConfig:
    """Get the global mCODE HeySol configuration."""
    global _config_instance
    if _config_instance is None:
        _config_instance = McodeHeySolConfig.with_mcode_integration(
            McodeConfig() if McodeConfig else None
        )
    return _config_instance


def reset_config() -> None:
    """Reset the global configuration (mainly for testing)."""
    global _config_instance
    _config_instance = None
