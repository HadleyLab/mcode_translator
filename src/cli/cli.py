"""
CLI for mCODE Translator.

This module provides a command-line interface that integrates
with heysol_api_client's CLI architecture and provides typed command structure.
"""

import sys
from pathlib import Path
from typing import Optional

import typer

# Import configuration
from ..config.heysol_config import get_config
# Import mCODE command groups
from .data_commands import data
from .patients_commands import patients
from .test_commands import test
from .trials_commands import trials

# Add heysol_api_client to path for imports
heysol_client_path = (
    Path(__file__).parent.parent.parent.parent / "heysol_api_client" / "src"
)
if str(heysol_client_path) not in sys.path:
    sys.path.insert(0, str(heysol_client_path))

# Import heysol CLI for integration
try:
    from cli import app as heysol_app
except ImportError:
    heysol_app = None

# Create the main Typer app
app = typer.Typer()

# Global state for CLI (similar to heysol_api_client pattern)
_global_config = None
_global_api_key = None
_global_base_url = None
_global_source = None
_global_verbose = False


def resolve_user_from_registry(user: str) -> tuple[str | None, str | None]:
    """Resolve user credentials from HeySol registry with debugging."""
    try:
        from heysol.registry_config import RegistryConfig

        # Specify the .env file path explicitly
        heysol_client_path = (
            Path(__file__).parent.parent.parent.parent / "heysol_api_client"
        )
        env_file = heysol_client_path / ".env"

        print(f"🔍 DEBUG: Looking for .env file at: {env_file}")
        print(f"🔍 DEBUG: .env file exists: {env_file.exists()}")

        registry = (
            RegistryConfig(str(env_file)) if env_file.exists() else RegistryConfig()
        )

        # Debug: List available instances
        available_users = registry.get_instance_names()
        print(f"🔍 DEBUG: Available registry users: {available_users}")

        # Try exact match first
        instance = registry.get_instance(user)
        print(f"🔍 DEBUG: Looking up user '{user}' in registry...")

        if instance:
            api_key = instance["api_key"]
            base_url = instance["base_url"]
            print(
                f"✅ DEBUG: Found user '{user}' - API key: {api_key[:10]}..."
                if api_key
                else "❌ DEBUG: No API key found"
            )
            print(f"✅ DEBUG: Base URL: {base_url}")
            return api_key, base_url

        # Try case-insensitive match
        print(f"🔍 DEBUG: User '{user}' not found, trying case-insensitive match...")
        user_lower = user.lower()
        for available_user in available_users:
            if available_user.lower() == user_lower:
                print(f"🔍 DEBUG: Found case-insensitive match: '{available_user}'")
                instance = registry.get_instance(available_user)
                if instance:
                    api_key = instance["api_key"]
                    base_url = instance["base_url"]
                    print(
                        f"✅ DEBUG: Found user '{available_user}' - API key: {api_key[:10]}..."
                        if api_key
                        else "❌ DEBUG: No API key found"
                    )
                    print(f"✅ DEBUG: Base URL: {base_url}")
                    return api_key, base_url

        # No match found
        print(
            f"❌ DEBUG: User '{user}' not found in registry (case-insensitive search also failed)"
        )
        print(f"🔍 DEBUG: Available users: {available_users}")
        print("💡 DEBUG: Try using one of the exact usernames above (case-sensitive)")
        return None, None

    except Exception as e:
        print(f"❌ DEBUG: Registry lookup failed: {e}")
        return None, None


@app.callback()
def cli_callback(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    api_key: Optional[str] = typer.Option(
        None, help="HeySol API key (overrides environment variable)"
    ),
    base_url: Optional[str] = typer.Option(
        None, help="Base URL for API (overrides default)"
    ),
    user: Optional[str] = typer.Option(None, help="User instance name from registry"),
    source: Optional[str] = typer.Option(
        None, help="Source identifier (overrides default)"
    ),
    model: Optional[str] = typer.Option(None, help="Default LLM model to use"),
    prompt: Optional[str] = typer.Option(None, help="Default prompt template to use"),
    workers: Optional[int] = typer.Option(None, help="Number of concurrent workers"),
):
    """
    mCODE Translator CLI - Integrated with HeySol API Client

    A comprehensive CLI for processing clinical trials and patient data into mCODE format,
    with integrated CORE Memory storage and HeySol API client functionality.

    Setup:
      1. Get your API key from: https://core.heysol.ai/settings/api
      2. Set environment variable: export HEYSOL_API_KEY="your-key-here"
      3. Or use --api-key option with each command
      4. Or use --user option with registered instance names

    Examples:
      # Fetch and process clinical trials
      mcode-cli trials fetch --condition "breast cancer" --limit 10

      # Process trials to mCODE format
      mcode-cli trials process trials.ndjson --model gpt-4

      # Summarize mCODE data
      mcode-cli trials summarize mcode_data.ndjson --store-in-memory

      # Fetch patient data
      mcode-cli patients fetch --patient-ids "123,456,789"

      # Process patient data to mCODE
      mcode-cli patients process patients.ndjson

      # Run optimization tests
      mcode-cli trials optimize --models gpt-4,claude-3 --prompts concise,comprehensive

      # Access CORE Memory features
      mcode-cli memory ingest "Clinical trial summary..."
      mcode-cli memory search "breast cancer treatments"
      mcode-cli spaces list
    """
    global _global_config, _global_api_key, _global_base_url, _global_source, _global_verbose

    # Initialize configuration
    _global_config = get_config()
    _global_verbose = verbose

    # Resolve credentials (similar to heysol_api_client pattern)
    resolved_api_key = api_key
    resolved_base_url = base_url

    # If user provided, try to resolve from heysol registry
    if user and not api_key:
        resolved_api_key, resolved_base_url = resolve_user_from_registry(user)

    # Override with provided values or config defaults
    _global_api_key = resolved_api_key or _global_config.get_api_key()
    _global_base_url = resolved_base_url or _global_config.get_base_url()
    _global_source = source or _global_config.get_heysol_config().source

    # Debug: Show final resolved credentials
    print(
        f"🔍 DEBUG: Final API key: {_global_api_key[:10]}..."
        if _global_api_key
        else "❌ DEBUG: No API key resolved"
    )
    print(f"🔍 DEBUG: Final base URL: {_global_base_url}")
    print(f"🔍 DEBUG: Final source: {_global_source}")

    # Store additional settings in global state
    if model:
        _global_config.mcode_default_model = model
    if prompt:
        _global_config.mcode_default_prompt = prompt
    if workers is not None:
        _global_config.mcode_workers = workers


# Add command groups
app.add_typer(trials, name="trials")
app.add_typer(patients, name="patients")
app.add_typer(data, name="data")
app.add_typer(test, name="test")

# Add heysol commands if available
if heysol_app:
    app.add_typer(heysol_app, name="memory")


def get_global_config():
    """Get the global configuration instance."""
    return _global_config


def get_global_api_key() -> Optional[str]:
    """Get the global API key."""
    return _global_api_key


def get_global_base_url() -> Optional[str]:
    """Get the global base URL."""
    return _global_base_url


def get_global_source() -> Optional[str]:
    """Get the global source identifier."""
    return _global_source


def is_verbose() -> bool:
    """Check if verbose mode is enabled."""
    return _global_verbose


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
