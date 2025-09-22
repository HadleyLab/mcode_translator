"""
Common utilities for CLI commands.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import click
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from heysol import HeySolClient, HeySolConfig, HeySolError, ValidationError


def load_config(api_key: Optional[str] = None, base_url: Optional[str] = None, source: Optional[str] = None) -> HeySolConfig:
    """Load configuration from various sources."""
    config = HeySolConfig.from_env()

    # Override with command line arguments
    if api_key:
        config.api_key = api_key
    if base_url:
        config.base_url = base_url
    if source:
        config.source = source

    # Override with environment variables if not set
    if not config.api_key:
        config.api_key = os.getenv("HEYSOL_API_KEY") or os.getenv("COREAI_API_KEY")

    return config


def create_client(api_key: Optional[str] = None, base_url: Optional[str] = None, skip_mcp: bool = False) -> HeySolClient:
    """Create and initialize HeySol client."""
    config = load_config(api_key, base_url)

    if not config.api_key:
        raise click.ClickException("API key is required. Set HEYSOL_API_KEY or COREAI_API_KEY environment variable, or use --api-key option.")

    try:
        client = HeySolClient(
            api_key=config.api_key,
            base_url=config.base_url,
            skip_mcp_init=skip_mcp
        )
        return client
    except Exception as e:
        raise click.ClickException(f"Error initializing client: {e}")


def format_json_output(data: Any, pretty: bool = False) -> str:
    """Format data as JSON output."""
    if pretty:
        return json.dumps(data, indent=2, default=str)
    return json.dumps(data, default=str)