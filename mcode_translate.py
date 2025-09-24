#!/usr/bin/env python3
"""
Main CLI entry point for the mCODE Translator.

This script provides a Click-based CLI interface.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the Click CLI
from src.cli.click_cli import cli


def main():
    """Main entry point - uses Click CLI."""
    cli()


if __name__ == "__main__":
    main()
