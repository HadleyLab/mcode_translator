#!/usr/bin/env python
# coding: utf-8

"""
mCODE Translator CLI Entry Point

This script provides a convenient entry point for the mCODE Translator CLI.
It handles path setup and launches the Typer-based CLI application.
"""

import os
import sys
from pathlib import Path

# Add heysol_api_client to path (required for HeySol imports)
heysol_path = Path(__file__).parent / "heysol_api_client" / "src"
if heysol_path.exists():
    sys.path.insert(0, str(heysol_path))

# Add src to path for mCODE Translator imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def main():
    """Launch the mCODE Translator CLI."""
    from cli import app
    app()

if __name__ == "__main__":
    main()