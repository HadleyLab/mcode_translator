#!/usr/bin/env python3
"""Production setup script for mCODE Translator.

This script prepares the environment for production deployment.
"""

import os
import sys
from pathlib import Path


def setup_production_environment():
    """Set up the production environment."""
    print("🚀 Setting up mCODE Translator v2.0 for production...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    # Check required files
    required_files = [
        "mcode_translator.py",
        "requirements.txt",
        "src/pipeline/__init__.py",
        "prompts/prompts_config.json",
        "models/models_config.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        sys.exit(1)
    
    # Check data directory
    data_dir = Path("data")
    if not data_dir.exists():
        print("📁 Creating data directory...")
        data_dir.mkdir()
    
    # Check for .env file
    if not Path(".env").exists():
        if Path(".env.example").exists():
            print("📝 Please copy .env.example to .env and configure your API keys")
        else:
            print("📝 Please create .env file with your API keys")
            print("   Required: OPENAI_API_KEY or DEEPSEEK_API_KEY")
    
    # Verify core imports
    try:
        sys.path.insert(0, str(Path.cwd()))
        from src.pipeline import McodePipeline
        print("✅ Core imports verified")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)
    
    print("\n🎉 Production setup complete!")
    print("\n📋 Next steps:")
    print("1. Configure .env with your API keys")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Test with: python mcode_translator.py --help")
    print("4. Process data: python mcode_translator.py data/sample_trial.json")


if __name__ == "__main__":
    setup_production_environment()