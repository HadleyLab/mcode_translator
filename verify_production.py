#!/usr/bin/env python3
"""
mCODE Translator v2.0 - Production Verification Script

Verifies all critical components for production deployment.
"""

import os
import sys
import json
import importlib.util
from pathlib import Path

def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a required file exists."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (MISSING)")
        return False

def check_json_valid(filepath: str, description: str) -> bool:
    """Check if JSON file is valid."""
    try:
        with open(filepath, 'r') as f:
            json.load(f)
        print(f"✅ {description}: Valid JSON")
        return True
    except Exception as e:
        print(f"❌ {description}: Invalid JSON - {e}")
        return False

def check_python_import(module_path: str, description: str) -> bool:
    """Check if Python module imports successfully."""
    try:
        if module_path.startswith('src.'):
            # Add src to path temporarily
            sys.path.insert(0, 'src')
            module_name = module_path[4:]  # Remove 'src.' prefix
        else:
            module_name = module_path
        
        importlib.import_module(module_name)
        print(f"✅ {description}: Import successful")
        return True
    except Exception as e:
        print(f"❌ {description}: Import failed - {e}")
        return False

def main():
    """Run production verification checks."""
    print("🔍 mCODE Translator v2.0 - Production Verification")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 0
    
    # Core files
    core_files = [
        ("mcode_translator.py", "Main CLI interface"),
        ("README.md", "Documentation"),
        ("requirements.txt", "Dependencies"),
        ("setup_production.py", "Production setup"),
        ("LICENSE", "License file"),
        (".env.example", "Environment template"),
    ]
    
    for filepath, description in core_files:
        total_checks += 1
        if check_file_exists(filepath, description):
            checks_passed += 1
    
    # Configuration files
    config_files = [
        ("data/config.json", "Core configuration"),
        ("models/models_config.json", "Model configuration"),
        ("prompts/prompts_config.json", "Prompt configuration"),
        ("data/gold_standard_reference.json", "Gold standard reference"),
    ]
    
    for filepath, description in config_files:
        total_checks += 1
        if check_file_exists(filepath, description):
            checks_passed += 1
            total_checks += 1
            if check_json_valid(filepath, f"{description} (JSON validity)"):
                checks_passed += 1
    
    # Python modules
    python_modules = [
        ("src.pipeline", "Core pipeline module"),
        ("src.utils.config", "Configuration utilities"),
        ("src.utils.prompt_loader", "Prompt loader"),
        ("src.utils.logging_config", "Logging configuration"),
    ]
    
    for module_path, description in python_modules:
        total_checks += 1
        if check_python_import(module_path, description):
            checks_passed += 1
    
    # Directory structure
    required_dirs = [
        "src/pipeline",
        "src/utils", 
        "prompts",
        "models",
        "data",
        "tests",
        "archive"
    ]
    
    for dir_path in required_dirs:
        total_checks += 1
        if check_file_exists(dir_path, f"Directory: {dir_path}"):
            checks_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Verification Results: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("🎉 Production verification PASSED - Ready for deployment!")
        return 0
    else:
        print("⚠️  Production verification FAILED - Issues need resolution")
        return 1

if __name__ == "__main__":
    sys.exit(main())