#!/usr/bin/env python3
"""
Complete mCODE Translation Workflow - Command Line Version

Converted from Jupyter notebook for command-line execution.
"""

import os
import sys
from pathlib import Path

# Change to project root directory
current_dir = Path.cwd()
if current_dir.name == 'examples':
    project_root = current_dir.parent
    os.chdir(project_root)
    print("📁 Changed working directory to:", project_root)
else:
    project_root = current_dir

# Add project root to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print("✅ Added project root to Python path")

# Configure API keys (replace with your actual keys)
os.environ['CLINICAL_TRIALS_API_KEY'] = 'your_clinical_trials_api_key_here'
os.environ['COREAI_API_KEY'] = 'your_core_memory_api_key_here'

print("✅ API keys configured")
print("✅ Environment ready")
print("📍 Current working directory:", Path.cwd())

# Import required modules
import json
import subprocess

# Get available models and prompts dynamically
print()
print("🤖 Discovering available models and prompts...")

# Get available models
models_output = subprocess.run([
    "python", "-m", "src.cli.trials_optimizer", "--list-models"
], capture_output=True, text=True, cwd=".")

# Get available prompts
prompts_output = subprocess.run([
    "python", "-m", "src.cli.trials_optimizer", "--list-prompts"
], capture_output=True, text=True, cwd=".")

# Parse the output to extract model and prompt names
models_lines = models_output.stdout.strip().split('\n')
prompts_lines = prompts_output.stdout.strip().split('\n')

# Extract model names (skip the header line)
AVAILABLE_MODELS = [line.split('• ')[1] for line in models_lines if '• ' in line]

# Extract prompt names (skip the header line)
AVAILABLE_PROMPTS = [line.split('• ')[1] for line in prompts_lines if '• ' in line]

print("🤖 Available models (" + str(len(AVAILABLE_MODELS)) + "):", ', '.join(AVAILABLE_MODELS))
print("📝 Available prompts (" + str(len(AVAILABLE_PROMPTS)) + "):", ', '.join(AVAILABLE_PROMPTS))

# Create comma-separated strings for command-line usage
MODELS_STR = ','.join(AVAILABLE_MODELS)
PROMPTS_STR = ','.join(AVAILABLE_PROMPTS)

print()
print("📋 Command-ready strings:")
print("   Models:", MODELS_STR)
print("   Prompts:", PROMPTS_STR)

print()
print("🧪 Full optimization using ALL available models and prompts")
combinations = len(AVAILABLE_MODELS) * len(AVAILABLE_PROMPTS)
print("This tests all combinations:", len(AVAILABLE_MODELS), "models ×", len(AVAILABLE_PROMPTS), "prompts =", combinations, "total combinations")
print("The optimizer only works with files, never uses APIs directly")
print("Using 15 concurrent workers for maximum speed")

# Run the optimization
print()
print("🚀 Starting optimization...")
print("💡 Note: Using default trials file. Specify --trials-file to use a different file.")
result = subprocess.run([
    "python", "-m", "src.cli.trials_optimizer",
    "--cv-folds", "3",
    "--prompts", PROMPTS_STR,
    "--models", MODELS_STR,
    "--max-combinations", "0",
    "--save-config", "optimal_config.json",
    "--workers", "15",
    "--verbose"
], cwd=".")

print()
print("🎉 Optimization completed with exit code:", result.returncode)

# Load optimal config
config_file = Path('optimal_config.json')
if config_file.exists():
    with open(config_file, 'r') as f:
        optimal_config = json.load(f)

    BEST_MODEL = optimal_config['optimal_settings']['model']
    BEST_PROMPT = optimal_config['optimal_settings']['prompt']

    print("🎯 Optimal configuration:")
    print("   Model:", BEST_MODEL)
    print("   Prompt:", BEST_PROMPT)
    print("   CV score:", str(optimal_config['optimal_settings']['cv_score'])[:5])
else:
    print("⚠️  Using default configuration (optimization may have failed)")
    BEST_MODEL = 'deepseek-coder'
    BEST_PROMPT = 'direct_mcode_evidence_based_concise'
    print("   Model:", BEST_MODEL)
    print("   Prompt:", BEST_PROMPT)

print()
print("✅ mCODE translation workflow completed!")
