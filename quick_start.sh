#!/bin/bash

# 🚀 MCODE Translator - Quick Start Script
# Get up and running with MCODE Translator in under 5 minutes!

echo "🚀 MCODE Translator - Quick Start"
echo "================================="
echo ""

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Check API key - use default HEYSOL_API_KEY unless registry is in effect
if [ -z "$HEYSOL_API_KEY" ]; then
    echo "❌ No API key found!"
    echo ""
    echo "📝 To get started:"
    echo "1. Visit: https://core.heysol.ai/settings/api"
    echo "2. Generate an API key"
    echo "3. Set environment variable:"
    echo "   export HEYSOL_API_KEY='your-api-key-here'"
    echo "4. Or create a .env file with:"
    echo "   HEYSOL_API_KEY=your-api-key-here"
    echo ""
    echo "Then run this script again!"
    exit 1
fi

echo "✅ API key found (ends with: ...${HEYSOL_API_KEY: -4})"
echo ""

# Check if we're in the right directory
if [ ! -f "src/cli/cli.py" ]; then
    echo "❌ Not in MCODE Translator directory!"
    echo "💡 Please run this script from the mcode_translator directory"
    exit 1
fi

echo "✅ In MCODE Translator directory"
echo ""

# Install dependencies if needed
echo "🔧 Checking dependencies..."
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo "💡 Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -e .
else
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    echo "✅ Dependencies ready"
fi

echo ""

# Run the quick start script
echo "🎯 Running MCODE Translator quick start..."
echo ""

python quick_start.py

echo ""
echo "🎉 Quick start completed!"
echo ""
echo "💡 What to try next:"
echo "   📖 Explore examples: ls examples/"
echo "   🖥️ Use CLI: python -m cli --help"
echo "   📚 Read docs: cat README.md"
echo "   🔬 Try the comprehensive demo: python examples/comprehensive_demo.py"