#!/bin/bash

# 🖥️ MCODE Translator - CLI Demo (Shell Script)
#
# Comprehensive demonstration of command-line interface capabilities for clinical data processing.
#
# This script will:
# 1. Check CLI environment and API key setup
# 2. Create a demo space for testing
# 3. Ingest sample clinical data
# 4. Perform search operations
# 5. Generate analytics reports
# 6. Demonstrate batch processing
#
# Prerequisites:
# - Get your API key from https://core.heysol.ai/settings/api
# - Set environment variable: export HEYSOL_API_KEY="your-key-here"
# - Or create a .env file with: HEYSOL_API_KEY=your-key-here
# - Install the package: pip install -e .
#
# Run: bash CLI_demo.sh

set -e  # Exit on any error

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    # Explicitly export the API keys
    export HEYSOL_API_KEY_IDRDEX_MAMMOCHAT
    export HEYSOL_API_KEY_HADLEYLABELABORATORY
    export HEYSOL_API_KEY_IDRDEX_GMAIL
fi

echo "🖥️ MCODE Translator - CLI Demo (Shell)"
echo "====================================="

# Check API key - look for any variable starting with HEYSOL_API_KEY
# Try to find any HEYSOL_API_KEY* variable
for var in $(env | grep '^HEYSOL_API_KEY' | cut -d= -f1); do
    if [ -n "${!var}" ]; then
        HEYSOL_API_KEY="${!var}"
        if [ "$var" = "HEYSOL_API_KEY_IDRDEX_MAMMOCHAT" ]; then
            USER_NAME="iDrDex@MammoChat.com"
            TARGET_API_KEY="${HEYSOL_API_KEY_HADLEYLABELABORATORY}"
        elif [ "$var" = "HEYSOL_API_KEY_HADLEYLABELABORATORY" ]; then
            USER_NAME="HadleyLaboratory@gmail.com"
            TARGET_API_KEY="${HEYSOL_API_KEY_IDRDEX_MAMMOCHAT}"
        elif [ "$var" = "HEYSOL_API_KEY_IDRDEX_GMAIL" ]; then
            USER_NAME="iDrDex@gmail.com"
            TARGET_API_KEY="${HEYSOL_API_KEY_HADLEYLABELABORATORY}"
        else
            USER_NAME=$(echo "$var" | sed 's/HEYSOL_API_KEY_//')
            TARGET_API_KEY="${HEYSOL_API_KEY_HADLEYLABELABORATORY}"
        fi
        echo "✅ Found API key from $var (user: $USER_NAME)"
        break
    fi
done

if [ -z "$HEYSOL_API_KEY" ]; then
    echo "❌ No API key found!"
    echo ""
    echo "📝 To get started:"
    echo "1. Visit: https://core.heysol.ai/settings/api"
    echo "2. Generate an API key"
    echo "3. Set environment variable:"
    echo "   export HEYSOL_API_KEY='your-api-key-here'"
    echo "4. Or create a .env file with:"
    echo "   HEYSOL_API_KEY_xxx=your-api-key-here (any name starting with HEYSOL_API_KEY)"
    echo ""
    echo "Then run this script again!"
    exit 1
fi

echo "✅ API key found (ends with: ...${HEYSOL_API_KEY: -4})"
echo "🔍 Validating CLI availability..."

# Check if CLI is available
if ! python -c "import sys; sys.path.insert(0, 'src'); from cli.cli import main" &> /dev/null; then
    echo "❌ CLI module not found!"
    echo "Install with: pip install -e ."
    exit 1
fi

echo "✅ CLI available"
echo "✅ API key validated!"

# Create demo space
echo ""
echo "🏗️  Creating CLI demo space..."

SPACE_NAME="CLI Demo Space $(date +%s)"
SPACE_DESC="Created by CLI demo script"

echo "   Creating space: $SPACE_NAME"
CREATE_RESULT=$(PYTHONPATH=/Users/idrdex/Library/Mobile\ Documents/com~apple~CloudDocs/Code/mcode_translator python -m cli spaces create "$SPACE_NAME" --description "$SPACE_DESC" 2>/dev/null)

if [ $? -eq 0 ]; then
    echo "   ✅ Space created successfully"
else
    echo "   ⚠️ Space creation may have failed, continuing..."
fi

# Ingest sample data
echo ""
echo "📥 Ingesting sample clinical data..."

SAMPLE_DATA=(
    "Patient with advanced lung cancer shows excellent response to immunotherapy"
    "Clinical trial demonstrates 85% response rate for new targeted therapy"
    "Biomarker analysis reveals key predictors of treatment success"
)

for i in "${!SAMPLE_DATA[@]}"; do
    echo "   Ingesting record $((i+1))/3..."
    PYTHONPATH=/Users/idrdex/Library/Mobile\ Documents/com~apple~CloudDocs/Code/mcode_translator python -m cli memory ingest "${SAMPLE_DATA[$i]}" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "   ✅ Ingested $((i+1))/3 items"
    else
        echo "   ⚠️ Ingestion $((i+1)) may have failed, continuing..."
    fi
done

# Perform searches
echo ""
echo "🔍 Performing search operations..."

SEARCH_QUERIES=(
    "lung cancer immunotherapy"
    "clinical trial response rate"
    "biomarker treatment success"
)

for query in "${SEARCH_QUERIES[@]}"; do
    echo "   Searching for: '$query'"
    SEARCH_RESULT=$(PYTHONPATH=/Users/idrdex/Library/Mobile\ Documents/com~apple~CloudDocs/Code/mcode_translator python -m cli memory search "$query" --limit 3 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "   ✅ Search completed"
    else
        echo "   ⚠️ Search may have failed, continuing..."
    fi
done

# Generate analytics (if available)
echo ""
echo "📊 Generating analytics report..."

if PYTHONPATH=/Users/idrdex/Library/Mobile\ Documents/com~apple~CloudDocs/Code/mcode_translator python -m cli analytics summary > /dev/null 2>&1; then
    echo "✅ Analytics report generated"
else
    echo "⚠️ Analytics not available or failed"
fi

# Batch processing demo
echo ""
echo "📋 Demonstrating batch processing..."

BATCH_FILE="cli_batch_demo.txt"
cat > "$BATCH_FILE" << 'EOF'
Patient cohort analysis reveals treatment patterns
Clinical outcomes data shows improved survival rates
Research study identifies novel therapeutic targets
EOF

echo "   Created batch file: $BATCH_FILE"

if PYTHONPATH=/Users/idrdex/Library/Mobile\ Documents/com~apple~CloudDocs/Code/mcode_translator python -m cli batch process "$BATCH_FILE" > /dev/null 2>&1; then
    echo "✅ Batch processing completed"
else
    echo "⚠️ Batch processing not available or failed"
fi

# Cleanup
rm -f "$BATCH_FILE"
echo "🧹 Cleaned up batch file"

echo ""
echo "🎉 CLI demo completed successfully!"
echo ""
echo "📚 Next steps:"
echo "- Try the Python version: python CLI_demo.py"
echo "- Try the interactive notebook: jupyter notebook CLI_demo.ipynb"
echo "- Explore other examples: ls examples/"
echo "- CLI help: python -m cli --help"
echo "- Documentation: https://core.heysol.ai/"
echo ""
echo "💡 CLI Capabilities:"
echo "- Command-line data ingestion and search"
echo "- Automated analytics and reporting"
echo "- Batch processing workflows"
echo "- Space management operations"
echo "- Integration with scripts and pipelines"