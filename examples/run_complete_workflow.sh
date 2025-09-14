#!/bin/bash
# mCODE Translator - Complete Workflow Script
#
# This script demonstrates a complete end-to-end workflow:
# 1. Fetch clinical trials
# 2. Process trials with mCODE mapping
# 3. Fetch patient data
# 4. Process patients with trial eligibility
# 5. Run optimization
#
# Usage: ./examples/run_complete_workflow.sh

set -e  # Exit on any error

# Initialize conda for bash
__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
        . "$(conda info --base)/etc/profile.d/conda.sh"
    else
        export PATH="$(conda info --base)/bin:$PATH"
    fi
fi
unset __conda_setup

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLES_DIR="$PROJECT_ROOT/examples"
DATA_DIR="$EXAMPLES_DIR/data"
WORKFLOW_DIR="$DATA_DIR/workflow_$(date +%Y%m%d_%H%M%S)"

# Create directories
mkdir -p "$WORKFLOW_DIR"
mkdir -p "$DATA_DIR"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check environment
check_environment() {
    log "Checking environment..."

    # Check if in conda environment
    if [[ "$CONDA_DEFAULT_ENV" != *"mcode_translator"* ]]; then
        warning "Not in mcode_translator conda environment"
        warning "Run: conda activate mcode_translator"
        warning "Continuing anyway..."
    else
        success "Conda environment: $CONDA_DEFAULT_ENV"
    fi

    # Check if we're in the right directory
    if [[ ! -f "src/cli/trials_fetcher.py" ]]; then
        error "Not in project root directory"
        error "Please run from: $PROJECT_ROOT"
        exit 1
    fi

    # Check Python
    if ! command -v python &> /dev/null; then
        error "Python not found"
        exit 1
    fi

    success "Environment check passed"
}

# Run CLI command with error handling
run_cli() {
    local description="$1"
    shift
    local cmd_args=("$@")

    log "$description"
    echo "Command: conda activate mcode_translator && python -m ${cmd_args[*]}"

    if conda activate mcode_translator && python -m "${cmd_args[@]}"; then
        success "$description completed"
    else
        error "$description failed"
        return 1
    fi
}

# Main workflow
main() {
    echo "🧪 mCODE Translator - Complete Workflow"
    echo "======================================"
    echo "Workflow directory: $WORKFLOW_DIR"
    echo

    check_environment

    # Step 1: Fetch clinical trials
    echo
    echo "📋 STEP 1: Fetch Clinical Trials"
    echo "-------------------------------"
    run_cli "Fetching breast cancer trials" \
        src.cli.trials_fetcher \
        --condition "breast cancer" \
        --limit 3 \
        -o "$WORKFLOW_DIR/trials.json" \
        --verbose

    # Step 2: Process trials with mCODE
    echo
    echo "🔬 STEP 2: Process Trials with mCODE"
    echo "-----------------------------------"
    run_cli "Processing trials with mCODE mapping" \
        src.cli.trials_processor \
        "$WORKFLOW_DIR/trials.json" \
        --model deepseek-coder \
        --prompt direct_mcode_evidence_based_concise \
        --store-in-core-memory \
        --verbose

    # Step 3: Fetch patient data
    echo
    echo "👥 STEP 3: Fetch Patient Data"
    echo "----------------------------"
    run_cli "Fetching breast cancer patient data" \
        src.cli.patients_fetcher \
        --archive breast_cancer_10_years \
        --limit 5 \
        -o "$WORKFLOW_DIR/patients.json" \
        --verbose

    # Step 4: Process patients with trial eligibility
    echo
    echo "🎯 STEP 4: Process Patients with Trial Eligibility"
    echo "-------------------------------------------------"
    run_cli "Processing patients with trial criteria filtering" \
        src.cli.patients_processor \
        --patients "$WORKFLOW_DIR/patients.json" \
        --trials "$WORKFLOW_DIR/trials.json" \
        --store-in-core-memory \
        --verbose

    # Step 5: Run optimization (optional)
    echo
    echo "⚡ STEP 5: Optimize Parameters"
    echo "----------------------------"
    run_cli "Optimizing mCODE translation parameters" \
        src.cli.trials_optimizer \
        --trials-file "$WORKFLOW_DIR/trials.json" \
        --max-combinations 4 \
        --save-config "$WORKFLOW_DIR/optimized_config.json" \
        --verbose

    # Summary
    echo
    echo "🎉 WORKFLOW COMPLETED SUCCESSFULLY!"
    echo "==================================="
    echo "Generated files in: $WORKFLOW_DIR"
    echo
    echo "Files created:"
    ls -la "$WORKFLOW_DIR/"
    echo
    echo "📊 Summary:"
    echo "• Trials fetched: $(jq 'length' "$WORKFLOW_DIR/trials.json" 2>/dev/null || echo 'N/A')"
    echo "• Patients fetched: $(jq 'length' "$WORKFLOW_DIR/patients.json" 2>/dev/null || echo 'N/A')"
    echo "• Optimization config saved: optimized_config.json"
    echo
    echo "💡 Next steps:"
    echo "• Review the generated JSON files"
    echo "• Check CORE Memory for stored results"
    echo "• Use optimized_config.json for future processing"
    echo "• Run individual CLI commands for more control"
}

# Cleanup on error
cleanup() {
    echo
    warning "Workflow interrupted. Cleaning up..."
    # Could add cleanup logic here if needed
}

# Set up error handling
trap cleanup ERR

# Run main workflow
main "$@"