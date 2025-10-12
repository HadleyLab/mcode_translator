#!/usr/bin/env python3
"""
🚀 mCODE Translator - CLI Commands Workflow Demo

This example demonstrates the complete CLI workflow for the mCODE Translator,
showing how to use various CLI commands for data ingestion, processing,
optimization, and analysis.
"""

import sys
import subprocess
import time
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a CLI command and return success status."""
    print(f"\n🔧 {description}")
    print(f"   Command: {cmd}")
    print("-" * 60)

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        if result.returncode == 0:
            print("✅ Success")
            # Show first few lines of output
            lines = result.stdout.strip().split('\n')[:5]
            if lines and lines[0]:
                print("   Output preview:")
                for line in lines:
                    if line.strip():
                        print(f"   {line}")
            return True
        else:
            print("❌ Failed")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False

    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def cli_workflow_demo() -> bool:
    """Demonstrate the complete CLI workflow."""
    print("🚀 mCODE Translator - CLI Commands Workflow Demo")
    print("=" * 70)
    print()
    print("This demo shows the complete CLI workflow:")
    print("1. System status check")
    print("2. Trial data ingestion")
    print("3. mCODE processing")
    print("4. Optimization")
    print("5. Results analysis")
    print()

    # Step 1: Check system status
    success = run_command(
        "python mcode-cli.py status",
        "Step 1: Checking system status and connectivity"
    )
    if not success:
        print("⚠️  System status check failed, but continuing with demo...")
    time.sleep(1)

    # Step 2: Ingest trial data (small batch for demo)
    success = run_command(
        "python mcode-cli.py data ingest-trials --cancer-type 'breast' --limit 2 --engine 'regex'",
        "Step 2: Ingesting breast cancer trial data (RegexEngine)"
    )
    if not success:
        print("⚠️  Data ingestion failed, but continuing with demo...")
    time.sleep(1)

    # Step 3: Process with mCODE extraction
    success = run_command(
        "python mcode-cli.py mcode summarize NCT02364999 --engine 'regex'",
        "Step 3: Extracting mCODE elements from a specific trial"
    )
    if not success:
        print("⚠️  mCODE processing failed, but continuing with demo...")
    time.sleep(1)

    # Step 4: Compare engines
    success = run_command(
        "python mcode-cli.py mcode summarize NCT02364999 --compare-engines",
        "Step 4: Comparing RegexEngine vs LLMEngine performance"
    )
    if not success:
        print("⚠️  Engine comparison failed, but continuing with demo...")
    time.sleep(1)

    # Step 5: Show optimization options
    success = run_command(
        "python mcode-cli.py mcode optimize-trials --help",
        "Step 5: Showing optimization command options"
    )
    if not success:
        print("⚠️  Optimization help failed, but continuing with demo...")
    time.sleep(1)

    # Step 6: Memory operations (if available)
    success = run_command(
        "python mcode-cli.py memory stats",
        "Step 6: Checking CORE Memory statistics"
    )
    if not success:
        print("⚠️  Memory stats not available (API key not configured)")
    time.sleep(1)

    # Step 7: Configuration check
    success = run_command(
        "python mcode-cli.py config show",
        "Step 7: Displaying current configuration"
    )
    if not success:
        print("⚠️  Config check failed, but continuing with demo...")
    time.sleep(1)

    print("\n🎉 CLI Workflow Demo completed!")
    print()
    print("💡 Key CLI Commands Demonstrated:")
    print("   • status          - System health check")
    print("   • data ingest-trials - Fetch and process trial data")
    print("   • mcode summarize  - Extract mCODE from specific trials")
    print("   • mcode optimize-trials - Optimize processing parameters")
    print("   • memory stats     - CORE Memory usage statistics")
    print("   • config show      - Current configuration display")
    print()
    print("🔧 Command Categories:")
    print("   • data             - Data ingestion and fetching")
    print("   • mcode            - mCODE processing and analysis")
    print("   • memory           - CORE Memory operations")
    print("   • config           - Configuration management")
    print("   • patients         - Patient data processing")
    print("   • trials           - Trial-specific operations")
    print()
    print("📚 Getting Help:")
    print("   python mcode-cli.py --help          # Main help")
    print("   python mcode-cli.py <command> --help # Command-specific help")
    print()
    print("⚙️  Configuration:")
    print("   Set HEYSOL_API_KEY for LLM features and CORE Memory")
    print("   Use 'python mcode-cli.py config setup' for guided setup")

    return True


if __name__ == "__main__":
    success = cli_workflow_demo()
    sys.exit(0 if success else 1)