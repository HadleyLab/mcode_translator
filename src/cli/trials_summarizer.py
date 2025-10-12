#!/usr/bin/env python3
"""
Trials Summarizer CLI - Generate natural language summaries for clinical trials.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflows.trials_summarizer import TrialsSummarizerWorkflow
from src.config.heysol_config import get_config


def main(args):
    """
    CLI main function for trials summarizer.

    Args:
        args: Parsed command line arguments
    """
    try:
        config = get_config()

        # Create workflow instance
        workflow = TrialsSummarizerWorkflow(config)

        # Read input trials
        input_file = getattr(args, 'input_file', None)
        if not input_file:
            print("❌ No input file specified")
            sys.exit(1)

        import json
        trials_data = []
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    trials_data.append(json.loads(line))

        if not trials_data:
            print("❌ No trial data found in input file")
            sys.exit(1)

        # Extract parameters from args
        kwargs = {
            'trials_data': trials_data,
            'store_in_memory': getattr(args, 'ingest', False),
        }

        # Execute workflow
        result = workflow.execute(**kwargs)

        if result.success:
            print("✅ Trials summarization completed successfully!")
            if result.data:
                print(f"Total trials summarized: {len(result.data)}")

            # Write output if specified
            output_file = getattr(args, 'output_file', None)
            if output_file and result.data:
                import json
                with open(output_file, 'w') as f:
                    for trial in result.data:
                        json.dump(trial, f)
                        f.write('\n')
                print(f"💾 Results saved to: {output_file}")

            if result.metadata:
                print(f"Metadata: {result.metadata}")
            sys.exit(0)
        else:
            print(f"❌ Trials summarization failed: {result.error_message}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)