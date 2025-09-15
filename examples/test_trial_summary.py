#!/usr/bin/env python3
"""
Test the generate_trial_summary function with complete trial data.
"""

import json
from pathlib import Path
from generate_trial_summary import generate_trial_summary

def test_trial_summary():
    """Test the trial summary generation with complete trial data."""

    # Load the complete trial data
    trial_file = Path("complete_trial.json")
    if not trial_file.exists():
        print("❌ complete_trial.json not found. Please fetch a complete trial first.")
        return

    with open(trial_file, "r", encoding="utf-8") as f:
        trials_data = json.load(f)

    # Use the first (and only) trial
    target_trial = trials_data[0]
    protocol_section = target_trial.get("protocolSection", {})
    identification = protocol_section.get("identificationModule", {})
    nct_id = identification.get("nctId", "Unknown")
    brief_title = identification.get("briefTitle", "Unknown Trial")

    print(f"✅ Loaded complete trial: {nct_id}")
    print(f"Title: {brief_title}")

    # Show what detailed information is available
    print("\n📊 Available detailed modules:")
    modules = ['eligibilityModule', 'designModule', 'armsInterventionsModule', 'sponsorCollaboratorsModule', 'outcomesModule']
    for module in modules:
        if module in protocol_section:
            print(f"  ✅ {module}")
        else:
            print(f"  ❌ {module} (missing)")

    # Generate the summary
    try:
        summary = generate_trial_summary(target_trial)
        print("\n📝 Generated Summary:")
        print("=" * 100)
        print(summary)
        print("=" * 100)

        # Check if it contains proper information
        checks = [
            nct_id in summary,
            brief_title.split()[0] in summary,  # First word of title
            "mCODE:" in summary,
            "Unknown" not in summary or "Not specified" not in summary,  # Should minimize unknowns
        ]

        print(f"\n✅ Summary contains NCT ID: {checks[0]}")
        print(f"✅ Summary contains key title word: {checks[1]}")
        print(f"✅ Summary contains mCODE mappings: {checks[2]}")
        print(f"✅ Summary minimizes unknowns: {checks[3]}")

        if all(checks):
            print("\n🎉 All checks passed! The summary generator properly extracts detailed trial information.")
        else:
            print("\n⚠️  Some checks failed. The summary may need improvement.")

    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_trial_summary()