import sys
import json
import pytest
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.pipeline.mcode_pipeline import McodePipeline
from src.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

# Define paths
EXAMPLES_DIR = Path("examples/breast_cancer_data")
TEST_CASES_FILE = EXAMPLES_DIR / "breast_cancer_her2_positive.trial.json"
GOLD_STANDARD_FILE = EXAMPLES_DIR / "breast_cancer_her2_positive.gold.json"

# Load test data
logger.info("Loading test cases data...")
with open(TEST_CASES_FILE, "r") as f:
    test_cases_data = json.load(f)

# Load gold standard data
logger.info("Loading gold standard data...")
with open(GOLD_STANDARD_FILE, "r") as f:
    gold_standard_data = json.load(f)

# Get test cases and gold standards
test_cases = test_cases_data.get("test_cases", {})
gold_standards = gold_standard_data.get("gold_standard", {})

# Prepare test data by pairing test cases with their gold standards
logger.info("Preparing test data...")
test_data = []
for case_id, case_data in test_cases.items():
    if case_id in gold_standards:
        test_data.append(
            {
                "id": case_id,
                "trial_context": case_data,
                "expected_mcode": gold_standards[case_id].get("expected_mcode_mappings", {})
            }
        )

logger.info(f"Prepared {len(test_data)} test cases.")

@pytest.mark.parametrize("trial_data", test_data)
def test_direct_mcode_pipeline(trial_data):
    """
    Test the Direct to Mcode pipeline with various cancer trial data.
    """
    logger.info(f"Testing with trial: {trial_data['id']}")

    # Initialize the pipeline
    logger.info("Initializing pipeline...")
    pipeline = McodePipeline(prompt_name="direct_text_to_mcode_mapping")

    # Run the pipeline
    logger.info("Running pipeline...")
    result = pipeline.process_clinical_trial(trial_data["trial_context"])
    
    # Basic validation
    assert result is not None, "Pipeline should produce a result."
    assert "mcode_mappings" in result.__dict__, "Result should have 'mcode_mappings' attribute."
    
    # Check that we have some mappings
    generated_mappings = result.mcode_mappings
    assert len(generated_mappings) > 0, "Pipeline should generate at least one Mcode mapping"
    
    logger.info(f"Generated mappings: {len(generated_mappings)}")
    
    # Save generated mappings for analysis
    import json
    output_file = Path(f"direct_pipeline_output_{trial_data['id']}.json")
    with open(output_file, 'w') as f:
        json.dump({
            'trial_id': trial_data['id'],
            'generated_mappings': generated_mappings,
            'expected_mappings': trial_data["expected_mcode"].get("mapped_elements", [])
        }, f, indent=2)
    logger.info(f"Generated mappings saved to {output_file}")
    
    # Check that each mapping has the required fields
    for mapping in generated_mappings:
        assert "code" in mapping, "Each mapping should have a 'code' field"
        assert "mapping_rationale" in mapping, "Each mapping should have a 'mapping_rationale' field"
        assert "mapping_confidence" in mapping, "Each mapping should have a 'mapping_confidence' field"
        
    logger.info(f"Test passed for trial: {trial_data['id']}")
        
    logger.info(f"Test passed for trial: {trial_data['id']}")
