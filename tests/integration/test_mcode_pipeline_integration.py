"""
Complete integration test for the mCODE extraction pipeline.
This test runs the actual extraction and validates the results.
"""

import pytest
import json
import os
import tempfile
from pathlib import Path


class TestMCODEPipelineIntegration:
    """Integration tests for the complete mCODE extraction pipeline"""
    
    @pytest.fixture
    def nct_id(self):
        """Return the NCT ID for testing"""
        return "NCT01698281"
    
    @pytest.fixture
    def output_file(self):
        """Create a temporary output file for test results"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            yield f.name
        # Clean up after test
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    def test_complete_mcode_extraction_pipeline(self, nct_id, output_file):
        """
        Test the complete mCODE extraction pipeline from CLI execution
        to result validation.
        """
        # Run the fetcher with process-criteria flag
        import subprocess
        import sys
        
        # Build the command
        cmd = [
            sys.executable, 
            "src/data_fetcher/fetcher.py", 
            "--nct-id", nct_id,
            "--process-criteria",
            "--export", output_file
        ]
        
        # Execute the command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        # Verify command executed successfully
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        
        # Verify output file was created
        assert os.path.exists(output_file), "Output file was not created"
        
        # Load and validate the results
        with open(output_file, 'r') as f:
            study_data = json.load(f)
        
        # Verify basic study structure
        assert "protocolSection" in study_data
        assert "mcodeResults" in study_data
        
        # Extract mCODE results
        mcode_results = study_data["mcodeResults"]
        
        # Validate mCODE results structure
        self._validate_mcode_results_structure(mcode_results)
        
        # Validate specific content from NCT01698281
        self._validate_nct01698281_content(mcode_results)
        
        # Save the validated results for reference
        self._save_validated_results(study_data, nct_id)
    
    def _validate_mcode_results_structure(self, mcode_results):
        """Validate the structure of mCODE results"""
        # Verify all required sections exist
        assert "nlp" in mcode_results
        assert "codes" in mcode_results
        assert "mappings" in mcode_results
        assert "structured_data" in mcode_results
        assert "validation" in mcode_results
        
        # Validate NLP section
        nlp = mcode_results["nlp"]
        assert "entities" in nlp
        assert "features" in nlp
        
        # Validate features
        features = nlp["features"]
        required_features = [
            "genomic_variants", "biomarkers", "cancer_characteristics",
            "treatment_history", "performance_status", "demographics"
        ]
        for feature in required_features:
            assert feature in features, f"Missing feature: {feature}"
        
        # Validate codes section
        codes = mcode_results["codes"]
        assert "extracted_codes" in codes
        assert "mapped_entities" in codes
        assert "metadata" in codes
        
        # Validate structured data section
        structured_data = mcode_results["structured_data"]
        assert "bundle" in structured_data
        assert "resources" in structured_data
        assert "validation" in structured_data
        
        # Validate validation section
        validation = mcode_results["validation"]
        assert "valid" in validation
        assert "errors" in validation
        assert "warnings" in validation
        assert "compliance_score" in validation
    
    def _validate_nct01698281_content(self, mcode_results):
        """Validate specific content from NCT01698281 extraction"""
        features = mcode_results["nlp"]["features"]
        
        # Validate biomarkers - should be triple negative
        biomarkers = features["biomarkers"]
        er_biomarker = next((bm for bm in biomarkers if bm["name"] == "ER"), None)
        pr_biomarker = next((bm for bm in biomarkers if bm["name"] == "PR"), None)
        her2_biomarker = next((bm for bm in biomarkers if bm["name"] == "HER2"), None)
        lhrh_biomarker = next((bm for bm in biomarkers if bm["name"] == "LHRHRECEPTOR"), None)
        
        assert er_biomarker is not None and er_biomarker["status"] == "negative"
        assert pr_biomarker is not None and pr_biomarker["status"] == "negative"
        assert her2_biomarker is not None and her2_biomarker["status"] == "negative"
        assert lhrh_biomarker is not None and lhrh_biomarker["status"] == "positive"
        
        # Validate demographics - should be female only, age ≥ 18
        demographics = features["demographics"]
        assert demographics["gender"] == ["female"]
        assert demographics["age"]["min"] == 18
        
        # Validate cancer characteristics - should be metastatic
        cancer_chars = features["cancer_characteristics"]
        assert "Stage IV" in cancer_chars["stage"]
        assert "metastatic" in cancer_chars["stage"].lower()
        
        # Validate treatment history - should mention chemotherapy constraints
        treatment_history = features["treatment_history"]
        assert len(treatment_history["chemotherapy"]) > 0
        
        # Validate performance status - ECOG ≤ 2
        performance_status = features["performance_status"]
        assert "≤2" in performance_status["ecog"]
    
    def _save_validated_results(self, study_data, nct_id):
        """Save validated results for reference and future testing"""
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"mcode_results_{nct_id}.json"
        
        with open(output_file, 'w') as f:
            json.dump(study_data, f, indent=2)
        
        print(f"Validated mCODE results saved to: {output_file}")
    
    def test_mcode_results_consistency(self):
        """Test that mCODE results maintain consistency across runs"""
        # This test would compare results from multiple runs to ensure consistency
        # For now, we'll validate that our test data structure is consistent
        
        test_data_file = Path("test_results/mcode_results_NCT01698281.json")
        if test_data_file.exists():
            with open(test_data_file, 'r') as f:
                previous_results = json.load(f)
            
            # Basic structure validation
            assert "mcodeResults" in previous_results
            mcode_results = previous_results["mcodeResults"]
            
            # Validate structure consistency
            self._validate_mcode_results_structure(mcode_results)
            
            # Validate content consistency
            self._validate_nct01698281_content(mcode_results)
    
    def test_mcode_compliance_validation(self):
        """Test that mCODE validation results are reasonable"""
        test_data_file = Path("test_results/mcode_results_NCT01698281.json")
        if test_data_file.exists():
            with open(test_data_file, 'r') as f:
                study_data = json.load(f)
            
            mcode_results = study_data["mcodeResults"]
            validation = mcode_results["validation"]
            
            # Validation should be successful
            assert validation["valid"] is True
            
            # Compliance score should be reasonable
            assert 0.5 <= validation["compliance_score"] <= 1.0
            
            # Should not have critical errors
            assert len(validation["errors"]) == 0
            
            # Warnings are acceptable (e.g., for missing optional fields)
            assert isinstance(validation["warnings"], list)


if __name__ == '__main__':
    pytest.main([__file__, "-v"])