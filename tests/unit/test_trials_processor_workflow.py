"""
Unit tests for ClinicalTrialsProcessorWorkflow.
"""

from unittest.mock import Mock, patch

import pytest

from src.workflows.trials_processor import TrialsProcessor


class TestTrialsProcessor:
    """Test cases for TrialsProcessor."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return Mock()

    @pytest.fixture
    def workflow(self, config):
        """Create a test workflow instance."""
        return TrialsProcessor(config)

    def test_workflow_initialization(self, workflow):
        """Test workflow initializes correctly."""
        assert workflow is not None
        assert hasattr(workflow, "execute")
        assert hasattr(workflow, "process_single_trial")

    @patch("src.workflows.trials_processor_workflow.McodePipeline")
    def test_execute_no_trials_data(self, mock_pipeline_class, workflow):
        """Test execute fails with no trial data."""
        result = workflow.execute(trials_data=[])

        assert result.success is False
        assert "No trial data provided" in result.error_message

    @patch("src.workflows.trials_processor_workflow.McodePipeline")
    @patch("asyncio.run")
    def test_execute_success(self, mock_asyncio_run, mock_pipeline_class, workflow):
        """Test successful execution."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline

        # Mock async processing result
        mock_asyncio_run.return_value = ([{"trial_id": "NCT123", "processed": True}], 1, 0)

        result = workflow.execute(trials_data=[{"nct_id": "NCT123"}])

        assert result.success is True
        assert len(result.data) == 1
        assert result.metadata["total_trials"] == 1
        assert result.metadata["successful"] == 1
        assert result.metadata["failed"] == 0

    def test_process_single_trial_validation(self, workflow):
        """Test single trial processing with validation."""
        trial = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT123"},
                "eligibilityModule": {"eligibilityCriteria": "Test criteria"},
            }
        }

        # Should pass validation
        assert workflow.validate_trial_data(trial) is True

    def test_process_single_trial_validation_missing_nct(self, workflow):
        """Test single trial processing validation fails without NCT ID."""
        trial = {"protocolSection": {"identificationModule": {}}}

        # Should fail validation
        assert workflow.validate_trial_data(trial) is False

    def test_get_processing_stats_no_pipeline(self, workflow):
        """Test getting processing stats when pipeline not initialized."""
        stats = workflow.get_processing_stats()

        assert stats["status"] == "pipeline_not_initialized"

    @patch("src.workflows.trials_processor_workflow.McodePipeline")
    def test_get_processing_stats_with_pipeline(self, mock_pipeline_class, workflow):
        """Test getting processing stats when pipeline is initialized."""
        # Initialize pipeline
        mock_pipeline = Mock()
        mock_pipeline.llm_mapper = Mock()
        mock_pipeline.llm_mapper.model_name = "test-model"
        mock_pipeline.prompt_name = "test-prompt"
        mock_pipeline_class.return_value = mock_pipeline
        workflow.pipeline = mock_pipeline

        stats = workflow.get_processing_stats()

        assert stats["status"] == "ready"
        assert stats["model"] == "test-model"
        assert stats["prompt_template"] == "test-prompt"

    def test_check_trial_has_full_data_none_trial(self, workflow):
        """Test checking full data with None trial."""
        assert workflow._check_trial_has_full_data(None) is False

    def test_check_trial_has_full_data_empty_dict(self, workflow):
        """Test checking full data with empty dict."""
        assert workflow._check_trial_has_full_data({}) is False

    def test_check_trial_has_full_data_minimal_trial(self, workflow):
        """Test checking full data with minimal trial structure."""
        trial = {
            "protocolSection": {
                "eligibilityModule": {
                    "eligibilityCriteria": "Test criteria that is long enough to pass the check for detailed eligibility and contains more than 100 characters to satisfy the requirement for full trial data validation in the system."
                },
                "armsInterventionsModule": {
                    "interventions": [{"description": "Detailed intervention"}]
                },
                "outcomesModule": {"primaryOutcomes": [{"title": "Primary Outcome"}]},
            },
            "derivedSection": {},
        }
        assert workflow._check_trial_has_full_data(trial) is True

    def test_extract_trial_metadata_none_trial(self, workflow):
        """Test extracting metadata from None trial."""
        metadata = workflow._extract_trial_metadata(None)
        assert isinstance(metadata, dict)
        assert len(metadata) == 0

    def test_extract_trial_metadata_empty_trial(self, workflow):
        """Test extracting metadata from empty trial."""
        metadata = workflow._extract_trial_metadata({})
        assert isinstance(metadata, dict)
        # Empty trial still produces metadata structure with None values
        assert "nct_id" in metadata
        assert metadata["nct_id"] is None

    def test_extract_trial_metadata_full_trial(self, workflow):
        """Test extracting metadata from complete trial."""
        trial = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT123456",
                    "briefTitle": "Test Trial",
                    "officialTitle": "Official Test Trial",
                },
                "statusModule": {
                    "overallStatus": "Recruiting",
                    "startDateStruct": {"date": "2023-01-01"},
                    "completionDateStruct": {"date": "2024-01-01"},
                },
                "designModule": {
                    "studyType": "Interventional",
                    "phases": ["Phase 2"],
                    "primaryPurpose": "Treatment",
                },
                "eligibilityModule": {
                    "minimumAge": "18 Years",
                    "maximumAge": "65 Years",
                    "sex": "All",
                    "healthyVolunteers": False,
                },
                "conditionsModule": {"conditions": [{"name": "Breast Cancer"}]},
                "armsInterventionsModule": {"interventions": [{"name": "Drug A"}]},
            }
        }

        metadata = workflow._extract_trial_metadata(trial)

        assert metadata["nct_id"] == "NCT123456"
        assert metadata["brief_title"] == "Test Trial"
        assert metadata["overall_status"] == "Recruiting"
        assert metadata["study_type"] == "Interventional"
        assert metadata["minimum_age"] == "18 Years"
        assert metadata["conditions"] == ["Breast Cancer"]
        assert metadata["interventions"] == ["Drug A"]

    def test_extract_trial_id_success(self, workflow):
        """Test successful trial ID extraction."""
        trial = {"protocolSection": {"identificationModule": {"nctId": "NCT123456"}}}
        result = workflow._extract_trial_id(trial)
        assert result == "NCT123456"

    def test_extract_trial_id_missing_fields(self, workflow):
        """Test trial ID extraction with missing fields."""
        trial = {}
        result = workflow._extract_trial_id(trial)
        assert result.startswith("unknown_trial_")

    def test_format_trial_mcode_element_basic(self, workflow):
        """Test formatting mCODE element."""
        result = workflow._format_trial_mcode_element("CancerCondition", "SNOMED", "12345")
        assert "(mCODE: CancerCondition; SNOMED:12345)" in result

    def test_convert_trial_mcode_to_mappings_format_list(self, workflow):
        """Test converting mCODE elements list to mappings format."""
        mcode_elements = {
            "conditions": [{"display": "Breast Cancer", "system": "SNOMED", "code": "123"}]
        }

        mappings = workflow._convert_trial_mcode_to_mappings_format(mcode_elements)

        assert len(mappings) == 1
        assert mappings[0]["mcode_element"] == "conditions"
        assert mappings[0]["value"] == "Breast Cancer"
        assert mappings[0]["system"] == "SNOMED"
        assert mappings[0]["code"] == "123"

    def test_convert_trial_mcode_to_mappings_format_dict(self, workflow):
        """Test converting mCODE elements dict to mappings format."""
        mcode_elements = {
            "primaryOutcome": {"display": "Overall Survival", "system": "SNOMED", "code": "456"}
        }

        mappings = workflow._convert_trial_mcode_to_mappings_format(mcode_elements)

        assert len(mappings) == 1
        assert mappings[0]["mcode_element"] == "primaryOutcome"
        assert mappings[0]["value"] == "Overall Survival"
        assert mappings[0]["system"] == "SNOMED"
        assert mappings[0]["code"] == "456"

    def test_clear_workflow_caches(self, workflow):
        """Test clearing workflow caches."""
        # Should not raise any exceptions
        workflow.clear_workflow_caches()

    def test_get_cache_stats(self, workflow):
        """Test getting cache statistics."""
        stats = workflow.get_cache_stats()
        assert isinstance(stats, dict)
