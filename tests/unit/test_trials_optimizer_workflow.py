"""
Tests for TrialsOptimizerWorkflow.
"""

import pytest
from unittest.mock import Mock, patch

from src.workflows.trials_optimizer import TrialsOptimizerWorkflow
from src.utils.config import Config


class TestTrialsOptimizerWorkflow:
    """Test cases for TrialsOptimizerWorkflow."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return Config()

    @pytest.fixture
    def workflow(self):
        """Create a test workflow instance."""
        return TrialsOptimizerWorkflow()

    @pytest.fixture
    def mock_trials_data(self):
        """Create mock trial data for testing."""
        return [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT00123456",
                        "briefTitle": "Mock Breast Cancer Trial 1",
                    }
                }
            },
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT00234567",
                        "briefTitle": "Mock Breast Cancer Trial 2",
                    }
                }
            },
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT00345678",
                        "briefTitle": "Mock Breast Cancer Trial 3",
                    }
                }
            },
        ]

    def test_workflow_initialization(self, workflow):
        """Test workflow initializes correctly."""
        assert workflow is not None
        assert workflow.memory_space == "optimization"

    def test_generate_combinations(self, workflow):
        """Test combination generation."""
        prompts = ["prompt1", "prompt2"]
        models = ["model1", "model2"]
        max_combinations = 3

        combinations = workflow.cross_validator.generate_combinations(
            prompts, models, max_combinations
        )

        assert len(combinations) == 3  # Limited by max_combinations
        assert all("prompt" in combo and "model" in combo for combo in combinations)

    def test_generate_combinations_full(self, workflow):
        """Test combination generation without limit."""
        prompts = ["prompt1", "prompt2"]
        models = ["model1"]
        max_combinations = 10

        combinations = workflow.cross_validator.generate_combinations(
            prompts, models, max_combinations
        )

        assert len(combinations) == 2  # All combinations
        expected_combinations = [
            {"prompt": "prompt1", "model": "model1"},
            {"prompt": "prompt2", "model": "model1"},
        ]
        assert combinations == expected_combinations


    def test_create_kfold_splits(self, workflow, mock_trials_data):
        """Test k-fold split creation via execution manager."""
        n_samples = len(mock_trials_data)  # 3 trials
        n_folds = 3

        folds = workflow.execution_manager._create_kfold_splits(n_samples, n_folds)

        assert len(folds) == n_folds
        # All samples should be distributed
        total_samples = sum(len(fold) for fold in folds)
        assert total_samples == n_samples
        # Each fold should have samples
        for fold in folds:
            assert len(fold) > 0

    def test_validate_combination(self, workflow):
        """Test combination validation."""
        # Valid combination
        assert (
            workflow.validate_combination(
                "direct_mcode_evidence_based_concise", "deepseek-coder"
            )
            is True
        )

        # Invalid prompt
        assert (
            workflow.validate_combination("invalid_prompt", "deepseek-coder") is False
        )

        # Invalid model
        assert (
            workflow.validate_combination(
                "direct_mcode_evidence_based_concise", "invalid_model"
            )
            is False
        )

    def test_get_available_prompts(self, workflow):
        """Test getting available prompts."""
        prompts = workflow.get_available_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        assert "direct_mcode_evidence_based_concise" in prompts

    def test_get_available_models(self, workflow):
        """Test getting available models."""
        models = workflow.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "deepseek-coder" in models

    @patch("src.pipeline.McodePipeline")
    def test_execute_success(
        self, mock_pipeline_class, workflow, mock_trials_data
    ):
        """Test successful workflow execution."""
        # Mock pipeline for successful processing
        from unittest.mock import AsyncMock

        mock_pipeline = AsyncMock()
        mock_result = Mock()
        mock_result.mcode_mappings = [
            {
                "element_type": "CancerCondition",
                "display": "Breast Cancer",
                "confidence_score": 0.9,
            },
            {
                "element_type": "CancerTreatment",
                "display": "Chemotherapy",
                "confidence_score": 0.8,
            },
            {
                "element_type": "PatientDemographics",
                "display": "Female",
                "confidence_score": 0.7,
            },
            {"element_type": "TNMStage", "display": "T2N1M0", "confidence_score": 0.6},
            {
                "element_type": "GeneticMarker",
                "display": "HER2+",
                "confidence_score": 0.8,
            },
            {
                "element_type": "CancerCondition",
                "display": "Metastatic",
                "confidence_score": 0.7,
            },
            {
                "element_type": "CancerTreatment",
                "display": "Targeted Therapy",
                "confidence_score": 0.9,
            },
            {
                "element_type": "PatientDemographics",
                "display": "Age 45",
                "confidence_score": 0.8,
            },
        ]
        mock_result.validation_results = Mock(compliance_score=0.85)
        mock_pipeline.process.return_value = mock_result
        mock_pipeline_class.return_value = mock_pipeline

        result = workflow.execute(
            trials_data=mock_trials_data,
            cv_folds=3,
            prompts=["direct_mcode_evidence_based_concise"],
            models=["deepseek-coder"],
            max_combinations=1,
        )

        assert result.success is True
        assert "total_combinations_tested" in result.metadata
        assert "cv_folds" in result.metadata
        assert result.metadata["cv_folds"] == 3

    def test_execute_no_trials_data(self, workflow):
        """Test execution fails with no trial data."""
        result = workflow.execute(trials_data=[], cv_folds=3)

        assert result.success is False
        assert "No trial data provided" in result.error_message

    @patch("src.pipeline.McodePipeline")
    @patch(
        "src.workflows.trials_optimizer_workflow.TrialsOptimizerWorkflow._set_default_llm_spec"
    )
    def test_execute_calls_set_default_spec(
        self, mock_set_default, mock_pipeline_class, workflow, mock_trials_data
    ):
        """Test that execute calls set default LLM spec on success."""
        # Mock successful pipeline with good score
        from unittest.mock import AsyncMock

        mock_pipeline = AsyncMock()
        mock_result = Mock()
        # Create proper McodeElement objects
        from src.shared.models import McodeElement

        mock_mappings = [
            McodeElement(
                element_type="CancerCondition",
                code="C50",
                display="Breast Cancer",
                system="ICD-10",
                confidence_score=0.9,
                evidence_text="Patient has breast cancer",
            ),
            McodeElement(
                element_type="CancerTreatment",
                code="CHEMO",
                display="Chemotherapy",
                system="Treatment",
                confidence_score=0.8,
                evidence_text="Patient receives chemotherapy",
            ),
            McodeElement(
                element_type="PatientDemographics",
                code="FEMALE",
                display="Female",
                system="Demographics",
                confidence_score=0.7,
                evidence_text="Patient is female",
            ),
            McodeElement(
                element_type="TNMStage",
                code="T2N1M0",
                display="T2N1M0",
                system="TNM",
                confidence_score=0.6,
                evidence_text="Tumor stage T2N1M0",
            ),
            McodeElement(
                element_type="GeneticMarker",
                code="HER2",
                display="HER2+",
                system="Biomarker",
                confidence_score=0.8,
                evidence_text="HER2 positive",
            ),
            McodeElement(
                element_type="CancerCondition",
                code="C78",
                display="Metastatic",
                system="ICD-10",
                confidence_score=0.7,
                evidence_text="Metastatic disease",
            ),
            McodeElement(
                element_type="CancerTreatment",
                code="TARGETED",
                display="Targeted Therapy",
                system="Treatment",
                confidence_score=0.9,
                evidence_text="Targeted therapy prescribed",
            ),
            McodeElement(
                element_type="PatientDemographics",
                code="AGE45",
                display="Age 45",
                system="Demographics",
                confidence_score=0.8,
                evidence_text="Patient age 45",
            ),
            McodeElement(
                element_type="CancerCondition",
                code="C50.9",
                display="Invasive Ductal",
                system="ICD-10",
                confidence_score=0.8,
                evidence_text="Invasive ductal carcinoma",
            ),
            McodeElement(
                element_type="CancerTreatment",
                code="SURGERY",
                display="Surgery",
                system="Treatment",
                confidence_score=0.9,
                evidence_text="Surgical intervention",
            ),
        ]
        mock_result.mcode_mappings = mock_mappings
        mock_result.validation_results = Mock(compliance_score=0.9)
        mock_result.source_references = [Mock()] * 5
        mock_pipeline.process.return_value = mock_result
        mock_pipeline_class.return_value = mock_pipeline

        workflow.execute(
            trials_data=mock_trials_data,
            cv_folds=3,
            prompts=["direct_mcode_evidence_based_concise"],
            models=["deepseek-coder"],
            max_combinations=1,
        )

        # Should have called set default spec since score > 0
        mock_set_default.assert_called_once()

    def test_cv_folds_adjustment(self, workflow, mock_trials_data):
        """Test CV folds adjustment when more folds than trials."""
        trials_count = len(mock_trials_data)  # 3 trials

        with patch(
            "src.pipeline.McodePipeline"
        ) as mock_pipeline_class:
            from unittest.mock import AsyncMock

            mock_pipeline = AsyncMock()
            mock_result = Mock()
            mock_result.mcode_mappings = [
                {
                    "element_type": "CancerCondition",
                    "display": "Breast Cancer",
                    "confidence_score": 0.9,
                },
                {
                    "element_type": "CancerTreatment",
                    "display": "Chemotherapy",
                    "confidence_score": 0.8,
                },
                {
                    "element_type": "PatientDemographics",
                    "display": "Female",
                    "confidence_score": 0.7,
                },
                {
                    "element_type": "TNMStage",
                    "display": "T2N1M0",
                    "confidence_score": 0.6,
                },
                {
                    "element_type": "GeneticMarker",
                    "display": "HER2+",
                    "confidence_score": 0.8,
                },
            ]
            mock_result.validation_results = Mock(compliance_score=0.8)
            mock_pipeline.process.return_value = mock_result
            mock_pipeline_class.return_value = mock_pipeline

            result = workflow.execute(
                trials_data=mock_trials_data,
                cv_folds=5,  # More folds than trials
                prompts=["direct_mcode_evidence_based_concise"],
                models=["deepseek-coder"],
                max_combinations=1,
            )

            # Should adjust folds to match trial count
            assert result.metadata["cv_folds"] == trials_count
