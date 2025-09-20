#!/usr/bin/env python3
"""
End-to-End Tests for Researcher Workflow

Tests the complete researcher workflow from trial discovery to summary generation:
1. Trial Discovery (trials_fetcher) → 2. Analysis (trials_processor) → 3. Summary Generation (trials_summarizer)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cli.trials_fetcher import main as trials_fetcher_main
from src.cli.trials_processor import main as trials_processor_main
from src.cli.trials_summarizer import main as trials_summarizer_main
from src.shared.models import WorkflowResult


class TestResearcherWorkflowE2E:
    """End-to-end tests for the complete researcher workflow."""

    @pytest.fixture
    def sample_trial_data(self):
        """Sample trial data for testing."""
        return {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "briefTitle": "Phase 2 Study of Test Drug in Breast Cancer",
                    "officialTitle": "A Phase 2 Study of Test Drug in Patients with Breast Cancer"
                },
                "statusModule": {
                    "overallStatus": "Recruiting"
                },
                "designModule": {
                    "phases": ["Phase 2"]
                },
                "conditionsModule": {
                    "conditions": [
                        {
                            "name": "Breast Cancer",
                            "code": "254837009",
                            "codeSystem": "http://snomed.info/sct"
                        }
                    ]
                },
                "eligibilityModule": {
                    "eligibilityCriteria": "Inclusion Criteria:\n- Age >= 18 years\n- Histologically confirmed breast cancer",
                    "minimumAge": "18 Years",
                    "sex": "All",
                    "healthyVolunteers": False
                },
                "armsInterventionsModule": {
                    "interventions": [
                        {
                            "type": "Drug",
                            "name": "Test Drug",
                            "description": "Experimental targeted therapy"
                        }
                    ]
                }
            }
        }

    @pytest.fixture
    def mock_api_response(self, sample_trial_data):
        """Mock API response for ClinicalTrials.gov."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_trial_data
        return mock_response

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for processing and summarization."""
        mock_service = MagicMock()

        # Mock processing response
        mock_processing_result = MagicMock()
        mock_processing_result.McodeResults = {
            "mcode_mappings": [
                {
                    "element_type": "PrimaryCancerCondition",
                    "confidence": 0.95,
                    "evidence": "Trial conditions include breast cancer"
                },
                {
                    "element_type": "CancerStage",
                    "confidence": 0.90,
                    "evidence": "Phase 2 trial for breast cancer"
                }
            ]
        }
        mock_processing_result.__dict__ = {
            "McodeResults": mock_processing_result.McodeResults,
            "protocolSection": {
                "identificationModule": {"nctId": "NCT12345678"}
            }
        }

        # Mock summarization response
        mock_summary_result = MagicMock()
        mock_summary_result.McodeResults = {
            "natural_language_summary": "This Phase 2 clinical trial evaluates Test Drug in patients with breast cancer. The study includes patients aged 18 and older with histologically confirmed breast cancer. Participants receive experimental targeted therapy as intervention.",
            "mcode_mappings": mock_processing_result.McodeResults["mcode_mappings"]
        }
        mock_summary_result.__dict__ = {
            "McodeResults": mock_summary_result.McodeResults,
            "protocolSection": {
                "identificationModule": {"nctId": "NCT12345678"}
            }
        }

        # Configure mock service
        async def mock_process_trial(trial_data, **kwargs):
            return mock_processing_result

        async def mock_summarize_trial(trial_data, **kwargs):
            return mock_summary_result

        mock_service.process_trial = mock_process_trial
        mock_service.summarize_trial = mock_summarize_trial

        return mock_service

    @pytest.fixture
    def mock_memory_storage(self):
        """Mock CORE Memory storage."""
        mock_storage = MagicMock()
        mock_storage.store_trial_data.return_value = True
        mock_storage.store_processed_trial.return_value = True
        mock_storage.store_trial_summary.return_value = True
        return mock_storage

    @patch('src.utils.fetcher.requests.get')
    def test_researcher_workflow_trial_discovery(self, mock_get, sample_trial_data, mock_api_response, tmp_path):
        """Test the trial discovery phase of researcher workflow."""
        # Setup API mock
        mock_get.return_value = mock_api_response

        # Create temporary output file
        output_file = tmp_path / "trials.ndjson"

        # Test CLI execution
        import argparse
        args = argparse.Namespace(
            condition=None,
            nct_id="NCT12345678",
            nct_ids=None,
            limit=10,
            output_file=str(output_file),
            verbose=False,
            log_level="INFO",
            config=None
        )

        # Capture stdout since CLI prints to stdout
        import io
        import sys
        from contextlib import redirect_stdout

        stdout_capture = io.StringIO()
        with redirect_stdout(stdout_capture):
            trials_fetcher_main(args)

        # Verify output file was created
        assert output_file.exists()

        # Verify output contains expected data
        with open(output_file, 'r') as f:
            content = f.read().strip()
            assert content
            trial_data = json.loads(content)
            assert trial_data["protocolSection"]["identificationModule"]["nctId"] == "NCT12345678"

        # Verify success message in stdout
        output = stdout_capture.getvalue()
        assert "✅ Trials fetch completed successfully!" in output
        assert "Total trials fetched: 1" in output

    @patch('src.pipeline.llm_service.LLMService')
    @patch('src.storage.mcode_memory_storage.McodeMemoryStorage')
    def test_researcher_workflow_trial_analysis(self, mock_memory_storage_class, mock_llm_service_class, sample_trial_data, tmp_path):
        """Test the trial analysis phase of researcher workflow."""
        from src.shared.models import McodeElement

        # Setup mocks
        mock_memory = MagicMock()
        mock_memory.store_processed_trial.return_value = True
        mock_memory_storage_class.return_value = mock_memory

        mock_llm = MagicMock()
        # Mock the async map_to_mcode method to return McodeElement instances
        async def mock_map_to_mcode(text):
            return [
                McodeElement(
                    element_type="PrimaryCancerCondition",
                    code="254837009",
                    system="http://snomed.info/sct",
                    display="Breast Cancer",
                    confidence_score=0.95,
                    evidence_text="Trial conditions include breast cancer"
                )
            ]
        mock_llm.map_to_mcode = mock_map_to_mcode
        mock_llm_service_class.return_value = mock_llm

        # Create input file with trial data
        input_file = tmp_path / "input_trials.ndjson"
        with open(input_file, 'w') as f:
            json.dump(sample_trial_data, f)
            f.write('\n')

        # Create output file
        output_file = tmp_path / "mcode_trials.ndjson"

        # Test CLI execution
        import argparse
        args = argparse.Namespace(
            input_file=str(input_file),
            output_file=str(output_file),
            ingest=True,
            memory_source="test",
            model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise",
            workers=1,
            verbose=False,
            log_level="INFO",
            config=None
        )

        trials_processor_main(args)

        # Verify output file was created
        assert output_file.exists()

        # Verify output contains expected mCODE data
        with open(output_file, 'r') as f:
            content = f.read().strip()
            assert content
            mcode_data = json.loads(content)
            assert mcode_data["trial_id"] == "NCT12345678"
            assert "mcode_elements" in mcode_data
            assert "mcode_mappings" in mcode_data["mcode_elements"]
            assert len(mcode_data["mcode_elements"]["mcode_mappings"]) > 0

    @patch('src.services.summarizer.McodeSummarizer')
    @patch('src.storage.mcode_memory_storage.McodeMemoryStorage')
    def test_researcher_workflow_summary_generation(self, mock_memory_storage_class, mock_summarizer_class, sample_trial_data, tmp_path):
        """Test the summary generation phase of researcher workflow."""
        # Setup mocks
        mock_memory = MagicMock()
        mock_memory.store_trial_summary.return_value = True
        mock_memory_storage_class.return_value = mock_memory

        mock_summarizer = MagicMock()
        mock_summarizer.create_trial_summary.return_value = "This Phase 2 clinical trial evaluates Test Drug in patients with breast cancer. The study includes patients aged 18 and older with histologically confirmed breast cancer."
        mock_summarizer_class.return_value = mock_summarizer

        # Create input file with mCODE trial data
        input_file = tmp_path / "mcode_input.ndjson"
        mcode_trial_data = {
            "trial_id": "NCT12345678",
            "mcode_elements": {
                "mcode_mappings": [
                    {
                        "element_type": "PrimaryCancerCondition",
                        "confidence": 0.95,
                        "evidence": "Trial conditions include breast cancer"
                    }
                ]
            },
            "original_trial_data": sample_trial_data
        }
        with open(input_file, 'w') as f:
            json.dump(mcode_trial_data, f)
            f.write('\n')

        # Create output file
        output_file = tmp_path / "summaries.ndjson"

        # Test CLI execution
        import argparse
        args = argparse.Namespace(
            input_file=str(input_file),
            output_file=str(output_file),
            ingest=True,
            memory_source="test",
            model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise",
            workers=1,
            verbose=False,
            log_level="INFO",
            config=None
        )

        trials_summarizer_main(args)

        # Verify output file was created
        assert output_file.exists()

        # Verify output contains expected summary data
        with open(output_file, 'r') as f:
            content = f.read().strip()
            assert content
            summary_data = json.loads(content)
            assert summary_data["trial_id"] == "NCT12345678"
            assert "summary" in summary_data
            assert "mcode_elements" in summary_data

    @patch('src.services.summarizer.McodeSummarizer')
    @patch('src.pipeline.llm_service.LLMService')
    @patch('src.storage.mcode_memory_storage.McodeMemoryStorage')
    @patch('src.utils.fetcher.requests.get')
    def test_complete_researcher_workflow_integration(self, mock_get, mock_memory_storage_class,
                                                     mock_llm_service_class, mock_summarizer_class,
                                                     sample_trial_data, mock_api_response, tmp_path):
        """Test the complete end-to-end researcher workflow integration."""
        from src.shared.models import McodeElement

        # Setup API mock
        mock_get.return_value = mock_api_response

        # Setup storage mocks
        mock_memory = MagicMock()
        mock_memory.store_processed_trial.return_value = True
        mock_memory.store_trial_summary.return_value = True
        mock_memory_storage_class.return_value = mock_memory

        # Setup LLM service mock
        mock_llm = MagicMock()
        async def mock_map_to_mcode(text):
            return [
                McodeElement(
                    element_type="PrimaryCancerCondition",
                    code="254837009",
                    system="http://snomed.info/sct",
                    display="Breast Cancer",
                    confidence_score=0.95,
                    evidence_text="Trial conditions include breast cancer"
                )
            ]
        mock_llm.map_to_mcode = mock_map_to_mcode
        mock_llm_service_class.return_value = mock_llm

        # Setup summarizer mock
        mock_summarizer = MagicMock()
        mock_summarizer.create_trial_summary.return_value = "This Phase 2 clinical trial evaluates Test Drug in patients with breast cancer."
        mock_summarizer_class.return_value = mock_summarizer

        # Create temporary files
        trials_file = tmp_path / "trials.ndjson"
        mcode_file = tmp_path / "mcode_trials.ndjson"
        summary_file = tmp_path / "summaries.ndjson"

        # Step 1: Fetch trials
        import argparse
        fetch_args = argparse.Namespace(
            condition=None, nct_id="NCT12345678", nct_ids=None, limit=10,
            output_file=str(trials_file), verbose=False, log_level="INFO", config=None
        )

        import io
        import sys
        from contextlib import redirect_stdout
        stdout_capture = io.StringIO()
        with redirect_stdout(stdout_capture):
            trials_fetcher_main(fetch_args)

        assert trials_file.exists()

        # Step 2: Process trials
        process_args = argparse.Namespace(
            input_file=str(trials_file), output_file=str(mcode_file), ingest=True,
            memory_source="test", model="deepseek-coder", prompt="direct_mcode_evidence_based_concise",
            workers=1, verbose=False, log_level="INFO", config=None
        )

        trials_processor_main(process_args)
        assert mcode_file.exists()

        # Step 3: Generate summaries
        summary_args = argparse.Namespace(
            input_file=str(mcode_file), output_file=str(summary_file), ingest=True,
            memory_source="test", model="deepseek-coder", prompt="direct_mcode_evidence_based_concise",
            workers=1, verbose=False, log_level="INFO", config=None
        )

        trials_summarizer_main(summary_args)
        assert summary_file.exists()

        # Verify data flow between steps
        with open(summary_file, 'r') as f:
            final_output = json.loads(f.read().strip())
            assert final_output["trial_id"] == "NCT12345678"
            assert "summary" in final_output
            assert "mcode_elements" in final_output

    def test_researcher_workflow_error_handling(self, tmp_path):
        """Test error handling in researcher workflow."""
        # Test with non-existent input file for processor
        nonexistent_file = tmp_path / "nonexistent.ndjson"

        import argparse
        args = argparse.Namespace(
            input_file=str(nonexistent_file), output_file=None, ingest=False,
            memory_source="test", model="gpt-4", prompt="direct_mcode_evidence_based_concise",
            workers=1, verbose=False, log_level="INFO", config=None
        )

        with pytest.raises(SystemExit):
            trials_processor_main(args)

    @patch('src.cli.trials_fetcher.TrialsFetcherWorkflow')
    def test_researcher_workflow_invalid_nct_id(self, mock_workflow_class):
        """Test handling of invalid NCT ID in trial discovery."""
        # Setup mock workflow to return failure
        mock_workflow = MagicMock()
        mock_workflow.execute.return_value = WorkflowResult(
            success=False,
            error_message="Invalid NCT ID format",
            metadata={"fetch_type": "single_nct"}
        )
        mock_workflow_class.return_value = mock_workflow

        import argparse
        args = argparse.Namespace(
            condition=None, nct_id="INVALID", nct_ids=None, limit=10,
            output_file=None, verbose=False, log_level="INFO", config=None
        )

        import io
        import sys
        from contextlib import redirect_stdout
        stdout_capture = io.StringIO()
        with redirect_stdout(stdout_capture):
            with pytest.raises(SystemExit) as exc_info:
                trials_fetcher_main(args)

        # Verify exit code is 1 (error)
        assert exc_info.value.code == 1

        # Verify error was handled (CLI should exit with error)
        output = stdout_capture.getvalue()
        assert "❌ Trials fetch failed" in output