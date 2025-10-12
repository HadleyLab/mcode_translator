"""
Unit tests for TrialExtractor class.
"""

from unittest.mock import patch

import pytest

from src.workflows.trial_extractor import TrialExtractor


class TestTrialExtractor:
    """Test cases for TrialExtractor class."""

    @pytest.fixture
    def trial_extractor(self):
        """Create TrialExtractor instance."""
        return TrialExtractor()

    @pytest.fixture
    def sample_trial_data(self):
        """Sample trial data for testing."""
        return {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "briefTitle": "Test Trial",
                    "officialTitle": "Official Test Trial Title",
                },
                "eligibilityModule": {
                    "minimumAge": "18 Years",
                    "maximumAge": "65 Years",
                    "sex": "All",
                    "healthyVolunteers": True,
                    "eligibilityCriteria": "Detailed eligibility criteria text",
                },
                "conditionsModule": {
                    "conditions": [
                        {"name": "Breast Cancer", "code": "C50"},
                        {"name": "Diabetes", "code": "E10"},
                        "Lung Cancer",
                    ]
                },
                "armsInterventionsModule": {
                    "interventions": [
                        {
                            "type": "Drug",
                            "name": "Test Drug",
                            "description": "Test drug description",
                        },
                        {
                            "type": "Procedure",
                            "name": "Surgery",
                            "description": "Surgical procedure",
                        },
                    ]
                },
                "designModule": {
                    "studyType": "Interventional",
                    "phases": ["Phase 1", "Phase 2"],
                    "primaryPurpose": "Treatment",
                    "enrollmentInfo": {"count": 100, "type": "Actual"},
                },
                "statusModule": {
                    "overallStatus": "Completed",
                    "startDateStruct": {"date": "2020-01-01"},
                    "completionDateStruct": {"date": "2022-01-01"},
                    "primaryCompletionDateStruct": {"date": "2021-12-31"},
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Test Sponsor", "class": "Industry"},
                    "responsibleParty": {
                        "name": "Test Investigator",
                        "type": "Principal Investigator",
                        "affiliation": "Test University",
                    },
                },
            }
        }

    def test_extract_trial_mcode_elements_success(self, trial_extractor, sample_trial_data):
        """Test successful extraction of mCODE elements."""
        result = trial_extractor.extract_trial_mcode_elements(sample_trial_data)

        assert isinstance(result, dict)
        assert "TrialIdentifier" in result
        assert "TrialAgeCriteria" in result
        assert "TrialCancerConditions" in result
        assert "TrialMedicationInterventions" in result
        assert "TrialStudyType" in result
        assert "TrialStatus" in result
        assert "TrialLeadSponsor" in result

    def test_extract_trial_mcode_elements_empty_input(self, trial_extractor):
        """Test extraction with empty input."""
        result = trial_extractor.extract_trial_mcode_elements({})
        assert result == {}

    def test_extract_trial_mcode_elements_invalid_protocol_section(self, trial_extractor):
        """Test extraction with invalid protocol section."""
        result = trial_extractor.extract_trial_mcode_elements({"protocolSection": "invalid"})
        assert result == {}

    def test_extract_trial_mcode_elements_exception_handling(self, trial_extractor):
        """Test exception handling during extraction."""
        with patch.object(
            trial_extractor, "_extract_trial_identification", side_effect=Exception("Test error")
        ):
            result = trial_extractor.extract_trial_mcode_elements({"protocolSection": {}})
            # Should return empty dict on exception
            assert result == {}

    def test_extract_trial_identification_complete(self, trial_extractor, sample_trial_data):
        """Test extraction of trial identification with all fields."""
        identification = sample_trial_data["protocolSection"]["identificationModule"]
        result = trial_extractor._extract_trial_identification(identification)

        assert result["TrialIdentifier"]["code"] == "NCT12345678"
        assert result["TrialTitle"]["display"] == "Test Trial"
        assert result["TrialOfficialTitle"]["display"] == "Official Test Trial Title"

    def test_extract_trial_identification_partial(self, trial_extractor):
        """Test extraction of trial identification with partial data."""
        identification = {"nctId": "NCT99999999"}
        result = trial_extractor._extract_trial_identification(identification)

        assert result["TrialIdentifier"]["code"] == "NCT99999999"
        assert "TrialTitle" not in result
        assert "TrialOfficialTitle" not in result

    def test_extract_trial_identification_empty(self, trial_extractor):
        """Test extraction of trial identification with empty data."""
        result = trial_extractor._extract_trial_identification({})
        assert result == {}

    def test_extract_trial_eligibility_mcode_complete(self, trial_extractor, sample_trial_data):
        """Test extraction of eligibility criteria with all fields."""
        eligibility = sample_trial_data["protocolSection"]["eligibilityModule"]
        result = trial_extractor._extract_trial_eligibility_mcode(eligibility)

        assert result["TrialAgeCriteria"]["minimumAge"] == "18 Years"
        assert result["TrialAgeCriteria"]["maximumAge"] == "65 Years"
        assert result["TrialSexCriteria"]["code"] == "all"
        assert result["TrialHealthyVolunteers"]["allowed"] is True
        assert "eligibility criteria text" in result["TrialEligibilityCriteria"]["text"]

    def test_extract_trial_eligibility_mcode_partial(self, trial_extractor):
        """Test extraction of eligibility criteria with partial data."""
        eligibility = {"minimumAge": "21 Years", "sex": "Female"}
        result = trial_extractor._extract_trial_eligibility_mcode(eligibility)

        assert result["TrialAgeCriteria"]["minimumAge"] == "21 Years"
        assert result["TrialSexCriteria"]["code"] == "female"
        assert "TrialHealthyVolunteers" not in result
        assert "TrialEligibilityCriteria" not in result

    def test_extract_trial_eligibility_mcode_invalid_input(self, trial_extractor):
        """Test extraction of eligibility criteria with invalid input."""
        result = trial_extractor._extract_trial_eligibility_mcode("invalid")
        assert result == {}

    def test_extract_trial_conditions_mcode_cancer_conditions(
        self, trial_extractor, sample_trial_data
    ):
        """Test extraction of cancer conditions."""
        conditions = sample_trial_data["protocolSection"]["conditionsModule"]
        result = trial_extractor._extract_trial_conditions_mcode(conditions)

        assert "TrialCancerConditions" in result
        assert len(result["TrialCancerConditions"]) == 2  # Breast Cancer and Lung Cancer
        assert result["TrialCancerConditions"][0]["display"] == "breast cancer"
        assert result["TrialCancerConditions"][1]["display"] == "lung cancer"

        assert "TrialComorbidConditions" in result
        assert len(result["TrialComorbidConditions"]) == 1
        assert result["TrialComorbidConditions"][0]["display"] == "diabetes"

    def test_extract_trial_conditions_mcode_string_conditions(self, trial_extractor):
        """Test extraction of conditions when they are strings."""
        conditions = {"conditions": ["Hypertension", "cancer"]}
        result = trial_extractor._extract_trial_conditions_mcode(conditions)

        assert "TrialCancerConditions" in result
        assert len(result["TrialCancerConditions"]) == 1
        assert result["TrialCancerConditions"][0]["display"] == "cancer"

        assert "TrialComorbidConditions" in result
        assert len(result["TrialComorbidConditions"]) == 1
        assert result["TrialComorbidConditions"][0]["display"] == "hypertension"

    def test_extract_trial_conditions_mcode_invalid_input(self, trial_extractor):
        """Test extraction of conditions with invalid input."""
        result = trial_extractor._extract_trial_conditions_mcode("invalid")
        assert result == {}

    def test_extract_trial_conditions_mcode_empty_conditions(self, trial_extractor):
        """Test extraction of conditions with empty conditions list."""
        conditions = {"conditions": []}
        result = trial_extractor._extract_trial_conditions_mcode(conditions)
        assert result == {}

    def test_extract_trial_interventions_mcode_medication_and_other(
        self, trial_extractor, sample_trial_data
    ):
        """Test extraction of interventions with medication and other types."""
        interventions = sample_trial_data["protocolSection"]["armsInterventionsModule"]
        result = trial_extractor._extract_trial_interventions_mcode(interventions)

        assert "TrialMedicationInterventions" in result
        assert len(result["TrialMedicationInterventions"]) == 1
        assert result["TrialMedicationInterventions"][0]["display"] == "Test Drug"
        assert result["TrialMedicationInterventions"][0]["interventionType"] == "drug"

        assert "TrialOtherInterventions" in result
        assert len(result["TrialOtherInterventions"]) == 1
        assert result["TrialOtherInterventions"][0]["display"] == "Surgery"
        assert result["TrialOtherInterventions"][0]["interventionType"] == "procedure"

    def test_extract_trial_interventions_mcode_invalid_input(self, trial_extractor):
        """Test extraction of interventions with invalid input."""
        result = trial_extractor._extract_trial_interventions_mcode("invalid")
        assert result == {}

    def test_extract_trial_interventions_mcode_invalid_intervention(self, trial_extractor):
        """Test extraction of interventions with invalid intervention data."""
        interventions = {"interventions": ["invalid"]}
        result = trial_extractor._extract_trial_interventions_mcode(interventions)
        assert result == {}

    def test_extract_trial_design_mcode_complete(self, trial_extractor, sample_trial_data):
        """Test extraction of trial design with all fields."""
        design = sample_trial_data["protocolSection"]["designModule"]
        result = trial_extractor._extract_trial_design_mcode(design)

        assert result["TrialStudyType"]["display"] == "Interventional"
        assert result["TrialPhase"]["display"] == "Phase 1, Phase 2"
        assert result["TrialPrimaryPurpose"]["display"] == "Treatment"
        assert result["TrialEnrollment"]["count"] == 100
        assert result["TrialEnrollment"]["type"] == "Actual"

    def test_extract_trial_design_mcode_partial(self, trial_extractor):
        """Test extraction of trial design with partial data."""
        design = {"studyType": "Observational"}
        result = trial_extractor._extract_trial_design_mcode(design)

        assert result["TrialStudyType"]["display"] == "Observational"
        assert "TrialPhase" not in result
        assert "TrialPrimaryPurpose" not in result
        assert "TrialEnrollment" not in result

    def test_extract_trial_design_mcode_invalid_input(self, trial_extractor):
        """Test extraction of trial design with invalid input."""
        result = trial_extractor._extract_trial_design_mcode("invalid")
        assert result == {}

    def test_extract_trial_temporal_mcode_complete(self, trial_extractor, sample_trial_data):
        """Test extraction of temporal information with all fields."""
        status = sample_trial_data["protocolSection"]["statusModule"]
        result = trial_extractor._extract_trial_temporal_mcode(status)

        assert result["TrialStatus"]["display"] == "Completed"
        assert result["TrialStartDate"]["date"] == "2020-01-01"
        assert result["TrialCompletionDate"]["date"] == "2022-01-01"
        assert result["TrialPrimaryCompletionDate"]["date"] == "2021-12-31"

    def test_extract_trial_temporal_mcode_partial(self, trial_extractor):
        """Test extraction of temporal information with partial data."""
        status = {"overallStatus": "Recruiting"}
        result = trial_extractor._extract_trial_temporal_mcode(status)

        assert result["TrialStatus"]["display"] == "Recruiting"
        assert "TrialStartDate" not in result
        assert "TrialCompletionDate" not in result

    def test_extract_trial_temporal_mcode_invalid_input(self, trial_extractor):
        """Test extraction of temporal information with invalid input."""
        result = trial_extractor._extract_trial_temporal_mcode("invalid")
        assert result == {}

    def test_extract_trial_sponsor_mcode_complete(self, trial_extractor, sample_trial_data):
        """Test extraction of sponsor information with all fields."""
        sponsor = sample_trial_data["protocolSection"]["sponsorCollaboratorsModule"]
        result = trial_extractor._extract_trial_sponsor_mcode(sponsor)

        assert result["TrialLeadSponsor"]["name"] == "Test Sponsor"
        assert result["TrialLeadSponsor"]["class"] == "Industry"
        assert result["TrialResponsibleParty"]["name"] == "Test Investigator"
        assert result["TrialResponsibleParty"]["type"] == "Principal Investigator"
        assert result["TrialResponsibleParty"]["affiliation"] == "Test University"

    def test_extract_trial_sponsor_mcode_partial(self, trial_extractor):
        """Test extraction of sponsor information with partial data."""
        sponsor = {"leadSponsor": {"name": "Test Sponsor"}}
        result = trial_extractor._extract_trial_sponsor_mcode(sponsor)

        assert result["TrialLeadSponsor"]["name"] == "Test Sponsor"
        assert "TrialResponsibleParty" not in result

    def test_extract_trial_sponsor_mcode_invalid_input(self, trial_extractor):
        """Test extraction of sponsor information with invalid input."""
        result = trial_extractor._extract_trial_sponsor_mcode("invalid")
        assert result == {}

    def test_extract_trial_metadata_complete(self, trial_extractor, sample_trial_data):
        """Test extraction of trial metadata with complete data."""
        result = trial_extractor.extract_trial_metadata(sample_trial_data)

        assert result["nct_id"] == "NCT12345678"
        assert result["brief_title"] == "Test Trial"
        assert result["official_title"] == "Official Test Trial Title"
        assert result["overall_status"] == "Completed"
        assert result["start_date"] == "2020-01-01"
        assert result["completion_date"] == "2022-01-01"
        assert result["study_type"] == "Interventional"
        assert result["phase"] == ["Phase 1", "Phase 2"]
        assert result["primary_purpose"] == "Treatment"
        assert result["minimum_age"] == "18 Years"
        assert result["maximum_age"] == "65 Years"
        assert result["sex"] == "All"
        assert result["healthy_volunteers"] is True
        assert "Breast Cancer" in result["conditions"]
        assert "Test Drug" in result["interventions"]

    def test_extract_trial_metadata_invalid_input(self, trial_extractor):
        """Test extraction of trial metadata with invalid input."""
        result = trial_extractor.extract_trial_metadata("invalid")
        assert result == {}

    def test_extract_trial_metadata_exception_handling(self, trial_extractor):
        """Test exception handling in metadata extraction."""
        with patch("builtins.print") as mock_print:
            # Create data that will cause an exception
            trial_data = {"protocolSection": None}
            result = trial_extractor.extract_trial_metadata(trial_data)
            assert result == {}
            mock_print.assert_called()

    def test_extract_trial_id_success(self, trial_extractor, sample_trial_data):
        """Test successful extraction of trial ID."""
        result = trial_extractor.extract_trial_id(sample_trial_data)
        assert result == "NCT12345678"

    def test_extract_trial_id_missing_data(self, trial_extractor):
        """Test extraction of trial ID with missing data."""
        result = trial_extractor.extract_trial_id({})
        assert result.startswith("unknown_trial_")
        assert len(result) == 22  # "unknown_trial_" + 8 char hash

    def test_extract_trial_id_invalid_structure(self, trial_extractor):
        """Test extraction of trial ID with invalid structure."""
        result = trial_extractor.extract_trial_id({"protocolSection": {}})
        assert result.startswith("unknown_trial_")

    def test_check_trial_has_full_data_complete_trial(self, trial_extractor, sample_trial_data):
        """Test checking if trial has full data - should return True."""
        # Add derived section and outcomes to make it appear complete
        sample_trial_data["derivedSection"] = {"someData": "test"}
        sample_trial_data["protocolSection"]["outcomesModule"] = {
            "primaryOutcomes": [{"title": "Test Outcome"}]
        }
        # Add collaborators to reach the 3 indicator threshold
        sample_trial_data["protocolSection"]["sponsorCollaboratorsModule"]["collaborators"] = [
            {"name": "Test Collaborator"}
        ]
        result = trial_extractor.check_trial_has_full_data(sample_trial_data)
        assert result is True

    def test_check_trial_has_full_data_partial_trial(self, trial_extractor):
        """Test checking if trial has partial data - should return False."""
        partial_trial = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT12345678"},
                "eligibilityModule": {"eligibilityCriteria": "Short criteria"},
            }
        }
        result = trial_extractor.check_trial_has_full_data(partial_trial)
        assert result is False

    def test_check_trial_has_full_data_invalid_input(self, trial_extractor):
        """Test checking trial data completeness with invalid input."""
        result = trial_extractor.check_trial_has_full_data("invalid")
        assert result is False

    def test_check_trial_has_full_data_empty_input(self, trial_extractor):
        """Test checking trial data completeness with empty input."""
        result = trial_extractor.check_trial_has_full_data({})
        assert result is False

    def test_check_trial_has_full_data_none_input(self, trial_extractor):
        """Test checking trial data completeness with None input."""
        result = trial_extractor.check_trial_has_full_data(None)
        assert result is False
