"""
Trial Extractor - Extract mCODE elements from clinical trial data.

This module provides functionality to extract mCODE-relevant information
from clinical trial protocol sections.
"""

from typing import Any, Dict


class TrialExtractor:
    """Extract mCODE elements from clinical trial data."""

    def extract_trial_mcode_elements(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """Extract mCODE elements from clinical trial data."""
        mcode_elements: Dict[str, Any] = {}

        try:
            protocol_section = trial.get("protocolSection", {})
            if not isinstance(protocol_section, dict):
                return mcode_elements

            # Extract all mCODE-relevant sections
            extractors = [
                ("identificationModule", self._extract_trial_identification),
                ("eligibilityModule", self._extract_trial_eligibility_mcode),
                ("conditionsModule", self._extract_trial_conditions_mcode),
                ("armsInterventionsModule", self._extract_trial_interventions_mcode),
                ("designModule", self._extract_trial_design_mcode),
                ("statusModule", self._extract_trial_temporal_mcode),
                ("sponsorCollaboratorsModule", self._extract_trial_sponsor_mcode),
            ]

            for module_key, extractor_func in extractors:
                module_data = protocol_section.get(module_key, {})
                if isinstance(module_data, dict):
                    mcode_elements.update(extractor_func(module_data))

        except Exception as e:
            print(f"Error extracting trial mCODE elements: {e}")

        return mcode_elements

    def _extract_trial_identification(self, identification: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trial identification information."""
        elements = {}

        nct_id = identification.get("nctId")
        if nct_id:
            elements["TrialIdentifier"] = {
                "system": "https://clinicaltrials.gov",
                "code": nct_id,
                "display": f"Clinical Trial {nct_id}",
            }

        brief_title = identification.get("briefTitle")
        if brief_title:
            elements["TrialTitle"] = {
                "display": brief_title,
            }

        official_title = identification.get("officialTitle")
        if official_title:
            elements["TrialOfficialTitle"] = {
                "display": official_title,
            }

        return elements

    def _extract_trial_eligibility_mcode(self, eligibility: Dict[str, Any]) -> Dict[str, Any]:
        """Extract eligibility criteria in mCODE space for patient matching."""
        elements = {}

        # Type guard: ensure eligibility is a dict
        if not isinstance(eligibility, dict):
            print(f"Expected dict for eligibility, got {type(eligibility)}")
            return elements

        # Age criteria mapping to mCODE
        min_age = eligibility.get("minimumAge")
        max_age = eligibility.get("maximumAge")
        if min_age or max_age:
            elements["TrialAgeCriteria"] = {
                "minimumAge": min_age,
                "maximumAge": max_age,
                "ageUnit": "Years",  # Standardize to years
            }

        # Sex criteria mapping to mCODE AdministrativeGender
        sex = eligibility.get("sex")
        if sex:
            elements["TrialSexCriteria"] = {
                "system": "http://hl7.org/fhir/administrative-gender",
                "code": sex.lower(),
                "display": sex.capitalize(),
            }

        # Healthy volunteers criteria
        healthy_volunteers = eligibility.get("healthyVolunteers")
        if healthy_volunteers is not None:
            elements["TrialHealthyVolunteers"] = {
                "allowed": healthy_volunteers,
                "display": (
                    "Accepts healthy volunteers"
                    if healthy_volunteers
                    else "Does not accept healthy volunteers"
                ),
            }

        # Eligibility criteria text for detailed matching
        criteria_text = eligibility.get("eligibilityCriteria")
        if criteria_text:
            elements["TrialEligibilityCriteria"] = {
                "text": criteria_text,
                "display": "Detailed eligibility criteria for patient matching",
            }

        return elements

    def _extract_trial_conditions_mcode(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trial conditions as mCODE CancerCondition for matching."""
        elements = {}

        # Type guard: ensure conditions is a dict
        if not isinstance(conditions, dict):
            print(f"Expected dict for conditions, got {type(conditions)}")
            return elements

        condition_list = conditions.get("conditions", [])
        if condition_list:
            cancer_conditions = []
            comorbid_conditions = []

            for condition in condition_list:
                # Handle both dict and string formats for conditions
                if isinstance(condition, dict):
                    condition_name = condition.get("name", "").lower()
                    condition_code = condition.get("code", "Unknown")
                elif isinstance(condition, str):
                    # Handle case where condition is just a string
                    condition_name = condition.lower()
                    condition_code = "Unknown"
                    print(f"Condition is string format: {condition}")
                else:
                    print(f"Unexpected condition type {type(condition)}: {condition}")
                    continue

                # Check if it's a cancer condition
                if any(
                    cancer_term in condition_name
                    for cancer_term in [
                        "cancer",
                        "carcinoma",
                        "neoplasm",
                        "tumor",
                        "malignant",
                        "leukemia",
                        "lymphoma",
                        "sarcoma",
                        "glioma",
                        "melanoma",
                        "breast cancer",
                    ]
                ):
                    cancer_conditions.append(
                        {
                            "system": "http://snomed.info/sct",
                            "code": condition_code,
                            "display": (
                                condition_name if condition_name else "Unknown cancer condition"
                            ),
                            "interpretation": "Confirmed",
                        }
                    )
                else:
                    comorbid_conditions.append(
                        {
                            "system": "http://snomed.info/sct",
                            "code": condition_code,
                            "display": (condition_name if condition_name else "Unknown condition"),
                        }
                    )

            if cancer_conditions:
                elements["TrialCancerConditions"] = cancer_conditions
            if comorbid_conditions:
                elements["TrialComorbidConditions"] = comorbid_conditions

        return elements

    def _extract_trial_interventions_mcode(self, interventions: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trial interventions as mCODE CancerRelatedMedicationStatement."""
        elements = {}

        # Type guard: ensure interventions is a dict
        if not isinstance(interventions, dict):
            print(f"Expected dict for interventions, got {type(interventions)}")
            return elements

        intervention_list = interventions.get("interventions", [])
        if intervention_list:
            medication_interventions = []
            other_interventions = []

            for intervention in intervention_list:
                # Type guard: ensure intervention is a dict
                if not isinstance(intervention, dict):
                    print(
                        f"Expected dict for intervention, got {type(intervention)}: {intervention}"
                    )
                    continue

                intervention_type = intervention.get("type", "").lower()
                intervention_name = intervention.get("name", "")
                description = intervention.get("description", "")

                if intervention_type in ["drug", "biological", "device"]:
                    medication_interventions.append(
                        {
                            "system": "http://snomed.info/sct",
                            "code": "Unknown",  # Would need RxNorm mapping
                            "display": intervention_name,
                            "description": description,
                            "interventionType": intervention_type,
                        }
                    )
                else:
                    other_interventions.append(
                        {
                            "display": intervention_name,
                            "description": description,
                            "interventionType": intervention_type,
                        }
                    )

            if medication_interventions:
                elements["TrialMedicationInterventions"] = medication_interventions
            if other_interventions:
                elements["TrialOtherInterventions"] = other_interventions

        return elements

    def _extract_trial_design_mcode(self, design: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trial design elements for mCODE mapping."""
        elements = {}

        # Type guard: ensure design is a dict
        if not isinstance(design, dict):
            print(f"Expected dict for design, got {type(design)}")
            return elements

        # Study type
        study_type = design.get("studyType")
        if study_type:
            elements["TrialStudyType"] = {
                "display": study_type,
                "code": study_type.lower().replace(" ", "_"),
            }

        # Phase information
        phase = design.get("phases", [])
        if phase:
            elements["TrialPhase"] = {
                "display": ", ".join(phase),
                "phases": phase,
            }

        # Primary purpose
        primary_purpose = design.get("primaryPurpose")
        if primary_purpose:
            elements["TrialPrimaryPurpose"] = {
                "display": primary_purpose,
                "code": primary_purpose.lower().replace(" ", "_"),
            }

        # Enrollment information
        enrollment_info = design.get("enrollmentInfo", {})
        if enrollment_info:
            elements["TrialEnrollment"] = {
                "count": enrollment_info.get("count"),
                "type": enrollment_info.get("type"),
            }

        return elements

    def _extract_trial_temporal_mcode(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal information for trial phases and timelines."""
        elements = {}

        # Type guard: ensure status is a dict
        if not isinstance(status, dict):
            print(f"Expected dict for status, got {type(status)}")
            return elements

        # Overall status
        overall_status = status.get("overallStatus")
        if overall_status:
            elements["TrialStatus"] = {
                "display": overall_status,
                "code": overall_status.lower().replace(" ", "_"),
            }

        # Start date
        start_date = status.get("startDateStruct", {}).get("date")
        if start_date:
            elements["TrialStartDate"] = {
                "date": start_date,
                "display": f"Trial started on {start_date}",
            }

        # Completion date
        completion_date = status.get("completionDateStruct", {}).get("date")
        if completion_date:
            elements["TrialCompletionDate"] = {
                "date": completion_date,
                "display": f"Trial completed on {completion_date}",
            }

        # Primary completion date
        primary_completion_date = status.get("primaryCompletionDateStruct", {}).get("date")
        if primary_completion_date:
            elements["TrialPrimaryCompletionDate"] = {
                "date": primary_completion_date,
                "display": f"Primary completion on {primary_completion_date}",
            }

        return elements

    def _extract_trial_sponsor_mcode(self, sponsor: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sponsor and organization information."""
        elements = {}

        # Type guard: ensure sponsor is a dict
        if not isinstance(sponsor, dict):
            print(f"Expected dict for sponsor, got {type(sponsor)}")
            return elements

        # Lead sponsor
        lead_sponsor = sponsor.get("leadSponsor", {})
        if lead_sponsor:
            elements["TrialLeadSponsor"] = {
                "name": lead_sponsor.get("name"),
                "class": lead_sponsor.get("class"),
            }

        # Responsible party
        responsible_party = sponsor.get("responsibleParty", {})
        if responsible_party:
            elements["TrialResponsibleParty"] = {
                "name": responsible_party.get("name"),
                "type": responsible_party.get("type"),
                "affiliation": responsible_party.get("affiliation"),
            }

        return elements

    def extract_trial_metadata(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive trial metadata for storage."""
        metadata: Dict[str, Any] = {}

        try:
            # Ensure trial is a dict
            if not isinstance(trial, dict):
                print(f"Trial data is not a dict in metadata extraction: {type(trial)}")
                return metadata

            protocol_section = trial.get("protocolSection", {})
            if not isinstance(protocol_section, dict):
                print(
                    f"Protocol section is not a dict in metadata extraction: {type(protocol_section)}"
                )
                return metadata

            # Basic trial info
            identification = protocol_section.get("identificationModule", {})
            if isinstance(identification, dict):
                metadata["nct_id"] = identification.get("nctId")
                metadata["brief_title"] = identification.get("briefTitle")
                metadata["official_title"] = identification.get("officialTitle")

            # Status and dates
            status = protocol_section.get("statusModule", {})
            if isinstance(status, dict):
                metadata["overall_status"] = status.get("overallStatus")
                start_struct = status.get("startDateStruct", {})
                if isinstance(start_struct, dict):
                    metadata["start_date"] = start_struct.get("date")
                completion_struct = status.get("completionDateStruct", {})
                if isinstance(completion_struct, dict):
                    metadata["completion_date"] = completion_struct.get("date")

            # Design information
            design = protocol_section.get("designModule", {})
            if isinstance(design, dict):
                metadata["study_type"] = design.get("studyType")
                metadata["phase"] = design.get("phases", [])
                metadata["primary_purpose"] = design.get("primaryPurpose")

            # Eligibility
            eligibility = protocol_section.get("eligibilityModule", {})
            if isinstance(eligibility, dict):
                metadata["minimum_age"] = eligibility.get("minimumAge")
                metadata["maximum_age"] = eligibility.get("maximumAge")
                metadata["sex"] = eligibility.get("sex")
                metadata["healthy_volunteers"] = eligibility.get("healthyVolunteers")

            # Conditions and interventions
            conditions_module = protocol_section.get("conditionsModule", {})
            if isinstance(conditions_module, dict):
                conditions = conditions_module.get("conditions", [])
                if isinstance(conditions, list):
                    metadata["conditions"] = [
                        c.get("name") for c in conditions if isinstance(c, dict)
                    ]

            interventions_module = protocol_section.get("armsInterventionsModule", {})
            if isinstance(interventions_module, dict):
                interventions = interventions_module.get("interventions", [])
                if isinstance(interventions, list):
                    metadata["interventions"] = [
                        i.get("name") for i in interventions if isinstance(i, dict)
                    ]

        except Exception as e:
            print(f"Error extracting trial metadata: {e}")

        return metadata

    def extract_trial_id(self, trial: Dict[str, Any]) -> str:
        """Extract trial ID from trial data."""
        try:
            trial_id = trial["protocolSection"]["identificationModule"]["nctId"]
            return trial_id if isinstance(trial_id, str) else ""
        except (KeyError, TypeError):
            import hashlib

            return f"unknown_trial_{hashlib.md5(str(trial).encode('utf-8')).hexdigest()[:8]}"

    def check_trial_has_full_data(self, trial: Dict[str, Any]) -> bool:
        """Check if trial data appears to be complete (from full study API) vs partial (from search API)."""
        if not trial or not isinstance(trial, dict):
            return False

        protocol_section = trial.get("protocolSection", {})
        if not isinstance(protocol_section, dict):
            return False

        # Check for fields that indicate full study data
        indicators = []

        # 1. Detailed eligibility criteria (longer than search results)
        eligibility = protocol_section.get("eligibilityModule", {})
        if isinstance(eligibility, dict):
            criteria = eligibility.get("eligibilityCriteria", "")
            if criteria and len(criteria) > 100:  # Search results are usually truncated
                indicators.append(True)

        # 2. Interventions with detailed information
        arms = protocol_section.get("armsInterventionsModule", {})
        if isinstance(arms, dict):
            interventions = arms.get("interventions", [])
            if interventions and len(interventions) > 0:
                # Check if interventions have detailed descriptions
                detailed_interventions = any(
                    isinstance(i, dict) and i.get("description", "") for i in interventions
                )
                if detailed_interventions:
                    indicators.append(True)

        # 3. Outcomes module (rarely present in search results)
        outcomes = protocol_section.get("outcomesModule", {})
        if isinstance(outcomes, dict) and outcomes.get("primaryOutcomes"):
            indicators.append(True)

        # 4. Derived section (only in full study data)
        derived_section = trial.get("derivedSection")
        if derived_section and isinstance(derived_section, dict):
            indicators.append(True)

        # 5. Detailed sponsor/collaborator information
        sponsor = protocol_section.get("sponsorCollaboratorsModule", {})
        if isinstance(sponsor, dict):
            collaborators = sponsor.get("collaborators", [])
            if collaborators and len(collaborators) > 0:
                indicators.append(True)

        # Consider data complete if we have at least 3 indicators
        return len(indicators) >= 3
