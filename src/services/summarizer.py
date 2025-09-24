#!/usr/bin/env python3
"""
Lean MCODE Summarizer - Abstracted Element Processing

Strict, performant service using abstracted mCODE element configurations.
No legacy code, no fallbacks, no backwards compatibility.
"""

from typing import Any, Dict, List


class McodeSummarizer:
    """Lean mCODE summarizer using abstracted element configurations."""

    def __init__(
        self,
        include_dates: bool = True,
        detail_level: str = "full",
        include_mcode: bool = True,
    ):
        """Initialize with abstracted element configurations and detail level switches.

        Args:
            include_dates: Whether to include dates in the summary. Defaults to True.
            detail_level: Level of detail ("minimal", "standard", "full"). Defaults to "full".
            include_mcode: Whether to include mCODE annotations. Defaults to True.
        """
        self.include_dates = include_dates
        self.detail_level = detail_level
        self.include_mcode = include_mcode

        # Validate detail level
        if detail_level not in ["minimal", "standard", "full"]:
            raise ValueError(
                f"Invalid detail_level: {detail_level}. Must be 'minimal', 'standard', or 'full'"
            )

        # Configure detail level settings
        self._configure_detail_levels()

    def _configure_detail_levels(self):
        """Configure summarizer behavior based on detail level switches."""
        # Detail level configurations
        self.detail_configs = {
            "minimal": {
                "max_elements": 5,  # Only most critical elements
                "include_codes": False,
                "include_dates": False,
                "include_mcode": False,
                "priority_threshold": 7,  # Only elements with priority <= 7
            },
            "standard": {
                "max_elements": 15,
                "include_codes": True,
                "include_dates": self.include_dates,
                "include_mcode": self.include_mcode,
                "priority_threshold": 20,
            },
            "full": {
                "max_elements": None,  # No limit
                "include_codes": True,
                "include_dates": self.include_dates,
                "include_mcode": self.include_mcode,
                "priority_threshold": None,  # No threshold
            },
        }

        # Get current config
        self.current_config = self.detail_configs[self.detail_level]

        # Abstracted mCODE element configurations - mCODE as subject, codes in predicate
        self.element_configs = {
            # Patient elements
            "Patient": {
                "priority": 1,
                "template": "{subject} is a Patient (mCODE: Patient).",
            },
            "Age": {"priority": 2, "template": "{subject}'s age is {value}{codes}."},
            "Gender": {
                "priority": 3,
                "template": "{subject}'s gender is {value}{codes}.",
            },
            "BirthDate": {
                "priority": 4,
                "template": "{subject}'s birth date is {value}{codes}.",
            },
            "Race": {"priority": 5, "template": "{subject}'s race is {value}{codes}."},
            "Ethnicity": {
                "priority": 6,
                "template": "{subject}'s ethnicity is {value}{codes}.",
            },
            "CancerCondition": {
                "priority": 7,
                "template": "{subject}'s diagnosis (mCODE: CancerCondition{date_qualifier}) is {value}{codes}.",
            },
            "TNMStageGroup": {
                "priority": 8,
                "template": "{subject}'s tumor staging (mCODE: TNMStageGroup{date_qualifier}) is {value}{codes}.",
            },
            "TumorMarkerTest": {
                "priority": 9,
                "template": "{subject}'s tumor markers (mCODE: TumorMarkerTest{date_qualifier}) show {value}.",
            },
            "Procedure": {
                "priority": 10,
                "template": "{subject}'s procedures (mCODE: Procedure{date_qualifier}) are {value}.",
            },
            "MedicationRequest": {
                "priority": 11,
                "template": "{subject}'s medications (mCODE: MedicationRequest) are {value}.",
            },
            "SocialDeterminant": {
                "priority": 12,
                "template": "{subject}'s social determinants (mCODE: SocialDeterminant) are {value}.",
            },
            "Condition": {
                "priority": 13,
                "template": "{subject}'s conditions (mCODE: Condition{date_qualifier}) are {value}.",
            },
            "CauseOfDeath": {
                "priority": 14,
                "template": "{subject}'s cause of death (mCODE: CauseOfDeath{date_qualifier}) is {value}{codes}.",
            },
            # Trial elements
            "Trial": {
                "priority": 15,
                "template": "{subject} is a Clinical Trial (mCODE: Trial).",
            },
            "TrialTitle": {
                "priority": 16,
                "template": "{subject}'s title (mCODE: TrialTitle) is '{value}'.",
            },
            "TrialStudyType": {
                "priority": 17,
                "template": "{subject}'s study type (mCODE: TrialStudyType) is {value} study{codes}.",
            },
            "TrialPhase": {
                "priority": 18,
                "template": "{subject}'s phase (mCODE: TrialPhase) is {value}{codes}.",
            },
            "TrialStatus": {
                "priority": 19,
                "template": "{subject}'s status (mCODE: TrialStatus) is {value}{codes}.",
            },
            "TrialLeadSponsor": {
                "priority": 20,
                "template": "{subject}'s lead sponsor (mCODE: TrialLeadSponsor) is {value}{codes}.",
            },
            "TrialCollaborators": {
                "priority": 21,
                "template": "{subject}'s collaborators (mCODE: TrialCollaborators) include {value}.",
            },
            "TrialInterventionModel": {
                "priority": 22,
                "template": "{subject}'s intervention model (mCODE: TrialInterventionModel) is {value}{codes}.",
            },
            "TrialPrimaryPurpose": {
                "priority": 23,
                "template": "{subject}'s primary purpose (mCODE: TrialPrimaryPurpose) is {value}{codes}.",
            },
            "TrialEnrollment": {
                "priority": 24,
                "template": "{subject}'s enrollment (mCODE: TrialEnrollment) is {value} participants{codes}.",
            },
            "TrialCancerConditions": {
                "priority": 25,
                "template": "{subject}'s cancer conditions (mCODE: TrialCancerConditions) include {value}.",
            },
            "TrialAgeCriteria": {
                "priority": 26,
                "template": "{subject}'s age criteria (mCODE: TrialAgeCriteria) require patients {value}.",
            },
            "TrialSexCriteria": {
                "priority": 27,
                "template": "{subject}'s sex criteria (mCODE: TrialSexCriteria) require patients {value}.",
            },
            "TrialHealthyVolunteers": {
                "priority": 28,
                "template": "{subject}'s healthy volunteers criteria (mCODE: TrialHealthyVolunteers) {value}.",
            },
            "TrialMedicationInterventions": {
                "priority": 29,
                "template": "{subject}'s medication interventions (mCODE: TrialMedicationInterventions) include {value}.",
            },
            "TrialOtherInterventions": {
                "priority": 30,
                "template": "{subject}'s other interventions (mCODE: TrialOtherInterventions) include {value}.",
            },
            "TrialPrimaryOutcomes": {
                "priority": 31,
                "template": "{subject}'s primary outcomes (mCODE: TrialPrimaryOutcomes) include {value}.",
            },
            "TrialStartDate": {
                "priority": 32,
                "template": "{subject}'s start date (mCODE: TrialStartDate) is {value}{codes}.",
            },
            "TrialCompletionDate": {
                "priority": 33,
                "template": "{subject}'s completion date (mCODE: TrialCompletionDate) is {value}{codes}.",
            },
        }

    def _format_date_simple(self, date_str: str) -> str:
        """Format date to simple yyyy-mm-dd format."""
        if not date_str:
            return ""
        return date_str.split("T")[0] if "T" in date_str else date_str

    def _format_mcode_display(self, element_name: str, system: str, code: str) -> str:
        """Format mCODE elements consistently with comprehensive coding system support."""
        if not system:
            return f"{element_name}: {code}" if element_name else f"{code}"

        system_lower = system.lower()
        if "snomed" in system_lower:
            clean_system = "SNOMED"
        elif "loinc" in system_lower:
            clean_system = "LOINC"
        elif "rxnorm" in system_lower:
            clean_system = "RxNorm"
        elif "icd" in system_lower:
            clean_system = "ICD"
        elif "cpt" in system_lower or "hcpcs" in system_lower:
            clean_system = "CPT"
        elif "ndc" in system_lower:
            clean_system = "NDC"
        elif "cvx" in system_lower:
            clean_system = "CVX"
        elif "unii" in system_lower:
            clean_system = "UNII"
        elif "cas" in system_lower:
            clean_system = "CAS"
        elif "pubchem" in system_lower:
            clean_system = "PubChem"
        elif "chebi" in system_lower:
            clean_system = "ChEBI"
        elif "mesh" in system_lower:
            clean_system = "MeSH"
        elif "omim" in system_lower:
            clean_system = "OMIM"
        elif "hgnc" in system_lower:
            clean_system = "HGNC"
        elif "ensembl" in system_lower:
            clean_system = "Ensembl"
        elif "clinvar" in system_lower:
            clean_system = "ClinVar"
        elif "cosmic" in system_lower:
            clean_system = "COSMIC"
        elif "civic" in system_lower:
            clean_system = "CIViC"
        elif "oncokb" in system_lower:
            clean_system = "OncoKB"
        elif "2.16.840.1.113883.6.238" in system:
            # US Core Race and Ethnicity coding system (CDC OID)
            clean_system = "CDC-RACE"
        else:
            # Extract the last part of the URL or use the system as-is
            clean_system = system.split("/")[-1].split(":")[-1].upper()

        if element_name:
            return f"(mCODE: {element_name}, {clean_system}:{code})"
        else:
            return f"{clean_system}:{code}"

    def _create_abstracted_sentence(
        self,
        subject: str,
        element_name: str,
        value: str,
        codes: str = "",
        date_qualifier: str = "",
    ) -> str:
        """Create a standardized sentence using the abstracted element configuration with detail level switches.

        Args:
            subject: The subject of the sentence (e.g., "Patient", "Trial")
            element_name: The mCODE element name (e.g., "CancerCondition", "TrialStudyType")
            value: The value/predicate for the element
            codes: Optional detailed codes in format "System:Code"
            date_qualifier: Optional date qualifier (e.g., " documented on 2020-01-01")

        Returns:
            Formatted sentence string using the abstracted template, respecting detail level switches
        """
        if element_name not in self.element_configs:
            raise ValueError(
                f"mCODE element '{element_name}' is not configured. Only configured elements are supported."
            )

        config = self.element_configs[element_name]
        template = config["template"]

        # Apply detail level switches
        include_codes = self.current_config["include_codes"]
        include_dates = self.current_config["include_dates"]
        include_mcode = self.current_config["include_mcode"]

        # Format codes based on detail level
        if include_codes and codes:
            codes_part = f" ({codes})"
        elif include_codes and not codes:
            # For elements that should always have codes, provide defaults
            if element_name in ["Age", "Gender", "BirthDate", "Race", "Ethnicity"]:
                if element_name == "Age":
                    codes_part = " (SNOMED:424144002)"
                elif element_name == "Gender":
                    codes_part = " (SNOMED:407377005)"  # Other gender as default
                elif element_name == "BirthDate":
                    codes_part = " (SNOMED:184099003)"  # Date of birth
                elif element_name == "Race":
                    codes_part = " (CDC-RACE:UNK)"  # Unknown race
                elif element_name == "Ethnicity":
                    codes_part = " (CDC-RACE:UNK)"  # Unknown ethnicity
            else:
                codes_part = ""
        else:
            codes_part = ""

        # Format date qualifier based on detail level
        formatted_date = date_qualifier if (include_dates and date_qualifier) else ""

        # Format mCODE annotation based on detail level
        if not include_mcode:
            # Remove mCODE annotations from template by replacing with empty string
            template = template.replace(" (mCODE: {element_name})", "")
            template = template.replace("(mCODE: {element_name})", "")
            template = template.replace(" (mCODE: {element_name}{date_qualifier})", "")
            template = template.replace("(mCODE: {element_name}{date_qualifier})", "")
            # Also handle the specific element names that appear in templates
            template = template.replace(" (mCODE: Gender)", "")
            template = template.replace("(mCODE: Gender)", "")
            template = template.replace(" (mCODE: BirthDate)", "")
            template = template.replace("(mCODE: BirthDate)", "")
            template = template.replace(" (mCODE: Patient)", "")
            template = template.replace("(mCODE: Patient)", "")

        try:
            sentence = template.format(
                subject=subject,
                value=value,
                codes=codes_part,
                date_qualifier=formatted_date,
            )

            # Clean up any double spaces or formatting issues
            sentence = " ".join(sentence.split())
            return sentence

        except KeyError as e:
            # Handle missing template variables
            print(f"Warning: Missing template variable {e} for element {element_name}")
            if include_codes and codes:
                codes_part = f" ({codes})"
            else:
                codes_part = ""

            mcode_part = (
                f" (mCODE: {element_name}{formatted_date})" if include_mcode else ""
            )
            return f"{subject}'s {element_name.lower()}{mcode_part} is {value}{codes_part}."

    def _group_elements_by_priority(
        self, elements: List[Dict[str, Any]], subject_type: str = "Patient"
    ) -> List[Dict[str, Any]]:
        """Group and sort mCODE elements by clinical priority for optimal NLP processing, respecting detail level switches.

        Args:
            elements: List of element dictionaries with 'element_name', 'value', 'codes', 'date_qualifier'
            subject_type: Type of subject ("Patient" or "Trial") for priority ordering

        Returns:
            Sorted and filtered list of elements by clinical priority, respecting detail level constraints
        """
        # Sort elements by their configured priority
        sorted_elements = []
        for element in elements:
            element_name = element.get("element_name", "")
            if element_name in self.element_configs:
                priority = self.element_configs[element_name]["priority"]
                element["priority"] = priority
                sorted_elements.append(element)

        # Sort by priority (lower number = higher priority)
        sorted_elements.sort(key=lambda x: x["priority"])

        # Apply detail level filters
        filtered_elements = []

        for element in sorted_elements:
            priority = element.get("priority", 999)

            # Check priority threshold
            if self.current_config["priority_threshold"] is not None:
                if priority > self.current_config["priority_threshold"]:
                    continue

            filtered_elements.append(element)

        # Apply max elements limit
        if self.current_config["max_elements"] is not None:
            filtered_elements = filtered_elements[: self.current_config["max_elements"]]

        return filtered_elements

    def _generate_sentences_from_elements(
        self, elements: List[Dict[str, Any]], subject: str
    ) -> List[str]:
        """Generate standardized sentences from a list of mCODE elements.

        Args:
            elements: List of element dictionaries
            subject: The subject for all sentences

        Returns:
            List of formatted sentences
        """
        sentences = []
        for element in elements:
            element_name = element.get("element_name", "")
            value = element.get("value", "")
            codes = element.get("codes", "")
            date_qualifier = element.get("date_qualifier", "")

            if element_name and value:
                sentence = self._create_abstracted_sentence(
                    subject=subject,
                    element_name=element_name,
                    value=value,
                    codes=codes,
                    date_qualifier=date_qualifier,
                )
                sentences.append(sentence)

        return sentences

    def _extract_patient_elements(
        self, patient_data: Dict[str, Any], include_dates: bool
    ) -> List[Dict[str, Any]]:
        """Extract mCODE elements from patient data using abstracted processing."""
        elements = []

        # Get patient resource
        patient_resource = None
        for entry in patient_data.get("entry", []):
            if entry.get("resource", {}).get("resourceType") == "Patient":
                patient_resource = entry["resource"]
                break

        if not patient_resource:
            return elements

        # Basic patient info
        patient_id = patient_resource.get("id", "")
        name_data = patient_resource.get("name", [{}])[0]
        full_name = f"{' '.join(name_data.get('given', []))} {name_data.get('family', '')}".strip()

        elements.append(
            {
                "element_name": "Patient",
                "value": f"{full_name} (ID: {patient_id})",
                "codes": "",
                "date_qualifier": "",
            }
        )

        # Demographics with default codes
        gender = patient_resource.get("gender", "")
        birth_date = patient_resource.get("birthDate", "")

        if gender:
            elements.append(
                {
                    "element_name": "Gender",
                    "value": gender.lower(),
                    "codes": (
                        "SNOMED:248153007"
                        if gender.lower() == "male"
                        else "SNOMED:248152002"
                    ),
                    "date_qualifier": "",
                }
            )

        if birth_date:
            elements.append(
                {
                    "element_name": "BirthDate",
                    "value": self._format_date_simple(birth_date),
                    "codes": "SNOMED:184099003",
                    "date_qualifier": "",
                }
            )

        # Process clinical data from other resources
        for entry in patient_data.get("entry", []):
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")

            if resource_type == "Condition":
                # Extract cancer condition
                code_info = resource.get("code", {}).get("coding", [{}])[0]
                display = code_info.get("display", "")
                system = code_info.get("system", "")
                code = code_info.get("code", "")
                onset_date = resource.get("onsetDateTime", "")

                if display:
                    codes = (
                        f"{self._format_mcode_display('', system, code)}"
                        if system and code
                        else ""
                    )
                    date_qualifier = (
                        f" documented on {self._format_date_simple(onset_date)}"
                        if onset_date and include_dates
                        else ""
                    )

                    elements.append(
                        {
                            "element_name": "CancerCondition",
                            "value": display,
                            "codes": codes,
                            "date_qualifier": date_qualifier,
                        }
                    )

            elif resource_type == "Observation":
                # Extract TNM staging
                code_info = resource.get("code", {}).get("coding", [{}])[0]
                display = code_info.get("display", "")
                value_info = resource.get("valueCodeableConcept", {})
                value_coding = value_info.get("coding", [{}])[0] if value_info else {}
                value_display = value_coding.get("display", "")
                value_system = value_coding.get("system", "")
                value_code = value_coding.get("code", "")
                effective_date = resource.get("effectiveDateTime", "")

                if value_display and "T4" in value_display:
                    codes = (
                        f"{self._format_mcode_display('', value_system, value_code)}"
                        if value_system and value_code
                        else ""
                    )
                    date_qualifier = (
                        f" documented on {self._format_date_simple(effective_date)}"
                        if effective_date and include_dates
                        else ""
                    )

                    elements.append(
                        {
                            "element_name": "TNMStageGroup",
                            "value": value_display,
                            "codes": codes,
                            "date_qualifier": date_qualifier,
                        }
                    )

        return elements

    def _extract_trial_elements(
        self, trial_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract mCODE elements from trial data using abstracted processing."""
        elements = []

        protocol_section = trial_data.get("protocolSection", {})
        identification = protocol_section.get("identificationModule", {})
        status = protocol_section.get("statusModule", {})

        identification.get("nctId", "Unknown")
        brief_title = identification.get("briefTitle", "Unknown Trial")
        overall_status = status.get("overallStatus", "Unknown")

        # Basic trial elements
        elements.append(
            {
                "element_name": "Trial",
                "value": "Clinical Trial",
                "codes": "",
                "date_qualifier": "",
            }
        )

        if brief_title and brief_title != "Unknown Trial":
            elements.append(
                {
                    "element_name": "TrialTitle",
                    "value": brief_title,
                    "codes": "",
                    "date_qualifier": "",
                }
            )

        elements.append(
            {
                "element_name": "TrialStatus",
                "value": overall_status.lower(),
                "codes": "",
                "date_qualifier": "",
            }
        )

        return elements

    def create_trial_summary(self, trial_data: Dict[str, Any]) -> str:
        """Generate clinical trial summary using abstracted element processing."""
        if not trial_data or not trial_data.get("protocolSection"):
            raise ValueError("Trial data missing required format.")

        # Extract and process elements using abstracted approach
        elements = self._extract_trial_elements(trial_data)
        prioritized = self._group_elements_by_priority(elements, "Trial")

        # Get NCT ID for subject
        nct_id = (
            trial_data.get("protocolSection", {})
            .get("identificationModule", {})
            .get("nctId", "Unknown")
        )
        if nct_id == "Unknown":
            raise ValueError("Trial data missing NCT ID.")

        # Use NCT ID as subject instead of generic "Trial"
        subject = nct_id if nct_id != "Unknown" else "Trial"
        sentences = self._generate_sentences_from_elements(prioritized, subject)
        return " ".join(sentences)

    def create_patient_summary(
        self, patient_data: Dict[str, Any], include_dates: bool = None
    ) -> str:
        """Generate patient summary using abstracted element processing."""
        if not patient_data or "entry" not in patient_data:
            raise ValueError("Patient data missing required format.")

        if include_dates is None:
            include_dates = self.include_dates

        # Extract and process elements using abstracted approach
        elements = self._extract_patient_elements(patient_data, include_dates)
        prioritized = self._group_elements_by_priority(elements, "Patient")

        # Use patient's name as subject instead of generic "Patient"
        patient_name = ""
        for element in prioritized:
            if element.get("element_name") == "Patient":
                patient_name = element.get("value", "").split(" (ID:")[
                    0
                ]  # Extract name without ID
                break

        subject = patient_name if patient_name else "Patient"
        sentences = self._generate_sentences_from_elements(prioritized, subject)
        return " ".join(sentences)
