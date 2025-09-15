#!/usr/bin/env python3
"""
Centralized MCODE Summarization Service

This module provides a lean, performant, and strict service for generating
natural language summaries from mCODE data for both patients and clinical trials.
"""

from typing import Dict, Any, List

class McodeSummarizer:
    """A centralized service for creating mCODE summaries."""

    def __init__(self):
        pass

    def _format_mcode_display(self, element_name: str, system: str, code: str) -> str:
        """Centralized function to format mCODE elements consistently with comprehensive coding system support."""
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
        else:
            # Extract the last part of the URL or use the system as-is
            clean_system = system.split("/")[-1].split(":")[-1].upper()

        if element_name:
            return f"(mCODE: {element_name}, {clean_system}:{code})"
        else:
            return f"{clean_system}:{code}"

    def _format_date_simple(self, date_str: str) -> str:
        """Format date to simple yyyy-mm-dd format."""
        if not date_str:
            return ""
        # Extract just the date part (yyyy-mm-dd) from ISO format
        return date_str.split('T')[0] if 'T' in date_str else date_str

    def create_trial_summary(self, trial_data: Dict[str, Any]) -> str:
        """
        Generate a comprehensive clinical trial summary in natural language format.

        Args:
            trial_data: Clinical trial data from ClinicalTrials.gov API

        Returns:
            str: Natural language summary of the clinical trial

        Raises:
            ValueError: If trial data is missing required fields
        """
        if not trial_data or not trial_data.get("protocolSection"):
            raise ValueError("Trial data is missing or not in the expected format.")

        # Extract key information from trial data
        protocol_section = trial_data.get("protocolSection", {})
        identification = protocol_section.get("identificationModule", {})
        status = protocol_section.get("statusModule", {})
        eligibility = protocol_section.get("eligibilityModule", {})
        design = protocol_section.get("designModule", {})
        arms = protocol_section.get("armsInterventionsModule", {})
        sponsor = protocol_section.get("sponsorCollaboratorsModule", {})
        outcomes = protocol_section.get("outcomesModule", {})
        conditions = protocol_section.get("conditionsModule", {})

        # Trial identification
        nct_id = identification.get("nctId", "Unknown")
        brief_title = identification.get("briefTitle", "Unknown Trial")

        if nct_id == "Unknown" or brief_title == "Unknown Trial":
            raise ValueError("Trial data is missing required fields: nctId or briefTitle.")

        # Study design
        study_type = design.get("studyType", "INTERVENTIONAL")
        phases = design.get("phases", [])
        phase_text = " ".join(phases) if phases else "Not specified"
        display_phase = phase_text.lower().replace('phase', 'phase ') if phase_text != "Not specified" else phase_text
        primary_purpose = design.get("primaryPurpose", "Not specified")

        # Build clinical trial summary in patient narrative style for NLP processing
        clinical_note = []

        # Opening with trial identification and title
        clinical_note.append(f"{nct_id} is a clinical trial (mCODE: Trial) entitled '{brief_title}'.")

        # Trial characteristics with subject-predicate structure
        overall_status = status.get("overallStatus", "Unknown")
        last_known_status = status.get("lastKnownStatus", "Unknown")

        # Use last known status if overall status is UNKNOWN
        if overall_status == "UNKNOWN" and last_known_status != "Unknown":
            display_status = last_known_status.lower().replace('_', ' ')
        else:
            display_status = overall_status.lower()

        # Build status sentence - be concise and avoid "Not specified" values
        status_parts = [f"Trial is an {study_type.lower()} study (mCODE: TrialStudyType)"]

        # Only include phase if it's meaningful (not "Not specified")
        if display_phase != "Not specified":
            status_parts.append(f"in {display_phase} (mCODE: TrialPhase)")

        # Add status with proper grammar
        status_parts.append(f"that has been {display_status} (mCODE: TrialStatus)")

        clinical_note.append(" ".join(status_parts) + ".")

        # Sponsor and collaborators
        lead_sponsor = sponsor.get("leadSponsor", {})
        sponsor_name = lead_sponsor.get("name")
        if sponsor_name and sponsor_name != "Unknown Sponsor":
            collaborators = sponsor.get("collaborators", [])
            if collaborators:
                collab_names = [collab.get("name", "Unknown") for collab in collaborators]
                clinical_note.append(f"Trial is sponsored by {sponsor_name} (mCODE: TrialLeadSponsor) and has collaborators including {' and '.join(collab_names)} (mCODE: TrialCollaborators).")
            else:
                clinical_note.append(f"Trial is sponsored by {sponsor_name} (mCODE: TrialLeadSponsor).")

        # Study design and enrollment
        enrollment_info = design.get("enrollmentInfo", {})
        count = enrollment_info.get("count")
        design_info = design.get("designInfo", {})
        intervention_model = design_info.get("interventionModel")

        # Extract conditions for context
        condition_list = conditions.get("conditions", [])
        condition_names = []
        for cond in condition_list:
            if isinstance(cond, dict):
                condition_names.append(cond.get("name", "Unknown"))
            elif isinstance(cond, str):
                condition_names.append(cond)
            else:
                condition_names.append(str(cond))
        primary_condition = condition_names[0] if condition_names else "Unknown condition"

        # Build study description - be concise and avoid "unknown" values
        study_parts = []
        if intervention_model and intervention_model != "Unknown":
            study_parts.append(f"{intervention_model.lower().replace('_', '-')}")
        if primary_purpose and primary_purpose != "Not specified":
            study_parts.append(f"{primary_purpose.lower()}")

        study_description = " ".join(study_parts) if study_parts else ""

        # Create enrollment info only if we have a real count
        enrollment_text = ""
        if count and str(count).isdigit() and int(count) > 0:
            enrollment_text = f" that enrolled {count} participants (mCODE: TrialEnrollment)"

        # Main study description - focus on what we know
        if study_description and enrollment_text:
            clinical_note.append(f"Trial is a {study_description} study (mCODE: TrialInterventionModel; mCODE: TrialPrimaryPurpose){enrollment_text} to study {primary_condition} (mCODE: TrialCancerConditions).")
        elif study_description:
            clinical_note.append(f"Trial is a {study_description} study (mCODE: TrialInterventionModel; mCODE: TrialPrimaryPurpose) to study {primary_condition} (mCODE: TrialCancerConditions).")
        else:
            clinical_note.append(f"Trial is to study {primary_condition} (mCODE: TrialCancerConditions).")

        # Eligibility criteria - only include criteria that exist
        min_age = eligibility.get("minimumAge")
        max_age = eligibility.get("maximumAge")
        sex = eligibility.get("sex", "ALL")
        healthy_volunteers = eligibility.get("healthyVolunteers", False)

        eligibility_parts = []

        # Age criteria
        if min_age or max_age:
            if min_age and max_age:
                age_text = f"aged {min_age} to {max_age}"
            elif min_age:
                age_text = f"aged {min_age} or older"
            elif max_age:
                age_text = f"up to {max_age} of age"
            eligibility_parts.append(f"{age_text} (mCODE: TrialAgeCriteria)")

        # Sex criteria
        if sex != "ALL":
            sex_display = sex.lower()
            eligibility_parts.append(f"of {sex_display} gender (mCODE: TrialSexCriteria)")
        else:
            eligibility_parts.append("of any gender (mCODE: TrialSexCriteria)")

        # Healthy volunteers criteria
        volunteer_display = "accepts" if healthy_volunteers else "does not accept"
        eligibility_parts.append(f"and {volunteer_display} healthy volunteers (mCODE: TrialHealthyVolunteers)")

        # Only add eligibility section if we have meaningful criteria
        if eligibility_parts:
            clinical_note.append(f"Trial has eligibility criteria requiring patients {' '.join(eligibility_parts)}.")

        # Treatment interventions
        interventions = arms.get("interventions", [])
        drug_interventions = [i for i in interventions if i.get("type") == "DRUG"]
        other_interventions = [i for i in interventions if i.get("type") != "DRUG"]

        if drug_interventions:
            drug_names = [drug.get("name", "Unknown") for drug in drug_interventions]
            if len(drug_names) == 1:
                clinical_note.append(f"Trial evaluates {drug_names[0]} (mCODE: TrialMedicationInterventions).")
            else:
                clinical_note.append(f"Trial evaluates {', '.join(drug_names[:-1])}, and {drug_names[-1]} (mCODE: TrialMedicationInterventions).")

        if other_interventions:
            other_names = [int.get("name", "Unknown") for int in other_interventions]
            if len(other_names) == 1:
                clinical_note.append(f"Trial includes {other_names[0]} (mCODE: TrialOtherInterventions).")
            else:
                clinical_note.append(f"Trial includes {', '.join(other_names[:-1])}, and {other_names[-1]} (mCODE: TrialOtherInterventions).")

        # Outcomes and timeline
        primary_outcomes = outcomes.get("primaryOutcomes", [])
        if primary_outcomes:
            primary_measures = [outcome.get("measure", "Unknown") for outcome in primary_outcomes]
            if len(primary_measures) == 1:
                clinical_note.append(f"Trial has primary outcome of {primary_measures[0]} (mCODE: TrialPrimaryOutcomes).")
            else:
                clinical_note.append(f"Trial has primary outcomes of {', '.join(primary_measures[:-1])}, and {primary_measures[-1]} (mCODE: TrialPrimaryOutcomes).")

        start_date = status.get("startDateStruct", {}).get("date")
        completion_date = status.get("completionDateStruct", {}).get("date")

        timeline_info = []
        if start_date:
            timeline_info.append(f"started on {start_date} (mCODE: TrialStartDate)")
        if completion_date:
            timeline_info.append(f"completed on {completion_date} (mCODE: TrialCompletionDate)")

        if timeline_info:
            clinical_note.append(f"Trial {' and '.join(timeline_info)}.")

        # Return as one continuous paragraph
        return " ".join(clinical_note)

    def create_patient_summary(self, patient_data: Dict[str, Any]) -> str:
        """
        Generate a comprehensive patient summary optimized for NLP entity extraction and clinical trial matching.

        Args:
            patient_data: A dictionary representing a patient's FHIR bundle.

        Returns:
            A natural language summary of the patient's mCODE data optimized for clinical trial matching.

        Raises:
            ValueError: If the patient data is missing required fields.
        """
        if not patient_data or "entry" not in patient_data:
            raise ValueError("Patient data is missing or not in the expected format.")

        patient_resource = None
        for entry in patient_data.get("entry", []):
            if entry.get("resource", {}).get("resourceType") == "Patient":
                patient_resource = entry["resource"]
                break

        if not patient_resource:
            raise ValueError("No Patient resource found in the patient data.")

        patient_id = patient_resource.get("id")
        if not patient_id:
            raise ValueError("Patient resource is missing an 'id'.")

        name_data = patient_resource.get("name", [{}])[0]
        full_name = f"{' '.join(name_data.get('given', []))} {name_data.get('family', '')}".strip()
        gender = patient_resource.get("gender")
        birth_date = patient_resource.get("birthDate")

        if not all([full_name, gender, birth_date]):
            raise ValueError("Patient resource is missing one or more of the following: name, gender, birthDate.")

        # Calculate age
        age = "unknown"
        if birth_date:
            try:
                from datetime import datetime
                birth_datetime = datetime.fromisoformat(birth_date)
                age = (datetime.now() - birth_datetime).days // 365
            except (ValueError, TypeError):
                pass

        # Check if patient is deceased (has cause of death)
        is_deceased = False
        for entry in patient_data.get("entry", []):
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")
            if resource_type == "Observation":
                meta = resource.get("meta", {})
                profiles = meta.get("profile", [])
                code_info = resource.get("code", {}).get("coding", [{}])[0]
                display = code_info.get("display", "Unknown observation")
                if "cause of death" in display.lower():
                    is_deceased = True
                    break

        # Build natural language clinical note optimized for NLP
        deceased_modifier = "deceased " if is_deceased else ""
        clinical_note = [f"Patient {full_name} (ID: {patient_id}) is a {deceased_modifier}{age}-year-old {gender}."]

        # Collect comprehensive clinical information with dates and codes for mCODE consolidation
        primary_diagnosis = None
        diagnosis_info = {'dates': []}
        tnm_components = {}  # T, N, M staging with dates and codes
        biomarkers = {}  # ER, PR, HER2 with dates and codes
        procedures = []  # Procedures with dates
        medications = []  # Medications
        social_determinants = []  # Social determinants of health
        other_conditions = []
        cause_of_death = None
        cause_of_death_info = {'dates': []}

        # Extract all mCODE elements from the bundle
        for entry in patient_data.get("entry", []):
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")

            if resource_type == "Condition":
                # Extract condition information with dates
                meta = resource.get("meta", {})
                profiles = meta.get("profile", [])
                is_primary_cancer_condition = any("mcode-primary-cancer-condition" in p for p in profiles)

                code_info = resource.get("code", {}).get("coding", [{}])[0]
                display = code_info.get("display")
                onset_date = resource.get("onsetDateTime") or resource.get("recordedDate")

                if is_primary_cancer_condition and display:
                    if not primary_diagnosis: # Set only once
                        primary_diagnosis = display
                        diagnosis_info['system'] = code_info.get("system", "")
                        diagnosis_info['code'] = code_info.get("code", "")
                    if onset_date:
                        diagnosis_info['dates'].append(onset_date)
                elif display and display not in [primary_diagnosis]:
                    # Collect other conditions with dates, but exclude cause of death
                    if not (cause_of_death and display in cause_of_death[0]):
                        simple_date = self._format_date_simple(onset_date) if onset_date else None
                        date_info = f" (diagnosed {simple_date})" if simple_date else ""
                        other_conditions.append(f"{display}{date_info}")

            elif resource_type == "Observation":
                # Extract comprehensive observation information
                meta = resource.get("meta", {})
                profiles = meta.get("profile", [])
                code_info = resource.get("code", {}).get("coding", [{}])[0]
                display = code_info.get("display", "Unknown observation")
                value_info = resource.get("valueCodeableConcept", {})
                value_coding = value_info.get("coding", [{}])[0] if value_info else {}
                value_display = value_coding.get("display")
                value_system = value_coding.get("system", "")
                value_code = value_coding.get("code")
                effective_date = resource.get("effectiveDateTime") or resource.get("issued")

                if value_display:
                    # Categorize observations with dates and codes consolidated in mCODE
                    if any("mcode-tnm-distant-metastases-category" in p for p in profiles):
                        stage_value = value_display.lower().replace(' category (finding)', '').replace('m', 'M')
                        if 'M' not in tnm_components:
                            tnm_components['M'] = {'value': stage_value, 'system': value_system, 'code': value_code, 'dates': []}
                        if effective_date:
                            tnm_components['M']['dates'].append(effective_date)
                    elif any("mcode-tnm-primary-tumor-category" in p for p in profiles):
                        stage_value = value_display.lower().replace(' category (finding)', '').replace('t', 'T')
                        if 'T' not in tnm_components:
                            tnm_components['T'] = {'value': stage_value, 'system': value_system, 'code': value_code, 'dates': []}
                        if effective_date:
                            tnm_components['T']['dates'].append(effective_date)
                    elif any("mcode-tnm-regional-nodes-category" in p for p in profiles):
                        stage_value = value_display.lower().replace(' category (finding)', '').replace('n', 'N')
                        if 'N' not in tnm_components:
                            tnm_components['N'] = {'value': stage_value, 'system': value_system, 'code': value_code, 'dates': []}
                        if effective_date:
                            tnm_components['N']['dates'].append(effective_date)
                    elif any("mcode-cancer-stage-group" in p for p in profiles):
                        stage_value = value_display.lower().replace(' (qualifier value)', '').replace('stage ', 'Stage ')
                        if 'stage' not in tnm_components:
                            tnm_components['stage'] = {'value': stage_value, 'system': value_system, 'code': value_code, 'dates': []}
                        if effective_date:
                            tnm_components['stage']['dates'].append(effective_date)
                    elif any("mcode-tumor-marker-test" in p for p in profiles):
                        if "estrogen receptor" in display.lower():
                            status = "positive" if "positive" in value_display.lower() else "negative"
                            key = "ER"
                            if key not in biomarkers:
                                biomarkers[key] = {'status': status, 'system': value_system, 'code': value_code, 'dates': []}
                            if effective_date:
                                biomarkers[key]['dates'].append(effective_date)
                        elif "progesterone receptor" in display.lower():
                            status = "positive" if "positive" in value_display.lower() else "negative"
                            key = "PR"
                            if key not in biomarkers:
                                biomarkers[key] = {'status': status, 'system': value_system, 'code': value_code, 'dates': []}
                            if effective_date:
                                biomarkers[key]['dates'].append(effective_date)
                        elif "her2" in display.lower():
                            status = "positive" if "positive" in value_display.lower() else "negative"
                            key = "HER2"
                            if key not in biomarkers:
                                biomarkers[key] = {'status': status, 'system': value_system, 'code': value_code, 'dates': []}
                            if effective_date:
                                biomarkers[key]['dates'].append(effective_date)
                    elif "cause of death" in display.lower():
                        if not cause_of_death:
                            cause_of_death = value_display
                            cause_of_death_info['system'] = value_system
                            cause_of_death_info['code'] = value_code
                        if effective_date:
                            cause_of_death_info['dates'].append(effective_date)
                    elif "tobacco" in display.lower():
                        social_determinants.append(f"Tobacco use: {value_display}")
                    elif any("social-determinant" in p.lower() for p in profiles):
                        social_determinants.append(f"{display}: {value_display}")

            elif resource_type == "Procedure":
                # Extract procedure information with dates
                code_info = resource.get("code", {}).get("coding", [{}])[0]
                display = code_info.get("display")
                performed_date = resource.get("performedDateTime") or resource.get("performedPeriod", {}).get("start")
                if display:
                    simple_date = self._format_date_simple(performed_date) if performed_date else None
                    date_info = f" (performed {simple_date})" if simple_date else ""
                    procedures.append(f"{display}{date_info}")

            elif resource_type == "MedicationStatement" or resource_type == "MedicationRequest":
                # Extract medication information
                medication_info = resource.get("medicationCodeableConcept", {}).get("coding", [{}])[0]
                display = medication_info.get("display")
                if display:
                    medications.append(display)

        # Build comprehensive clinical note narrative following proper clinical note structure
        # Priority: Demographics (already in first sentence), Diagnosis, Staging, Biomarkers, Procedures, Medications, Social Determinants, Other Conditions, Cause of Death

        # PRIMARY DIAGNOSIS (clinically critical - include dates)
        if primary_diagnosis:
            # Remove needless parentheses from qualifiers
            clean_diagnosis = primary_diagnosis.replace(' (disorder)', ' disorder').replace(' (procedure)', ' procedure').replace(' (finding)', ' finding')
            code_str = ""
            if diagnosis_info.get('system') and diagnosis_info.get('code'):
                system_clean = self._format_mcode_display('', diagnosis_info['system'], diagnosis_info['code'])
                # Extract just the system:code part without parentheses
                if system_clean.startswith('(') and ':' in system_clean:
                    system_clean = system_clean[1:-1]  # Remove parentheses
                code_str = f"; {system_clean}"
            simple_dates = [self._format_date_simple(d) for d in set(diagnosis_info['dates'])]
            dates_str = f" documented on {', '.join(simple_dates)}" if diagnosis_info['dates'] else ""
            clinical_note.append(f"She has been diagnosed with {clean_diagnosis} (mCODE: CancerCondition{code_str}{dates_str}).")

        # TUMOR STAGING (clinically critical - collapse duplicate dates and codes)
        if tnm_components:
            # Group staging components by date for consolidation
            date_groups = {}
            stage_value = None

            for component in ['T', 'N', 'M']:
                if component in tnm_components:
                    comp_data = tnm_components[component]
                    date_key = tuple(sorted(set(comp_data['dates']))) if comp_data['dates'] else ()
                    if date_key not in date_groups:
                        date_groups[date_key] = {'codes': [], 'values': []}
                    date_groups[date_key]['values'].append(comp_data['value'])
                    if comp_data.get('system') and comp_data.get('code'):
                        system_clean = self._format_mcode_display('', comp_data['system'], comp_data['code'])
                        # Extract just the system:code part without parentheses
                        if system_clean.startswith('(') and ':' in system_clean:
                            system_clean = system_clean[1:-1]  # Remove parentheses
                        date_groups[date_key]['codes'].append(system_clean)

            # Handle overall stage separately
            if 'stage' in tnm_components:
                stage_data = tnm_components['stage']
                stage_value = stage_data['value']
                stage_date_key = tuple(sorted(set(stage_data['dates']))) if stage_data['dates'] else ()
                if stage_date_key not in date_groups:
                    date_groups[stage_date_key] = {'codes': [], 'values': []}
                if stage_data.get('system') and stage_data.get('code'):
                    system_clean = self._format_mcode_display('', stage_data['system'], stage_data['code'])
                    # Extract just the system:code part without parentheses
                    if system_clean.startswith('(') and ':' in system_clean:
                        system_clean = system_clean[1:-1]  # Remove parentheses
                    date_groups[stage_date_key]['codes'].append(system_clean)

            # Generate consolidated staging sentences with inline codes
            for date_key, data in date_groups.items():
                if data['values']:  # TNM components
                    simple_dates = [self._format_date_simple(d) for d in date_key]
                    dates_str = f" documented on {', '.join(simple_dates)}" if date_key else ""
                    # Create inline format: T4 (SNOMED:65565005) N1 (SNOMED:53623008) M1 (SNOMED:55440008)
                    components_with_codes = []
                    for i, value in enumerate(data['values']):
                        if i < len(data['codes']):
                            components_with_codes.append(f"{value} ({data['codes'][i]})")
                        else:
                            components_with_codes.append(value)
                    components_str = ' '.join(components_with_codes[:-1])
                    if len(components_with_codes) > 1:
                        components_str += f", and {components_with_codes[-1]}"
                    else:
                        components_str = components_with_codes[0]
                    clinical_note.append(f"Her tumor staging (mCODE: TNMStageGroup{dates_str}) is {components_str}.")
                elif stage_value and not data['values']:  # Overall stage only
                    codes_str = f"; {', '.join(data['codes'][:-1])}, and {data['codes'][-1]}" if len(data['codes']) > 1 else f"; {data['codes'][0]}" if data['codes'] else ""
                    simple_dates = [self._format_date_simple(d) for d in date_key]
                    dates_str = f" documented on {', '.join(simple_dates)}" if date_key else ""
                    clinical_note.append(f"She has {stage_value} disease (mCODE: TNMStageGroup{codes_str}{dates_str}).")

        # BIOMARKERS (clinically critical - collapse duplicate dates and codes)
        if biomarkers:
            # Group biomarkers by date for consolidation
            date_groups = {}
            for marker, data in biomarkers.items():
                date_key = tuple(sorted(set(data['dates']))) if data['dates'] else ()
                if date_key not in date_groups:
                    date_groups[date_key] = {'markers': [], 'codes': []}
                marker_info = f"{marker} {data['status']}"
                date_groups[date_key]['markers'].append(marker_info)
                if data.get('system') and data.get('code'):
                    system_clean = self._format_mcode_display('', data['system'], data['code'])
                    # Extract just the system:code part without parentheses
                    if system_clean.startswith('(') and ':' in system_clean:
                        system_clean = system_clean[1:-1]  # Remove parentheses
                    date_groups[date_key]['codes'].append(system_clean)

            # Generate consolidated biomarker sentences with inline codes
            for date_key, data in date_groups.items():
                if data['markers']:
                    simple_dates = [self._format_date_simple(d) for d in date_key]
                    dates_str = f" documented on {', '.join(simple_dates)}" if date_key else ""
                    # Create inline format: ER positive (SNOMED:12345) PR negative (SNOMED:67890) HER2 negative (SNOMED:54321)
                    markers_with_codes = []
                    for i, marker in enumerate(data['markers']):
                        if i < len(data['codes']):
                            markers_with_codes.append(f"{marker} ({data['codes'][i]})")
                        else:
                            markers_with_codes.append(marker)
                    markers_text = ", ".join(markers_with_codes[:-1])
                    if len(markers_with_codes) > 1:
                        markers_text += f", and {markers_with_codes[-1]}"
                    else:
                        markers_text = markers_with_codes[0]
                    clinical_note.append(f"Her tumor markers (mCODE: TumorMarkerTest{dates_str}) show {markers_text}.")

        # PROCEDURES (clinically relevant - include dates for major procedures only)
        if procedures:
            # Filter to only clinically significant procedures (exclude administrative ones like "Medication Reconciliation")
            significant_procedures = []
            for proc in procedures:
                # Include procedures that are clinically relevant (diagnostics, treatments, surgeries)
                if any(keyword in proc.lower() for keyword in [
                    'biopsy', 'chemotherapy', 'surgery', 'radiation', 'mammogram', 'ultrasonography',
                    'mri', 'ct', 'pet', 'bone scan', 'lumpectomy', 'mastectomy', 'reconstruction',
                    'immunotherapy', 'hormone therapy', 'targeted therapy', 'clinical trial'
                ]):
                    significant_procedures.append(proc)

            if significant_procedures:
                unique_procedures = list(set(significant_procedures))  # Remove duplicates
                if len(unique_procedures) == 1:
                    clinical_note.append(f"She underwent {unique_procedures[0]}.")
                else:
                    clinical_note.append(f"She underwent {', '.join(unique_procedures[:-1])}, and {unique_procedures[-1]}.")

        # MEDICATIONS (clinically relevant - no dates needed)
        if medications:
            unique_medications = list(set(medications))  # Remove duplicates
            if len(unique_medications) == 1:
                clinical_note.append(f"She is prescribed {unique_medications[0]}.")
            else:
                clinical_note.append(f"She is prescribed {', '.join(unique_medications[:-1])}, and {unique_medications[-1]}.")

        # SOCIAL DETERMINANTS (clinically relevant - no dates needed)
        if social_determinants:
            unique_social = list(set(social_determinants))  # Remove duplicates
            if len(unique_social) == 1:
                clinical_note.append(f"Social determinant: {unique_social[0]}.")
            else:
                clinical_note.append(f"Social determinants include {', '.join(unique_social[:-1])}, and {unique_social[-1]}.")

        # OTHER CONDITIONS (clinically relevant - include dates for significant conditions)
        if other_conditions:
            # Filter to significant conditions (exclude minor or administrative)
            significant_conditions = []
            for cond in other_conditions:
                # Include conditions that are clinically significant
                if not any(minor in cond.lower() for minor in [
                    'medication reconciliation', 'notifications', 'admission to orthopedic',
                    'patient discharge', 'certification procedure', 'initial patient assessment'
                ]):
                    significant_conditions.append(cond)

            if significant_conditions:
                if len(significant_conditions) == 1:
                    clinical_note.append(f"She also has {significant_conditions[0]}.")
                else:
                    clinical_note.append(f"She has other conditions including {', '.join(significant_conditions[:-1])}, and {significant_conditions[-1]}.")

        # CAUSE OF DEATH (clinically critical - include dates)
        if cause_of_death:
            # Remove needless parentheses from qualifiers
            clean_cause = cause_of_death.replace(' (disorder)', ' disorder').replace(' (procedure)', ' procedure').replace(' (finding)', ' finding')
            code_str = ""
            if cause_of_death_info.get('system') and cause_of_death_info.get('code'):
                system_clean = self._format_mcode_display('', cause_of_death_info['system'], cause_of_death_info['code'])
                # Extract just the system:code part without parentheses
                if system_clean.startswith('(') and ':' in system_clean:
                    system_clean = system_clean[1:-1]  # Remove parentheses
                code_str = f"; {system_clean}"
            simple_dates = [self._format_date_simple(d) for d in set(cause_of_death_info['dates'])]
            dates_str = f" documented on {', '.join(simple_dates)}" if cause_of_death_info['dates'] else ""
            clinical_note.append(f"Her cause of death was {clean_cause} (mCODE: CauseOfDeath{code_str}{dates_str}).")

        return " ".join(clinical_note)
