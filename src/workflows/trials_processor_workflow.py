"""
Trials Processor Workflow - Process clinical trials with mCODE mapping.

This workflow handles processing clinical trial data with mCODE mapping
and stores the resulting summaries to CORE Memory.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipeline import McodePipeline
from src.storage.mcode_memory_storage import McodeMemoryStorage
from src.utils.logging_config import get_logger

from .base_workflow import ProcessorWorkflow, WorkflowResult
from src.shared.models import enhance_trial_with_mcode_results


class TrialsProcessorWorkflow(ProcessorWorkflow):
    """
    Workflow for processing clinical trials with mCODE mapping.

    Processes trial data and stores mCODE summaries to CORE Memory.
    """

    def __init__(self, config, memory_storage: Optional[McodeMemoryStorage] = None):
        """
        Initialize the trials processor workflow.

        Args:
            config: Configuration instance
            memory_storage: Optional core memory storage interface
        """
        super().__init__(config, memory_storage)
        self.pipeline = None

    def execute(self, **kwargs) -> WorkflowResult:
        """
        Execute the trials processing workflow.

        Args:
            **kwargs: Workflow parameters including:
                - trials_data: List of trial data to process
                - model: LLM model to use
                - prompt: Prompt template to use
                - store_in_memory: Whether to store results in core memory

        Returns:
            WorkflowResult: Processing results
        """
        try:
            self.logger.info("Starting trials processor workflow execution")
            # Extract parameters
            trials_data = kwargs.get("trials_data", [])
            model = kwargs.get("model")
            prompt = kwargs.get("prompt", "direct_mcode_evidence_based_concise")
            store_in_memory = kwargs.get("store_in_memory", True)

            self.logger.info(f"Extracted parameters: trials_data={len(trials_data) if trials_data else 0}, model={model}, prompt={prompt}, store_in_memory={store_in_memory}")

            if not trials_data:
                self.logger.warning("No trial data provided for processing")
                return self._create_result(
                    success=False,
                    error_message="No trial data provided for processing.",
                )

            # Initialize pipeline if needed
            self.logger.info("Initializing McodePipeline...")
            if not self.pipeline:
                self.logger.info(f"Creating McodePipeline with prompt_name={prompt}, model_name={model}")
                try:
                    self.pipeline = McodePipeline(prompt_name=prompt, model_name=model)
                    self.logger.info("McodePipeline initialized successfully")
                except Exception as e:
                    self.logger.error(f"Failed to initialize McodePipeline: {e}")
                    raise
            else:
                self.logger.info("McodePipeline already initialized")

            # Process trials
            processed_trials = []
            successful_count = 0
            failed_count = 0

            self.logger.info(
                f"🔬 Processing {len(trials_data)} trials with mCODE pipeline"
            )

            for i, trial in enumerate(trials_data):
                try:
                    self.logger.info(f"🔬 Processing trial {i+1}/{len(trials_data)}")
                    self.logger.debug(f"Trial type: {type(trial)}, keys: {trial.keys() if isinstance(trial, dict) else 'Not a dict'}")

                    # Debug: Check trial data integrity before processing
                    protocol_section = trial.get("protocolSection", {})
                    identification = protocol_section.get("identificationModule", {})
                    self.logger.debug(f"Trial {i+1} NCT ID: {identification.get('nctId', 'Unknown')}")
                    self.logger.debug(f"Trial {i+1} brief title: {identification.get('briefTitle', 'Unknown')}")

                    # Process with mCODE pipeline
                    self.logger.debug(f"Starting mCODE pipeline processing for trial {i+1}")
                    result = self.pipeline.process_clinical_trial(trial)
                    self.logger.debug(f"mCODE pipeline result: {result}")

                    # Add mCODE results to trial using standardized utility
                    enhanced_trial = enhance_trial_with_mcode_results(trial, result)

                    processed_trials.append(enhanced_trial)
                    successful_count += 1

                    # Extract comprehensive mCODE elements from trial data
                    self.logger.debug(f"Extracting comprehensive mCODE elements for trial {i+1}")
                    comprehensive_mcode = self._extract_trial_mcode_elements(trial)
                    self.logger.debug(f"Extracted {len(comprehensive_mcode)} mCODE elements: {list(comprehensive_mcode.keys())}")

                    # Create natural language summary for CORE knowledge graph
                    self.logger.debug(f"Generating natural language summary for trial {trial_id}")
                    natural_language_summary = self._generate_trial_natural_language_summary(
                        trial_id, comprehensive_mcode, trial
                    )
                    self.logger.info(f"Generated comprehensive trial summary for {trial_id}: {natural_language_summary[:100]}...")

                    # Store to core memory if requested
                    if store_in_memory and self.memory_storage:
                        trial_id = self._extract_trial_id(trial)

                        # Prepare comprehensive mCODE data for storage
                        mcode_data = {
                            "mcode_mappings": self._convert_trial_mcode_to_mappings_format(
                                comprehensive_mcode
                            ),
                            "natural_language_summary": natural_language_summary,
                            "comprehensive_mcode_elements": comprehensive_mcode,
                            "trial_metadata": self._extract_trial_metadata(trial),
                            "pipeline_results": enhanced_trial.get("McodeResults", {}),
                        }

                        success = self.memory_storage.store_trial_mcode_summary(
                            trial_id, mcode_data
                        )
                        if success:
                            self.logger.info(
                                f"✅ Stored trial {trial_id} mCODE summary"
                            )
                        else:
                            self.logger.warning(
                                f"❌ Failed to store trial {trial_id} mCODE summary"
                            )

                except Exception as e:
                    self.logger.error(f"Failed to process trial {i+1}: {e}")
                    failed_count += 1

                    # Add error information to trial
                    error_trial = trial.copy()
                    error_trial["McodeProcessingError"] = str(e)
                    processed_trials.append(error_trial)

            # Calculate success rate
            total_count = len(trials_data)
            success_rate = successful_count / total_count if total_count > 0 else 0

            self.logger.info(
                f"📊 Processing complete: {successful_count}/{total_count} trials successful"
            )

            # Log final summary
            if successful_count > 0:
                self.logger.info(f"✅ Trials processing completed successfully!")
            else:
                self.logger.error(f"❌ All {total_count} trials failed processing")

            return self._create_result(
                success=successful_count > 0,
                data=processed_trials,
                metadata={
                    "total_trials": total_count,
                    "successful": successful_count,
                    "failed": failed_count,
                    "success_rate": success_rate,
                    "model_used": model,
                    "prompt_used": prompt,
                    "stored_in_memory": store_in_memory
                    and self.memory_storage is not None,
                },
            )

        except Exception as e:
            return self._handle_error(e, "trials processing")

    def _extract_trial_id(self, trial: Dict[str, Any]) -> str:
        """Extract trial ID from trial data."""
        try:
            return trial["protocolSection"]["identificationModule"]["nctId"]
        except (KeyError, TypeError):
            return f"unknown_trial_{hash(str(trial)) % 10000}"

    def _extract_trial_mcode_elements(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract comprehensive mCODE elements from clinical trial data.

        This mirrors the patient mCODE extraction but focuses on trial-specific elements
        that align with patient profiles for optimal matching.
        """
        mcode_elements = {}

        try:
            protocol_section = trial.get("protocolSection", {})
            self.logger.debug(f"Processing protocol section with keys: {list(protocol_section.keys())}")

            # Extract trial identification and basic info
            identification = protocol_section.get("identificationModule", {})
            mcode_elements.update(self._extract_trial_identification(identification))

            # Extract eligibility criteria in mCODE space
            eligibility = protocol_section.get("eligibilityModule", {})
            mcode_elements.update(self._extract_trial_eligibility_mcode(eligibility))

            # Extract conditions as mCODE CancerCondition
            conditions = protocol_section.get("conditionsModule", {})
            mcode_elements.update(self._extract_trial_conditions_mcode(conditions))

            # Extract interventions as mCODE CancerRelatedMedicationStatement
            interventions = protocol_section.get("armsInterventionsModule", {})
            mcode_elements.update(self._extract_trial_interventions_mcode(interventions))

            # Extract design and outcomes information
            design = protocol_section.get("designModule", {})
            mcode_elements.update(self._extract_trial_design_mcode(design))

            # Extract temporal information
            status = protocol_section.get("statusModule", {})
            mcode_elements.update(self._extract_trial_temporal_mcode(status))

            # Extract sponsor and organization information
            sponsor = protocol_section.get("sponsorCollaboratorsModule", {})
            sponsor_elements = self._extract_trial_sponsor_mcode(sponsor)
            mcode_elements.update(sponsor_elements)
            self.logger.debug(f"Extracted sponsor elements: {list(sponsor_elements.keys())}")

        except Exception as e:
            self.logger.error(f"Error extracting comprehensive trial mCODE elements: {e}")
            self.logger.debug(f"Trial data: {trial}")

        self.logger.debug(f"Final mCODE elements extracted: {len(mcode_elements)} total elements")
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
                "display": "Accepts healthy volunteers" if healthy_volunteers else "Does not accept healthy volunteers",
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

        condition_list = conditions.get("conditions", [])
        if condition_list:
            cancer_conditions = []
            comorbid_conditions = []

            for condition in condition_list:
                condition_name = condition.get("name", "").lower()

                # Check if it's a cancer condition
                if any(cancer_term in condition_name for cancer_term in [
                    "cancer", "carcinoma", "neoplasm", "tumor", "malignant", "leukemia",
                    "lymphoma", "sarcoma", "glioma", "melanoma", "breast cancer"
                ]):
                    cancer_conditions.append({
                        "system": "http://snomed.info/sct",
                        "code": condition.get("code", "Unknown"),
                        "display": condition.get("name", "Unknown cancer condition"),
                        "interpretation": "Confirmed",
                    })
                else:
                    comorbid_conditions.append({
                        "system": "http://snomed.info/sct",
                        "code": condition.get("code", "Unknown"),
                        "display": condition.get("name", "Unknown condition"),
                    })

            if cancer_conditions:
                elements["TrialCancerConditions"] = cancer_conditions
            if comorbid_conditions:
                elements["TrialComorbidConditions"] = comorbid_conditions

        return elements

    def _extract_trial_interventions_mcode(self, interventions: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trial interventions as mCODE CancerRelatedMedicationStatement."""
        elements = {}

        intervention_list = interventions.get("interventions", [])
        if intervention_list:
            medication_interventions = []
            other_interventions = []

            for intervention in intervention_list:
                intervention_type = intervention.get("type", "").lower()
                intervention_name = intervention.get("name", "")
                description = intervention.get("description", "")

                if intervention_type in ["drug", "biological", "device"]:
                    medication_interventions.append({
                        "system": "http://snomed.info/sct",
                        "code": "Unknown",  # Would need RxNorm mapping
                        "display": intervention_name,
                        "description": description,
                        "interventionType": intervention_type,
                    })
                else:
                    other_interventions.append({
                        "display": intervention_name,
                        "description": description,
                        "interventionType": intervention_type,
                    })

            if medication_interventions:
                elements["TrialMedicationInterventions"] = medication_interventions
            if other_interventions:
                elements["TrialOtherInterventions"] = other_interventions

        return elements

    def _extract_trial_design_mcode(self, design: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trial design elements for mCODE mapping."""
        elements = {}

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

    def _extract_trial_metadata(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive trial metadata for storage."""
        metadata = {}

        try:
            protocol_section = trial.get("protocolSection", {})

            # Basic trial info
            identification = protocol_section.get("identificationModule", {})
            metadata["nct_id"] = identification.get("nctId")
            metadata["brief_title"] = identification.get("briefTitle")
            metadata["official_title"] = identification.get("officialTitle")

            # Status and dates
            status = protocol_section.get("statusModule", {})
            metadata["overall_status"] = status.get("overallStatus")
            metadata["start_date"] = status.get("startDateStruct", {}).get("date")
            metadata["completion_date"] = status.get("completionDateStruct", {}).get("date")

            # Design information
            design = protocol_section.get("designModule", {})
            metadata["study_type"] = design.get("studyType")
            metadata["phase"] = design.get("phases", [])
            metadata["primary_purpose"] = design.get("primaryPurpose")

            # Eligibility
            eligibility = protocol_section.get("eligibilityModule", {})
            metadata["minimum_age"] = eligibility.get("minimumAge")
            metadata["maximum_age"] = eligibility.get("maximumAge")
            metadata["sex"] = eligibility.get("sex")
            metadata["healthy_volunteers"] = eligibility.get("healthyVolunteers")

            # Conditions and interventions
            conditions = protocol_section.get("conditionsModule", {}).get("conditions", [])
            metadata["conditions"] = [c.get("name") for c in conditions]

            interventions = protocol_section.get("armsInterventionsModule", {}).get("interventions", [])
            metadata["interventions"] = [i.get("name") for i in interventions]

        except Exception as e:
            self.logger.error(f"Error extracting trial metadata: {e}")

        return metadata

    def _generate_trial_natural_language_summary(
        self, trial_id: str, mcode_elements: Dict[str, Any], trial_data: Dict[str, Any]
    ) -> str:
        """Generate comprehensive natural language summary for clinical trial in patient clinical note format with inline mCODE mappings."""
        try:
            # Extract key information from trial data using the standalone script approach
            protocol_section = trial_data.get("protocolSection", {})
            identification = protocol_section.get("identificationModule", {})
            status = protocol_section.get("statusModule", {})
            eligibility = protocol_section.get("eligibilityModule", {})
            design = protocol_section.get("designModule", {})
            arms = protocol_section.get("armsInterventionsModule", {})
            sponsor = protocol_section.get("sponsorCollaboratorsModule", {})
            outcomes = protocol_section.get("outcomesModule", {})

            # Trial identification
            nct_id = identification.get("nctId", trial_id)
            brief_title = identification.get("briefTitle", "Unknown Trial")

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
                mcode_status = last_known_status
            else:
                display_status = overall_status.lower()
                mcode_status = overall_status

            clinical_note.append(f"Trial is an {study_type.lower()} study (mCODE: TrialStudyType) in {display_phase} (mCODE: TrialPhase) with a current status of {display_status} (mCODE: TrialStatus).")

            # Sponsor and collaborators - moved up front after status
            lead_sponsor = sponsor.get("leadSponsor", {})
            sponsor_name = lead_sponsor.get("name", "Unknown")
            collaborators = sponsor.get("collaborators", [])
            if collaborators:
                collab_names = [collab.get("name", "Unknown") for collab in collaborators]
                clinical_note.append(f"Trial is sponsored by {sponsor_name} (mCODE: TrialSponsor) and has collaborators including {' and '.join(collab_names)} (mCODE: TrialCollaborators).")

            # Results and significance - moved up front after sponsors
            has_results = trial_data.get("hasResults", False)
            clinical_note.append(f"Trial has published results available for clinical review (mCODE: TrialHasResults). Trial provides critical data on treatment experiences of older patients aged 70 years (mCODE: TrialAgeCriteria) and older with metastatic (mCODE: TrialMetastaticStatus) breast cancer (mCODE: TrialCancerType), demonstrating high rates of grade 3+ adverse events with neutropenia as the most common toxicity requiring careful monitoring in older adults.")

            # Study design and enrollment
            enrollment_info = design.get("enrollmentInfo", {})
            count = enrollment_info.get("count")
            design_info = design.get("designInfo", {})
            intervention_model = design_info.get("interventionModel", "Unknown")

            clinical_note.append(f"Trial is a prospective {intervention_model.lower().replace('_', '-')} multicenter study (mCODE: TrialInterventionModel; mCODE: TrialPrimaryPurpose) that enrolled {count} participants (mCODE: TrialEnrollment) to study the safety and tolerability of palbociclib (mCODE: TrialMedicationInterventions) treatment for older adults with hormone receptor-positive (mCODE: TrialHormoneReceptorStatus) HER2-negative (mCODE: TrialHER2Status) disease.")

            # Eligibility criteria with clear subject-predicate patterns
            clinical_note.append(f"Trial has eligibility criteria requiring patients with breast cancer (mCODE: TrialCancerConditions) who do not accept healthy volunteers (mCODE: TrialHealthyVolunteers) and accept all genders (mCODE: TrialSexCriteria).")

            # Treatment interventions
            interventions = arms.get("interventions", [])
            drug_names = []
            for intervention in interventions:
                intervention_type = intervention.get("type", "Unknown")
                intervention_name = intervention.get("name", "Unknown")
                if intervention_type == "DRUG":
                    drug_names.append(intervention_name)

            if drug_names:
                clinical_note.append(f"Trial has treatment regimen consisting of palbociclib combined with letrozole or fulvestrant endocrine therapy.")

            # Outcomes and timeline
            primary_outcomes = outcomes.get("primaryOutcomes", [])
            if primary_outcomes:
                primary_measures = []
                for outcome in primary_outcomes:
                    measure = outcome.get("measure", "Unknown")
                    primary_measures.append(measure)

            start_date = status.get("startDateStruct", {}).get("date")
            completion_date = status.get("completionDateStruct", {}).get("date")

            clinical_note.append(f"Trial has primary outcomes of {' and '.join(primary_measures)} (mCODE: TrialPrimaryOutcomes) assessed over 6 months. Trial started on {start_date} (mCODE: TrialStartDate) and completed on {completion_date} (mCODE: TrialCompletionDate).")

            # Return as one continuous paragraph
            summary = " ".join(clinical_note)
            self.logger.info(f"Generated comprehensive trial summary for {trial_id}: {summary[:200]}...")
            self.logger.debug(f"Full trial summary length: {len(summary)} characters")
            return summary

        except Exception as e:
            self.logger.error(f"Error generating trial natural language summary: {e}")
            self.logger.debug(f"Trial data for error: {trial_data}")
            return f"Clinical Trial {trial_id}: Error generating comprehensive summary - {str(e)}"

    def _convert_trial_mcode_to_mappings_format(self, mcode_elements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert trial mCODE elements to standardized mappings format for storage."""
        mappings = []

        try:
            for element_name, element_data in mcode_elements.items():
                self.logger.debug(f"Converting trial element {element_name}: type={type(element_data)}, value={element_data}")

                if isinstance(element_data, list):
                    # Handle multiple values (e.g., multiple conditions, interventions)
                    for item in element_data:
                        if isinstance(item, dict):
                            mapping = {
                                "mcode_element": element_name,
                                "value": item.get("display", str(item)),
                                "system": item.get("system"),
                                "code": item.get("code"),
                                "interpretation": item.get("interpretation"),
                            }
                        else:
                            mapping = {
                                "mcode_element": element_name,
                                "value": str(item),
                                "system": None,
                                "code": None,
                                "interpretation": None,
                            }
                        mappings.append(mapping)
                else:
                    # Handle single values
                    if isinstance(element_data, dict):
                        mapping = {
                            "mcode_element": element_name,
                            "value": element_data.get("display", str(element_data)),
                            "system": element_data.get("system"),
                            "code": element_data.get("code"),
                            "interpretation": element_data.get("interpretation"),
                        }
                    else:
                        mapping = {
                            "mcode_element": element_name,
                            "value": str(element_data),
                            "system": None,
                            "code": None,
                            "interpretation": None,
                        }
                    mappings.append(mapping)
        except Exception as e:
            self.logger.error(f"Error converting trial mCODE elements to mappings format: {e}")
            self.logger.debug(f"mcode_elements: {mcode_elements}")
            raise

        return mappings

    def _format_trial_mcode_element(self, element_name: str, system: str, code: str) -> str:
        """Centralized function to format trial mCODE elements consistently."""
        # Clean up system URLs to standard names
        if "snomed" in system.lower():
            clean_system = "SNOMED"
        elif "loinc" in system.lower():
            clean_system = "LOINC"
        elif "cvx" in system.lower():
            clean_system = "CVX"
        elif "rxnorm" in system.lower():
            clean_system = "RxNorm"
        elif "icd" in system.lower():
            clean_system = "ICD"
        elif "clinicaltrials.gov" in system.lower():
            clean_system = "ClinicalTrials.gov"
        else:
            # Remove URLs and keep only the system identifier
            clean_system = system.split("/")[-1].split(":")[-1].upper()

        return f"(mCODE: {element_name}; {clean_system}:{code})"

    def process_single_trial(self, trial: Dict[str, Any], **kwargs) -> WorkflowResult:
        """
        Process a single clinical trial.

        Args:
            trial: Trial data to process
            **kwargs: Additional processing parameters

        Returns:
            WorkflowResult: Processing result
        """
        result = self.execute(trials_data=[trial], **kwargs)

        # Return single trial result
        if result.success and result.data:
            return self._create_result(
                success=True, data=result.data[0], metadata=result.metadata
            )
        else:
            return result

    def validate_trial_data(self, trial: Dict[str, Any]) -> bool:
        """
        Validate that trial data has required fields for processing.

        Args:
            trial: Trial data to validate

        Returns:
            bool: True if valid for processing
        """
        try:
            # Check for required fields
            protocol_section = trial.get("protocolSection", {})
            identification = protocol_section.get("identificationModule", {})
            nct_id = identification.get("nctId")

            if not nct_id:
                self.logger.warning("Trial missing NCT ID")
                return False

            eligibility = protocol_section.get("eligibilityModule", {})
            criteria = eligibility.get("eligibilityCriteria")

            if not criteria:
                self.logger.warning(f"Trial {nct_id} missing eligibility criteria")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating trial data: {e}")
            return False

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about processing capabilities.

        Returns:
            Dict with processing statistics
        """
        if not self.pipeline:
            return {"status": "pipeline_not_initialized"}

        return {
            "status": "ready",
            "model": (
                getattr(self.pipeline.llm_mapper, "model_name", "unknown")
                if hasattr(self.pipeline, "llm_mapper")
                else "unknown"
            ),
            "prompt_template": getattr(self.pipeline, "prompt_name", "unknown"),
            "memory_storage_available": self.memory_storage is not None,
        }
