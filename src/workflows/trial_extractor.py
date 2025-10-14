from typing import Any, Dict


class TrialExtractor:

    def extract_trial_mcode_elements(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        protocol_section = trial["protocolSection"]

        mcode_elements: Dict[str, Any] = {}

        extractors = [
            ("identificationModule", self._extract_trial_identification),
            ("eligibilityModule", self._extract_trial_eligibility_mcode),
            ("conditionsModule", self._extract_trial_conditions_mcode),
            ("armsInterventionsModule", self._extract_trial_interventions_mcode),
            ("designModule", self._extract_trial_design_mcode),
            ("statusModule", self._extract_trial_temporal_mcode),
            ("sponsorCollaboratorsModule", self._extract_trial_sponsor_mcode),
            ("outcomesModule", self._extract_trial_outcomes_mcode),
            ("armsInterventionsModule", self._extract_trial_arms_groups_mcode),
            ("locationsModule", self._extract_trial_locations_mcode),
            ("contactsLocationsModule", self._extract_trial_investigators_mcode),
        ]

        for module_key, extractor_func in extractors:
            module_data = protocol_section.get(module_key, {})
            mcode_elements.update(extractor_func(module_data))

        return mcode_elements

    def _extract_trial_identification(self, identification: Dict[str, Any]) -> Dict[str, Any]:
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
        elements = {}

        min_age = eligibility.get("minimumAge")
        max_age = eligibility.get("maximumAge")
        if min_age or max_age:
            elements["TrialAgeCriteria"] = {
                "minimumAge": min_age,
                "maximumAge": max_age,
                "ageUnit": "Years",
            }

        sex = eligibility.get("sex")
        if sex:
            elements["TrialSexCriteria"] = {
                "system": "http://hl7.org/fhir/administrative-gender",
                "code": sex.lower(),
                "display": sex.capitalize(),
            }

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

        criteria_text = eligibility.get("eligibilityCriteria")
        if criteria_text:
            elements["TrialEligibilityCriteria"] = {
                "text": criteria_text,
                "display": "Detailed eligibility criteria for patient matching",
            }

        return elements

    def _extract_trial_conditions_mcode(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        elements = {}

        condition_list = conditions.get("conditions", [])
        if condition_list:
            cancer_conditions = []
            comorbid_conditions = []

            for condition in condition_list:
                if isinstance(condition, dict):
                    condition_name = condition.get("name", "").lower()
                    condition_code = condition.get("code", "Unknown")
                elif isinstance(condition, str):
                    condition_name = condition.lower()
                    condition_code = "Unknown"
                else:
                    continue

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
                            "display": condition_name or "Unknown cancer condition",
                            "interpretation": "Confirmed",
                        }
                    )
                else:
                    comorbid_conditions.append(
                        {
                            "system": "http://snomed.info/sct",
                            "code": condition_code,
                            "display": condition_name or "Unknown condition",
                        }
                    )

            if cancer_conditions:
                elements["TrialCancerConditions"] = cancer_conditions
            if comorbid_conditions:
                elements["TrialComorbidConditions"] = comorbid_conditions

        return elements

    def _extract_trial_interventions_mcode(self, interventions: Dict[str, Any]) -> Dict[str, Any]:
        elements = {}

        intervention_list = interventions.get("interventions", [])
        if intervention_list:
            medication_interventions = []
            other_interventions = []

            for intervention in intervention_list:
                if not isinstance(intervention, dict):
                    continue

                intervention_type = intervention.get("type", "").lower()
                intervention_name = intervention.get("name", "")
                description = intervention.get("description", "")
                dosage = intervention.get("arm_group_labels", [])
                schedule = intervention.get("other_names", [])

                if intervention_type in ["drug", "biological", "device"]:
                    medication_interventions.append(
                        {
                            "system": "http://snomed.info/sct",
                            "code": "Unknown",
                            "display": intervention_name,
                            "description": description,
                            "interventionType": intervention_type,
                            "dosage": dosage,
                            "schedule": schedule,
                        }
                    )
                else:
                    other_interventions.append(
                        {
                            "display": intervention_name,
                            "description": description,
                            "interventionType": intervention_type,
                            "dosage": dosage,
                            "schedule": schedule,
                        }
                    )

            if medication_interventions:
                elements["TrialMedicationInterventions"] = medication_interventions
            if other_interventions:
                elements["TrialOtherInterventions"] = other_interventions

        return elements

    def _extract_trial_design_mcode(self, design: Dict[str, Any]) -> Dict[str, Any]:
        elements = {}

        study_type = design.get("studyType")
        if study_type:
            elements["TrialStudyType"] = {
                "display": study_type,
                "code": study_type.lower().replace(" ", "_"),
            }

        phase = design.get("phases", [])
        if phase:
            elements["TrialPhase"] = {
                "display": ", ".join(phase),
                "phases": phase,
            }

        primary_purpose = design.get("primaryPurpose")
        if primary_purpose:
            elements["TrialPrimaryPurpose"] = {
                "display": primary_purpose,
                "code": primary_purpose.lower().replace(" ", "_"),
            }

        randomization = design.get("designInfo", {}).get("randomization", {})
        if randomization:
            elements["TrialRandomization"] = {
                "method": randomization.get("method"),
                "methodDetail": randomization.get("methodDetail"),
            }

        masking = design.get("designInfo", {}).get("maskingInfo", {})
        if masking:
            elements["TrialBlinding"] = {
                "masking": masking.get("masking"),
                "maskingDescription": masking.get("maskingDescription"),
                "whoMasked": masking.get("whoMasked", []),
            }

        enrollment_info = design.get("enrollmentInfo", {})
        if enrollment_info:
            elements["TrialEnrollment"] = {
                "count": enrollment_info.get("count"),
                "type": enrollment_info.get("type"),
            }

        return elements

    def _extract_trial_temporal_mcode(self, status: Dict[str, Any]) -> Dict[str, Any]:
        elements = {}

        overall_status = status.get("overallStatus")
        if overall_status:
            elements["TrialStatus"] = {
                "display": overall_status,
                "code": overall_status.lower().replace(" ", "_"),
            }

        start_date = status.get("startDateStruct", {}).get("date")
        if start_date:
            elements["TrialStartDate"] = {
                "date": start_date,
                "display": f"Trial started on {start_date}",
            }

        completion_date = status.get("completionDateStruct", {}).get("date")
        if completion_date:
            elements["TrialCompletionDate"] = {
                "date": completion_date,
                "display": f"Trial completed on {completion_date}",
            }

        primary_completion_date = status.get("primaryCompletionDateStruct", {}).get("date")
        if primary_completion_date:
            elements["TrialPrimaryCompletionDate"] = {
                "date": primary_completion_date,
                "display": f"Primary completion on {primary_completion_date}",
            }

        return elements

    def _extract_trial_sponsor_mcode(self, sponsor: Dict[str, Any]) -> Dict[str, Any]:
        elements = {}

        lead_sponsor = sponsor.get("leadSponsor", {})
        if lead_sponsor:
            elements["TrialLeadSponsor"] = {
                "name": lead_sponsor.get("name"),
                "class": lead_sponsor.get("class"),
            }

        responsible_party = sponsor.get("responsibleParty", {})
        if responsible_party:
            elements["TrialResponsibleParty"] = {
                "name": responsible_party.get("name"),
                "type": responsible_party.get("type"),
                "affiliation": responsible_party.get("affiliation"),
            }

        return elements

    def _extract_trial_outcomes_mcode(self, outcomes: Dict[str, Any]) -> Dict[str, Any]:
        elements = {}

        primary_outcomes = outcomes.get("primaryOutcomes", [])
        if primary_outcomes:
            primary_list = []
            for outcome in primary_outcomes:
                if isinstance(outcome, dict):
                    primary_list.append({
                        "measure": outcome.get("measure", ""),
                        "timeFrame": outcome.get("timeFrame", ""),
                        "description": outcome.get("description", ""),
                        "type": "primary"
                    })
            if primary_list:
                elements["TrialPrimaryOutcomes"] = primary_list

        secondary_outcomes = outcomes.get("secondaryOutcomes", [])
        if secondary_outcomes:
            secondary_list = []
            for outcome in secondary_outcomes:
                if isinstance(outcome, dict):
                    secondary_list.append({
                        "measure": outcome.get("measure", ""),
                        "timeFrame": outcome.get("timeFrame", ""),
                        "description": outcome.get("description", ""),
                        "type": "secondary"
                    })
            if secondary_list:
                elements["TrialSecondaryOutcomes"] = secondary_list

        return elements

    def _extract_trial_arms_groups_mcode(self, arms_interventions: Dict[str, Any]) -> Dict[str, Any]:
        elements = {}

        arms_groups = arms_interventions.get("armGroups", [])
        if arms_groups:
            arms_list = []
            for arm in arms_groups:
                if isinstance(arm, dict):
                    arm_data = {
                        "label": arm.get("label", ""),
                        "type": arm.get("type", ""),
                        "description": arm.get("description", ""),
                        "interventionNames": arm.get("interventionNames", [])
                    }
                    arms_list.append(arm_data)
            if arms_list:
                elements["TrialArmsGroups"] = arms_list

        return elements

    def _extract_trial_locations_mcode(self, locations: Dict[str, Any]) -> Dict[str, Any]:
        elements = {}

        location_list = locations.get("locations", [])
        if location_list:
            locations_data = []
            for location in location_list:
                if isinstance(location, dict):
                    location_data = {
                        "facility": location.get("facility", ""),
                        "city": location.get("city", ""),
                        "state": location.get("state", ""),
                        "zip": location.get("zip", ""),
                        "country": location.get("country", ""),
                        "status": location.get("status", ""),
                        "contacts": location.get("contacts", [])
                    }
                    locations_data.append(location_data)
            if locations_data:
                elements["TrialLocations"] = locations_data

        return elements

    def _extract_trial_investigators_mcode(self, contacts_locations: Dict[str, Any]) -> Dict[str, Any]:
        elements = {}

        overall_officials = contacts_locations.get("overallOfficials", [])
        if overall_officials:
            investigators = []
            for official in overall_officials:
                if isinstance(official, dict):
                    investigator_data = {
                        "name": official.get("name", ""),
                        "role": official.get("role", ""),
                        "affiliation": official.get("affiliation", "")
                    }
                    investigators.append(investigator_data)
            if investigators:
                elements["TrialInvestigators"] = investigators

        return elements

    def extract_trial_metadata(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}

        protocol_section = trial["protocolSection"]

        identification = protocol_section.get("identificationModule", {})
        metadata["nct_id"] = identification.get("nctId")
        metadata["brief_title"] = identification.get("briefTitle")
        metadata["official_title"] = identification.get("officialTitle")

        status = protocol_section.get("statusModule", {})
        metadata["overall_status"] = status.get("overallStatus")
        start_struct = status.get("startDateStruct", {})
        metadata["start_date"] = start_struct.get("date")
        completion_struct = status.get("completionDateStruct", {})
        metadata["completion_date"] = completion_struct.get("date")

        design = protocol_section.get("designModule", {})
        metadata["study_type"] = design.get("studyType")
        metadata["phase"] = design.get("phases", [])
        metadata["primary_purpose"] = design.get("primaryPurpose")

        eligibility = protocol_section.get("eligibilityModule", {})
        metadata["minimum_age"] = eligibility.get("minimumAge")
        metadata["maximum_age"] = eligibility.get("maximumAge")
        metadata["sex"] = eligibility.get("sex")
        metadata["healthy_volunteers"] = eligibility.get("healthyVolunteers")

        conditions_module = protocol_section.get("conditionsModule", {})
        conditions = conditions_module.get("conditions", [])
        metadata["conditions"] = [c.get("name") for c in conditions if isinstance(c, dict)]

        interventions_module = protocol_section.get("armsInterventionsModule", {})
        interventions = interventions_module.get("interventions", [])
        metadata["interventions"] = [i.get("name") for i in interventions if isinstance(i, dict)]

        return metadata

    def extract_trial_id(self, trial: Dict[str, Any]) -> str:
        trial_id = trial["protocolSection"]["identificationModule"]["nctId"]
        if isinstance(trial_id, str):
            return trial_id
        raise ValueError(f"Trial ID must be string, got {type(trial_id)}")

    def check_trial_has_full_data(self, trial: Dict[str, Any]) -> bool:
        protocol_section = trial["protocolSection"]

        indicators = []

        eligibility = protocol_section.get("eligibilityModule", {})
        criteria = eligibility.get("eligibilityCriteria", "")
        if criteria and len(criteria) > 100:
            indicators.append(True)

        arms = protocol_section.get("armsInterventionsModule", {})
        interventions = arms.get("interventions", [])
        if interventions and len(interventions) > 0:
            detailed_interventions = any(
                isinstance(i, dict) and i.get("description", "") for i in interventions
            )
            if detailed_interventions:
                indicators.append(True)

        outcomes = protocol_section.get("outcomesModule", {})
        if isinstance(outcomes, dict) and outcomes.get("primaryOutcomes"):
            indicators.append(True)

        derived_section = trial.get("derivedSection")
        if derived_section and isinstance(derived_section, dict):
            indicators.append(True)

        sponsor = protocol_section.get("sponsorCollaboratorsModule", {})
        collaborators = sponsor.get("collaborators", [])
        if collaborators and len(collaborators) > 0:
            indicators.append(True)

        return len(indicators) >= 3
