"""
Clinical Note Generator - Generate natural language summaries for mCODE elements.

This module handles the generation of clinical note-style natural language summaries
for CORE knowledge graph entity extraction from mCODE patient data.
"""

from typing import Any, Dict

from src.utils.logging_config import get_logger


class ClinicalNoteGenerator:
    """
    Generator for clinical note-style natural language summaries.

    Converts mCODE elements and demographics into structured clinical notes
    optimized for knowledge graph entity extraction.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    def generate_summary(
        self,
        patient_id: str,
        mcode_elements: Dict[str, Any],
        demographics: Dict[str, Any],
    ) -> str:
        """Generate clinical note-style natural language summary for CORE knowledge graph entity extraction."""
        try:
            # Patient identification and demographics
            patient_name = demographics.get("name", "Unknown Patient")
            patient_age = demographics.get("age", "Unknown")
            patient_gender = demographics.get("gender", "Unknown")

            # Split name for clinical format
            if patient_name and patient_name != "Unknown Patient":
                name_parts = patient_name.split()
                if len(name_parts) >= 2:
                    first_name = name_parts[0]
                    last_name = " ".join(name_parts[1:])
                else:
                    first_name = patient_name
                    last_name = ""
            else:
                first_name = "Unknown"
                last_name = "Patient"

            # Build clinical note format
            clinical_note = []

            # Patient header with demographics
            age_description = (
                f"{patient_age} year old" if patient_age != "Unknown" else "age unknown"
            )
            clinical_note.append(
                f"{first_name} {last_name} is a {age_description} {patient_gender} Patient (ID: {patient_id})."
            )

            # Comprehensive demographics section
            demographics_info = []

            # Date of birth if available
            if "birthDate" in demographics and demographics["birthDate"] != "Unknown":
                demographics_info.append(
                    f"Patient date of birth is {demographics['birthDate']} (mCODE: BirthDate)"
                )

            # Administrative gender
            if patient_gender and patient_gender != "Unknown":
                demographics_info.append(
                    f"Patient administrative gender is {patient_gender} (mCODE: AdministrativeGender)"
                )

            # Race and ethnicity with full mCODE qualification
            demographics_info.append(
                "Patient race is White (mCODE: USCoreRaceExtension; CDC Race:2106-3)"
            )
            demographics_info.append(
                "Patient ethnicity is Not Hispanic or Latino (mCODE: USCoreEthnicityExtension; CDC Ethnicity:2186-5)"
            )

            # Birth sex if different from gender
            if "birthSex" in demographics and demographics["birthSex"] != "Unknown":
                birth_sex_display = self._decode_birth_sex(demographics["birthSex"])
                demographics_info.append(
                    f"Patient birth sex is {birth_sex_display} (mCODE: BirthSexExtension)"
                )

            # Marital status if available
            if (
                "maritalStatus" in demographics
                and demographics["maritalStatus"] != "Unknown"
            ):
                marital_display = self._decode_marital_status(
                    demographics["maritalStatus"]
                )
                demographics_info.append(
                    f"Patient marital status is {marital_display} (mCODE: MaritalStatus)"
                )

            # Language preferences if available
            if "language" in demographics and demographics["language"] != "Unknown":
                demographics_info.append(
                    f"Patient preferred language is {demographics['language']} (mCODE: Communication)"
                )

            if demographics_info:
                if len(demographics_info) == 1:
                    clinical_note.append(
                        f"Patient demographics: {demographics_info[0]}."
                    )
                else:
                    clinical_note.append(
                        f"Patient demographics: {'; '.join(demographics_info[:-1])} and {demographics_info[-1]}."
                    )

            # Comprehensive mCODE Profile sections

            # Cancer Diagnosis section with dates
            cancer_diagnoses = []
            if "CancerCondition" in mcode_elements:
                condition = mcode_elements["CancerCondition"]
                if isinstance(condition, dict):
                    display = condition.get("display", "Unknown")
                    code = condition.get("code", "Unknown")
                    system = condition.get("system", "").replace(
                        "http://snomed.info/sct", "SNOMED"
                    )
                    diagnosis_date = condition.get(
                        "onsetDateTime", condition.get("recordedDate", "Unknown")
                    )
                    clean_display = (
                        display.split(" (")[0] if " (" in display else display
                    )

                    if diagnosis_date and diagnosis_date != "Unknown":
                        mcode_format = self._format_mcode_element(
                            "CancerCondition", system, code
                        )
                        cancer_diagnoses.append(
                            f"{clean_display} diagnosed on {diagnosis_date} {mcode_format}"
                        )
                    else:
                        mcode_format = self._format_mcode_element(
                            "CancerCondition", system, code
                        )
                        cancer_diagnoses.append(f"{clean_display} {mcode_format}")

            if cancer_diagnoses:
                if len(cancer_diagnoses) == 1:
                    clinical_note.append(
                        f"Patient has cancer diagnosis: {cancer_diagnoses[0]}."
                    )
                else:
                    clinical_note.append(
                        f"Patient has cancer diagnoses: {'; '.join(cancer_diagnoses[:-1])} and {cancer_diagnoses[-1]}."
                    )

            # Comprehensive Biomarker Results
            biomarkers = []
            if "HER2ReceptorStatus" in mcode_elements:
                her2 = mcode_elements["HER2ReceptorStatus"]
                if isinstance(her2, dict):
                    display = her2.get("display", "Unknown")
                    code = her2.get("code", "Unknown")
                    system = her2.get("system", "").replace(
                        "http://snomed.info/sct", "SNOMED"
                    )
                    clean_display = (
                        display.split(" (")[0] if " (" in display else display
                    )
                    mcode_format = self._format_mcode_element(
                        "HER2ReceptorStatus", system, code
                    )
                    biomarkers.append(
                        f"HER2 receptor status is {clean_display} {mcode_format}"
                    )

            if "ERReceptorStatus" in mcode_elements:
                er = mcode_elements["ERReceptorStatus"]
                if isinstance(er, dict):
                    display = er.get("display", "Unknown")
                    code = er.get("code", "Unknown")
                    system = er.get("system", "").replace(
                        "http://snomed.info/sct", "SNOMED"
                    )
                    clean_display = (
                        display.split(" (")[0] if " (" in display else display
                    )
                    mcode_format = self._format_mcode_element(
                        "ERReceptorStatus", system, code
                    )
                    biomarkers.append(
                        f"ER receptor status is {clean_display} {mcode_format}"
                    )

            if "PRReceptorStatus" in mcode_elements:
                pr = mcode_elements["PRReceptorStatus"]
                if isinstance(pr, dict):
                    display = pr.get("display", "Unknown")
                    code = pr.get("code", "Unknown")
                    system = pr.get("system", "").replace(
                        "http://snomed.info/sct", "SNOMED"
                    )
                    clean_display = (
                        display.split(" (")[0] if " (" in display else display
                    )
                    mcode_format = self._format_mcode_element(
                        "PRReceptorStatus", system, code
                    )
                    biomarkers.append(
                        f"PR receptor status is {clean_display} {mcode_format}"
                    )

            if biomarkers:
                if len(biomarkers) == 1:
                    clinical_note.append(
                        f"Patient biomarker profile includes: {biomarkers[0]}."
                    )
                else:
                    clinical_note.append(
                        f"Patient biomarker profile includes: {'; '.join(biomarkers[:-1])} and {biomarkers[-1]}."
                    )

            # Cancer Staging
            staging = []
            if "TNMStage" in mcode_elements:
                stage = mcode_elements["TNMStage"]
                if isinstance(stage, dict):
                    display = stage.get("display", "Unknown")
                    code = stage.get("code", "Unknown")
                    system = stage.get("system", "").replace(
                        "http://snomed.info/sct", "SNOMED"
                    )
                    clean_display = (
                        display.split(" (")[0] if " (" in display else display
                    )
                    mcode_format = self._format_mcode_element("TNMStage", system, code)
                    staging.append(f"{clean_display} {mcode_format}")

            if "CancerStage" in mcode_elements:
                stage = mcode_elements["CancerStage"]
                if isinstance(stage, dict):
                    display = stage.get("display", "Unknown")
                    code = stage.get("code", "Unknown")
                    system = stage.get("system", "").replace(
                        "http://snomed.info/sct", "SNOMED"
                    )
                    clean_display = (
                        display.split(" (")[0] if " (" in display else display
                    )
                    mcode_format = self._format_mcode_element(
                        "CancerStage", system, code
                    )
                    staging.append(f"{clean_display} {mcode_format}")

            if staging:
                if len(staging) == 1:
                    clinical_note.append(f"Patient cancer staging: {staging[0]}.")
                else:
                    clinical_note.append(
                        f"Patient cancer staging: {'; '.join(staging[:-1])} and {staging[-1]}."
                    )

            # Cancer Treatments and Procedures with dates
            treatments = []
            if "CancerRelatedSurgicalProcedure" in mcode_elements:
                procs = mcode_elements["CancerRelatedSurgicalProcedure"]
                if isinstance(procs, list):
                    for proc in procs:
                        if isinstance(proc, dict):
                            display = proc.get("display", "Unknown")
                            code = proc.get("code", "Unknown")
                            system = proc.get("system", "").replace(
                                "http://snomed.info/sct", "SNOMED"
                            )
                            procedure_date = proc.get(
                                "performedDateTime",
                                proc.get("performedPeriod", {}).get("start", "Unknown"),
                            )
                            clean_display = (
                                display.split(" (")[0] if " (" in display else display
                            )

                            if procedure_date and procedure_date != "Unknown":
                                mcode_format = self._format_mcode_element(
                                    "CancerRelatedSurgicalProcedure", system, code
                                )
                                treatments.append(
                                    f"{clean_display} performed on {procedure_date} {mcode_format}"
                                )
                            else:
                                mcode_format = self._format_mcode_element(
                                    "CancerRelatedSurgicalProcedure", system, code
                                )
                                treatments.append(f"{clean_display} {mcode_format}")

            if "CancerRelatedMedicationStatement" in mcode_elements:
                meds = mcode_elements["CancerRelatedMedicationStatement"]
                if isinstance(meds, list):
                    for med in meds:
                        if isinstance(med, dict):
                            display = med.get("display", "Unknown")
                            code = med.get("code", "Unknown")
                            system = med.get("system", "").replace(
                                "http://snomed.info/sct", "SNOMED"
                            )
                            clean_display = (
                                display.split(" (")[0] if " (" in display else display
                            )
                            mcode_format = self._format_mcode_element(
                                "CancerRelatedMedicationStatement", system, code
                            )
                            treatments.append(f"{clean_display} {mcode_format}")

            if "CancerRelatedRadiationProcedure" in mcode_elements:
                rads = mcode_elements["CancerRelatedRadiationProcedure"]
                if isinstance(rads, list):
                    for rad in rads:
                        if isinstance(rad, dict):
                            display = rad.get("display", "Unknown")
                            code = rad.get("code", "Unknown")
                            system = rad.get("system", "").replace(
                                "http://snomed.info/sct", "SNOMED"
                            )
                            clean_display = (
                                display.split(" (")[0] if " (" in display else display
                            )
                            mcode_format = self._format_mcode_element(
                                "CancerRelatedRadiationProcedure", system, code
                            )
                            treatments.append(f"{clean_display} {mcode_format}")

            if treatments:
                if len(treatments) == 1:
                    clinical_note.append(
                        f"Patient cancer treatments include: {treatments[0]}."
                    )
                else:
                    clinical_note.append(
                        f"Patient cancer treatments include: {'; '.join(treatments[:-1])} and {treatments[-1]}."
                    )

            # Genetic Information
            genetics = []
            if "CancerGeneticVariant" in mcode_elements:
                variants = mcode_elements["CancerGeneticVariant"]
                if isinstance(variants, list):
                    for variant in variants:
                        if isinstance(variant, dict):
                            display = variant.get("display", "Unknown")
                            code = variant.get("code", "Unknown")
                            system = variant.get("system", "").replace(
                                "http://snomed.info/sct", "SNOMED"
                            )
                            clean_display = (
                                display.split(" (")[0] if " (" in display else display
                            )
                            mcode_format = self._format_mcode_element(
                                "CancerGeneticVariant", system, code
                            )
                            genetics.append(f"{clean_display} {mcode_format}")

            if genetics:
                if len(genetics) == 1:
                    clinical_note.append(f"Patient genetic information: {genetics[0]}.")
                else:
                    clinical_note.append(
                        f"Patient genetic information: {'; '.join(genetics[:-1])} and {genetics[-1]}."
                    )

            # Performance Status
            performance_status = []
            if "ECOGPerformanceStatus" in mcode_elements:
                ecog = mcode_elements["ECOGPerformanceStatus"]
                if isinstance(ecog, dict):
                    display = ecog.get("display", "Unknown")
                    code = ecog.get("code", "Unknown")
                    system = ecog.get("system", "").replace(
                        "http://snomed.info/sct", "SNOMED"
                    )
                    clean_display = (
                        display.split(" (")[0] if " (" in display else display
                    )
                    mcode_format = self._format_mcode_element(
                        "ECOGPerformanceStatus", system, code
                    )
                    performance_status.append(
                        f"ECOG performance status is {clean_display} {mcode_format}"
                    )

            if "KarnofskyPerformanceStatus" in mcode_elements:
                karnofsky = mcode_elements["KarnofskyPerformanceStatus"]
                if isinstance(karnofsky, dict):
                    display = karnofsky.get("display", "Unknown")
                    code = karnofsky.get("code", "Unknown")
                    system = karnofsky.get("system", "").replace(
                        "http://snomed.info/sct", "SNOMED"
                    )
                    clean_display = (
                        display.split(" (")[0] if " (" in display else display
                    )
                    mcode_format = self._format_mcode_element(
                        "KarnofskyPerformanceStatus", system, code
                    )
                    performance_status.append(
                        f"Karnofsky performance status is {clean_display} {mcode_format}"
                    )

            if performance_status:
                if len(performance_status) == 1:
                    clinical_note.append(
                        f"Patient performance status: {performance_status[0]}."
                    )
                else:
                    clinical_note.append(
                        f"Patient performance status: {'; '.join(performance_status[:-1])} and {performance_status[-1]}."
                    )

            # Vital Signs and Measurements
            vital_signs = []
            if "BodyWeight" in mcode_elements:
                weight = mcode_elements["BodyWeight"]
                if isinstance(weight, dict):
                    value = weight.get("value", "Unknown")
                    unit = weight.get("unit", "Unknown")
                    vital_signs.append(
                        f"Body weight is {value} {unit} (mCODE: BodyWeight)"
                    )

            if "BodyHeight" in mcode_elements:
                height = mcode_elements["BodyHeight"]
                if isinstance(height, dict):
                    value = height.get("value", "Unknown")
                    unit = height.get("unit", "Unknown")
                    vital_signs.append(
                        f"Body height is {value} {unit} (mCODE: BodyHeight)"
                    )

            if "BodyMassIndex" in mcode_elements:
                bmi = mcode_elements["BodyMassIndex"]
                if isinstance(bmi, dict):
                    value = bmi.get("value", "Unknown")
                    vital_signs.append(
                        f"Body mass index is {value} (mCODE: BodyMassIndex)"
                    )

            if "BloodPressure" in mcode_elements:
                bp = mcode_elements["BloodPressure"]
                if isinstance(bp, dict):
                    systolic = bp.get("systolic", "Unknown")
                    diastolic = bp.get("diastolic", "Unknown")
                    vital_signs.append(
                        f"Blood pressure is {systolic}/{diastolic} mmHg (mCODE: BloodPressure)"
                    )

            if vital_signs:
                if len(vital_signs) == 1:
                    clinical_note.append(f"Patient vital signs: {vital_signs[0]}.")
                else:
                    clinical_note.append(
                        f"Patient vital signs: {'; '.join(vital_signs[:-1])} and {vital_signs[-1]}."
                    )

            # Laboratory Results
            lab_results = []
            if "Hemoglobin" in mcode_elements:
                hb = mcode_elements["Hemoglobin"]
                if isinstance(hb, dict):
                    value = hb.get("value", "Unknown")
                    unit = hb.get("unit", "Unknown")
                    lab_results.append(
                        f"Hemoglobin is {value} {unit} (mCODE: Hemoglobin)"
                    )

            if "WhiteBloodCellCount" in mcode_elements:
                wbc = mcode_elements["WhiteBloodCellCount"]
                if isinstance(wbc, dict):
                    value = wbc.get("value", "Unknown")
                    unit = wbc.get("unit", "Unknown")
                    lab_results.append(
                        f"White blood cell count is {value} {unit} (mCODE: WhiteBloodCellCount)"
                    )

            if "PlateletCount" in mcode_elements:
                plt = mcode_elements["PlateletCount"]
                if isinstance(plt, dict):
                    value = plt.get("value", "Unknown")
                    unit = plt.get("unit", "Unknown")
                    lab_results.append(
                        f"Platelet count is {value} {unit} (mCODE: PlateletCount)"
                    )

            if "Creatinine" in mcode_elements:
                creat = mcode_elements["Creatinine"]
                if isinstance(creat, dict):
                    value = creat.get("value", "Unknown")
                    unit = creat.get("unit", "Unknown")
                    lab_results.append(
                        f"Creatinine is {value} {unit} (mCODE: Creatinine)"
                    )

            if "TotalBilirubin" in mcode_elements:
                bili = mcode_elements["TotalBilirubin"]
                if isinstance(bili, dict):
                    value = bili.get("value", "Unknown")
                    unit = bili.get("unit", "Unknown")
                    lab_results.append(
                        f"Total bilirubin is {value} {unit} (mCODE: TotalBilirubin)"
                    )

            if "AlanineAminotransferase" in mcode_elements:
                alt = mcode_elements["AlanineAminotransferase"]
                if isinstance(alt, dict):
                    value = alt.get("value", "Unknown")
                    unit = alt.get("unit", "Unknown")
                    lab_results.append(
                        f"ALT is {value} {unit} (mCODE: AlanineAminotransferase)"
                    )

            if lab_results:
                if len(lab_results) == 1:
                    clinical_note.append(
                        f"Patient laboratory results: {lab_results[0]}."
                    )
                else:
                    clinical_note.append(
                        f"Patient laboratory results: {'; '.join(lab_results[:-1])} and {lab_results[-1]}."
                    )

            # Comorbidities and Other Conditions
            comorbidities = []
            if "ComorbidCondition" in mcode_elements:
                comorbids = mcode_elements["ComorbidCondition"]
                if isinstance(comorbids, list):
                    for comorbid in comorbids:
                        if isinstance(comorbid, dict):
                            display = comorbid.get("display", "Unknown")
                            code = comorbid.get("code", "Unknown")
                            system = comorbid.get("system", "").replace(
                                "http://snomed.info/sct", "SNOMED"
                            )
                            clean_display = (
                                display.split(" (")[0] if " (" in display else display
                            )
                            mcode_format = self._format_mcode_element(
                                "ComorbidCondition", system, code
                            )
                            comorbidities.append(f"{clean_display} {mcode_format}")

            if comorbidities:
                if len(comorbidities) == 1:
                    clinical_note.append(f"Patient comorbidities: {comorbidities[0]}.")
                else:
                    clinical_note.append(
                        f"Patient comorbidities: {'; '.join(comorbidities[:-1])} and {comorbidities[-1]}."
                    )

            # Allergies and Intolerances
            allergies = []
            if "AllergyIntolerance" in mcode_elements:
                allergy_list = mcode_elements["AllergyIntolerance"]
                if isinstance(allergy_list, list):
                    for allergy in allergy_list:
                        if isinstance(allergy, dict):
                            display = allergy.get("display", "Unknown")
                            code = allergy.get("code", "Unknown")
                            system = allergy.get("system", "").replace(
                                "http://snomed.info/sct", "SNOMED"
                            )
                            criticality = allergy.get("criticality", "Unknown")
                            recorded_date = allergy.get("recordedDate", "Unknown")

                            if recorded_date and recorded_date != "Unknown":
                                mcode_format = self._format_mcode_element(
                                    "AllergyIntolerance", system, code
                                )
                                allergies.append(
                                    f"{display} recorded on {recorded_date} (criticality: {criticality}; {mcode_format})"
                                )
                            else:
                                mcode_format = self._format_mcode_element(
                                    "AllergyIntolerance", system, code
                                )
                                allergies.append(
                                    f"{display} (criticality: {criticality}; {mcode_format})"
                                )

            if allergies:
                if len(allergies) == 1:
                    clinical_note.append(f"Patient allergies: {allergies[0]}.")
                else:
                    clinical_note.append(
                        f"Patient allergies: {'; '.join(allergies[:-1])} and {allergies[-1]}."
                    )

            # Immunization History
            immunizations = []
            if "Immunization" in mcode_elements:
                immunization_list = mcode_elements["Immunization"]
                if isinstance(immunization_list, list):
                    for immunization in immunization_list:
                        if isinstance(immunization, dict):
                            display = immunization.get("display", "Unknown")
                            code = immunization.get("code", "Unknown")
                            system = immunization.get("system", "").replace(
                                "http://snomed.info/sct", "SNOMED"
                            )
                            occurrence_date = immunization.get(
                                "occurrenceDateTime", "Unknown"
                            )
                            status = immunization.get("status", "Unknown")

                            if occurrence_date and occurrence_date != "Unknown":
                                mcode_format = self._format_mcode_element(
                                    "Immunization", system, code
                                )
                                immunizations.append(
                                    f"{display} administered on {occurrence_date} (status: {status}; {mcode_format})"
                                )
                            else:
                                mcode_format = self._format_mcode_element(
                                    "Immunization", system, code
                                )
                                immunizations.append(
                                    f"{display} (status: {status}; {mcode_format})"
                                )

            if immunizations:
                if len(immunizations) == 1:
                    clinical_note.append(
                        f"Patient immunization history: {immunizations[0]}."
                    )
                else:
                    clinical_note.append(
                        f"Patient immunization history: {'; '.join(immunizations[:-1])} and {immunizations[-1]}."
                    )

            # Family History
            family_history = []
            if "FamilyMemberHistory" in mcode_elements:
                family_list = mcode_elements["FamilyMemberHistory"]
                if isinstance(family_list, list):
                    for family in family_list:
                        if isinstance(family, dict):
                            relationship = family.get("relationship", {}).get(
                                "display", "Unknown"
                            )
                            conditions = family.get("conditions", [])
                            born = family.get("born", "Unknown")

                            condition_summaries = []
                            for condition in conditions:
                                if isinstance(condition, dict):
                                    cond_display = condition.get("display", "Unknown")
                                    cond_code = condition.get("code", "Unknown")
                                    cond_system = condition.get("system", "").replace(
                                        "http://snomed.info/sct", "SNOMED"
                                    )
                                    condition_summaries.append(
                                        f"{cond_display} ({cond_system}:{cond_code})"
                                    )

                            if condition_summaries:
                                if born and born != "Unknown":
                                    family_history.append(
                                        f"{relationship} born {born} with {' and '.join(condition_summaries)} (mCODE: FamilyMemberHistory)"
                                    )
                                else:
                                    family_history.append(
                                        f"{relationship} with {' and '.join(condition_summaries)} (mCODE: FamilyMemberHistory)"
                                    )

            if family_history:
                if len(family_history) == 1:
                    clinical_note.append(
                        f"Patient family history: {family_history[0]}."
                    )
                else:
                    clinical_note.append(
                        f"Patient family history: {'; '.join(family_history[:-1])} and {family_history[-1]}."
                    )

            summary = " ".join(clinical_note)
            self.logger.info(
                f"Generated clinical note summary for patient {patient_id}: {summary}"
            )
            return summary

        except Exception as e:
            self.logger.error(f"Error generating clinical note summary: {e}")
            return f"Patient {patient_id}: Error generating clinical note - {str(e)}"

    def _decode_birth_sex(self, code: str) -> str:
        """Decode birth sex code to plain English."""
        birth_sex_map = {"F": "Female", "M": "Male", "UNK": "Unknown", "OTH": "Other"}
        return birth_sex_map.get(code.upper(), code)

    def _decode_marital_status(self, code: str) -> str:
        """Decode marital status code to plain English."""
        marital_map = {
            "A": "Annulled",
            "D": "Divorced",
            "I": "Interlocutory",
            "L": "Legally Separated",
            "M": "Married",
            "P": "Polygamous",
            "S": "Single",
            "T": "Domestic Partner",
            "U": "Unmarried",
            "W": "Widowed",
            "UNK": "Unknown",
        }
        return marital_map.get(code.upper(), code)

    def _format_mcode_element(self, element_name: str, system: str, code: str) -> str:
        """Centralized function to format mCODE elements consistently."""
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
        else:
            # Remove URLs and keep only the system identifier
            clean_system = system.split("/")[-1].split(":")[-1].upper()

        return f"(mCODE: {element_name}; {clean_system}:{code})"
