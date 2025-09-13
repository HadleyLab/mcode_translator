#!/usr/bin/env python3
"""
mCODE Patients Preprocessor - Filter patient mCODE data based on clinical trial elements

Author: mCODE Translation Team
Version: 2.0.0
License: MIT
"""
import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv

from src.utils.config import Config
from src.utils.core_memory_client import CoreMemoryClient, CoreMemoryError
from src.utils.logging_config import get_logger, setup_logging
from src.utils.patient_generator import (PatientGenerator,
                                         create_patient_generator)

load_dotenv()


def extract_clinical_trial_mcode_elements(trial_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts relevant mCODE elements from clinical trial data.

    Args:
        trial_data (Dict[str, Any]): A dictionary containing the clinical trial data.

    Returns:
        Dict[str, Any]: A dictionary containing the extracted mCODE elements.
    """
    logger = get_logger(__name__)
    mcode_elements = {}

    # Extract from successful trials
    for trial in trial_data.get("successful_trials", []):
        if "McodeResults" in trial and trial["McodeResults"]:
            mcode_mappings = trial["McodeResults"].get("mcode_mappings", [])
            for mapping in mcode_mappings:
                mcode_element = mapping.get("mcode_element")
                value = mapping.get("value")
                if mcode_element and value:
                    mcode_elements[mcode_element] = value
        else:
            logger.warning(f"No McodeResults found in trial: {trial.get('nctId')}")

    # Extract from failed trials if they have McodeResults
    for trial in trial_data.get("failed_trials", []):
        if "McodeResults" in trial and trial["McodeResults"]:
            mcode_mappings = trial["McodeResults"].get("mcode_mappings", [])
            for mapping in mcode_mappings:
                mcode_element = mapping.get("mcode_element")
                value = mapping.get("value")
                if mcode_element and value:
                    mcode_elements[mcode_element] = value

    return mcode_elements


def extract_patient_mcode_elements(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts comprehensive mCODE elements from patient FHIR Bundle using standard mappings.

    Returns Dict with mCODE keys containing structured data (value, interpretation, date, etc.).
    """
    logger = get_logger(__name__)
    mcode_profile = {}

    # mCODE-to-FHIR mappings (comprehensive clinical trials matching)
    mcode_mappings = {
        # Core Cancer Elements
        "CancerCondition": {
            "resource_types": ["Condition"],
            "snomed_codes": [
                "254837009",
                "108294007",
                "254838001",
            ],  # Breast cancer, malignant neoplasm
            "loinc_codes": ["72113-7"],  # Cancer diagnosis
            "text_patterns": [
                "breast cancer",
                "malignant neoplasm",
                "carcinoma",
                "neoplasm",
            ],
            "priority_codes": ["C50"],  # ICD-10 breast cancer
            "require_histology": True,
        },
        "CancerStage": {
            "resource_types": ["Observation"],
            "loinc_codes": ["21908-9", "59453-7"],  # Stage group, TNM
            "snomed_codes": ["258232001", "385355005"],  # TNM categories
            "text_patterns": ["cancer stage", "tnm stage", "ajcc stage"],
            "related": ["TNMStage"],
        },
        "TNMStage": {
            "resource_types": ["Observation"],
            "loinc_codes": ["59453-7", "21908-9"],  # TNM, Stage group
            "snomed_codes": [
                "258232001",
                "261615002",
                "261634002",
                "258228008",
            ],  # Stage 1A-4
            "text_patterns": [
                "tnm",
                "stage group",
                "t category",
                "n category",
                "m category",
            ],
            "components": ["TumorSize", "LymphNodeInvolvement"],
        },
        # Biomarkers & Tumor Markers
        "HER2ReceptorStatus": {
            "resource_types": ["Observation"],
            "loinc_codes": ["48676-1", "85319-2", "85318-4"],  # IHC, FISH
            "snomed_codes": ["108283007"],  # HER2 receptor status
            "text_patterns": ["her2", "HER2", "neu"],
            "value_mappings": {
                "260385009": "Negative",
                "10828001": "Positive",
                "413444009": "Equivocal",
            },
            "is_biomarker": True,
        },
        "ERReceptorStatus": {
            "resource_types": ["Observation"],
            "loinc_codes": ["85313-5", "48607-9", "92136-1"],  # ER IHC, ER status
            "snomed_codes": ["445281001"],  # ER status
            "text_patterns": ["estrogen receptor", "er receptor", "er status"],
            "is_biomarker": True,
            "exclude_patterns": ["m0", "n0", "t0"],  # Avoid staging observations
        },
        "PRReceptorStatus": {
            "resource_types": ["Observation"],
            "loinc_codes": ["85314-3", "92142-9"],  # PR IHC, PR status
            "snomed_codes": ["445282008"],  # PR status
            "text_patterns": ["progesterone receptor", "pr receptor", "pr status"],
            "is_biomarker": True,
            "exclude_patterns": ["m0", "n0", "t0"],  # Avoid staging observations
        },
        "TumorMarkerTest": {
            "resource_types": ["Observation"],
            "mcode_profile": "mcode-tumor-marker-test",
            "text_patterns": ["tumor marker", "biomarker", "molecular test"],
            "exclude": ["HER2", "ER", "PR"],  # Handle specifically above
            "is_biomarker": True,
        },
        "GenomicVariant": {
            "resource_types": ["Observation"],
            "loinc_codes": ["53037-8", "81347-8"],  # Variant, gene study
            "snomed_codes": ["363344003", "363346002"],  # Genetic analysis
            "text_patterns": ["genomic variant", "mutation", "genetic test"],
            "mcode_profile": "mcode-genomic-variant",
        },
        "HistologyMorphologyBehavior": {
            "resource_types": ["Observation", "Condition"],
            "loinc_codes": ["43986-5"],  # Histology
            "snomed_codes": ["419377000"],  # Histological type
            "text_patterns": ["histology", "morphology", "grade", "differentiation"],
            "mcode_profile": "mcode-histology-morphology",
        },
        # Treatments & Procedures
        "CancerTreatment": {
            "resource_types": ["MedicationStatement", "Procedure"],
            "snomed_codes": [
                "386261006",
                "367336001",
                "278850018",
            ],  # Chemo, radiation, immunotherapy
            "text_patterns": [
                "chemotherapy",
                "radiation",
                "immunotherapy",
                "targeted therapy",
            ],
            "multiple": True,  # Array of treatments
        },
        "CancerRelatedMedication": {
            "resource_types": ["MedicationStatement"],
            "snomed_codes": ["763158003"],  # Cancer chemotherapy regimen
            "text_patterns": [
                "cancer medication",
                "antineoplastic",
                "antineoplastic agent",
            ],
            "multiple": True,
        },
        "CancerRelatedSurgicalProcedure": {
            "resource_types": ["Procedure"],
            "snomed_codes": [
                "387713003",
                "232347006",
                "712946003",
                "64368001",
                "449056005",
                "232153009",
                "232208006",
                "122464006",
                "387713003",
            ],  # Surgical procedures: cancer surgery, mastectomy, lumpectomy, biopsy, lumpectomy, sentinel node, etc.
            "text_patterns": [
                "biopsy",
                "mastectomy",
                "lumpectomy",
                "resection",
                "excision",
                "sentinel",
                "lymph node",
                "brachytherapy",
                "radiation therapy",
                "breast surgery",
            ],
            "exclude_patterns": [
                "medication",
                "reconciliation",
                "transfer",
                "administration",
                "consultation",
                "assessment",
                "monitoring",
            ],
            "multiple": True,
            "require_cancer_context": True,
        },
        "RadiationDose": {
            "resource_types": ["Observation"],
            "loinc_codes": ["55335-6"],  # Radiation dose
            "text_patterns": ["radiation dose", "radiation therapy", "rt dose"],
            "expect_quantity": True,
            "multiple": True,
        },
        # Adverse Events & Safety
        "AdverseEvent": {
            "resource_types": ["Observation"],
            "loinc_codes": ["75276-9"],  # Adverse event
            "snomed_codes": ["62315001"],  # Adverse reaction
            "text_patterns": ["adverse event", "toxicity", "side effect", "ae"],
            "mcode_profile": "mcode-adverse-event",
            "multiple": True,
        },
        # Comorbidities & Performance
        "CancerComorbidity": {
            "resource_types": ["Condition"],
            "snomed_codes": ["64572001"],  # Comorbidity
            "text_patterns": ["comorbidity", "concurrent condition"],
            "related_to": "CancerCondition",
        },
        "ECOGPerformanceStatus": {
            "resource_types": ["Observation"],
            "loinc_codes": ["89250-6"],  # ECOG score
            "text_patterns": ["ecog", "performance status"],
            "snomed_codes": ["288528006"],
        },
        # Patient Demographics
        "Patient": {
            "resource_types": ["Patient"],
            "fields": [
                "gender",
                "birthDate",
                "deceasedBoolean",
                "multipleBirthBoolean",
            ],
        },
        "PatientAgeGroup": {
            "resource_types": ["Patient"],
            "calculated": True,
            "age_ranges": {
                "<18": "Pediatric",
                "18-39": "Young Adult",
                "40-64": "Adult",
                "65+": "Senior",
            },
        },
        "PatientSex": {"resource_types": ["Patient"], "path": "gender"},
        "Race": {
            "resource_types": ["Patient"],
            "extension_url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
        },
        "Ethnicity": {
            "resource_types": ["Patient"],
            "extension_url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity",
        },
    }

    # Process each entry in the Bundle
    procedures_data = []  # Collect all procedures separately
    for entry in patient_data.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")

        # Special handling for Patient demographics (always first)
        if resource_type == "Patient" and not any(
            k in mcode_profile for k in ["PatientSex", "Race", "Ethnicity"]
        ):
            demographics = extract_demographics(resource)
            mcode_profile.update(demographics)
            continue

        # Collect all Procedure resources for later processing - use improved matching
        if resource_type == "Procedure":
            # Check if this procedure matches cancer-related criteria
            proc_mapping = mcode_mappings.get("CancerRelatedSurgicalProcedure", {})
            if match_mcode_resource(resource, proc_mapping):
                proc_value = extract_mcode_value(
                    resource, "CancerRelatedSurgicalProcedure", proc_mapping
                )
                if proc_value:
                    procedures_data.append(proc_value)
                    logger.debug(
                        f"Added procedure: {proc_value.get('display', 'Unknown')}"
                    )
            continue

        # Check each mCODE mapping for other resource types (skip procedures here)
        for mcode_key, mapping in mcode_mappings.items():
            if (
                resource_type in mapping.get("resource_types", [])
                and mcode_key not in mcode_profile
                and mcode_key != "CancerRelatedSurgicalProcedure"
            ):
                matched = match_mcode_resource(resource, mapping)
                if matched:
                    value = extract_mcode_value(resource, mcode_key, mapping)
                    if value:
                        mcode_profile[mcode_key] = value
                        logger.debug(f"Extracted {mcode_key}: {value}")
                        break  # One match per element

    # Add procedures as an array if we have multiple
    if len(procedures_data) > 1:
        mcode_profile["CancerRelatedSurgicalProcedure"] = procedures_data
        logger.info(
            f"Extracted {len(procedures_data)} CancerRelatedSurgicalProcedure elements"
        )
    elif len(procedures_data) == 1:
        mcode_profile["CancerRelatedSurgicalProcedure"] = procedures_data[0]
        logger.info(
            f"Extracted single CancerRelatedSurgicalProcedure: {procedures_data[0].get('display', 'Unknown')}"
        )
    elif procedures_data:
        logger.debug("No cancer-related procedures found in patient data")

    logger.info(
        f"Extracted {len(mcode_profile)} comprehensive mCODE elements from patient data: {list(mcode_profile.keys())}"
    )
    return mcode_profile


def match_mcode_resource(resource: Dict, mapping: Dict) -> bool:
    """Determine if resource matches mCODE mapping criteria."""
    code = resource.get("code", {})
    coding = code.get("coding", [])
    display = code.get("text", "").lower() if code.get("text") else ""

    # For procedures, be more specific about cancer-related context
    if mapping.get("key") == "CancerRelatedSurgicalProcedure":
        # Check if it's breast/cancer related
        cancer_context = any(
            term in display
            for term in ["breast", "cancer", "tumor", "neoplasm", "malignant"]
        )
        if not cancer_context:
            return False

        # Exclude non-procedural activities
        exclude_patterns = mapping.get("exclude_patterns", [])
        if any(pattern in display for pattern in exclude_patterns):
            return False

    # Highest priority: mCODE profile match (most specific)
    if mapping.get("mcode_profile"):
        profiles = resource.get("meta", {}).get("profile", [])
        if mapping["mcode_profile"] in profiles:
            return True

    # Highest priority: mCODE profile match (most specific)
    if mapping.get("mcode_profile"):
        profiles = resource.get("meta", {}).get("profile", [])
        if mapping["mcode_profile"] in profiles:
            return True

    # Next: Exact LOINC code match (biomarkers, stages, etc.)
    loinc_codes = mapping.get("loinc_codes", [])
    if loinc_codes:
        for c in coding:
            if c.get("system") == "http://loinc.org" and c.get("code") in loinc_codes:
                return True

    # Next: Exact SNOMED code match
    snomed_codes = mapping.get("snomed_codes", [])
    if snomed_codes:
        for c in coding:
            if (
                c.get("system") == "http://snomed.info/sct"
                and c.get("code") in snomed_codes
            ):
                return True

    # For biomarkers, check if it's a tumor marker test with valueCodeableConcept (Positive/Negative)
    if mapping.get("is_biomarker") and resource.get("valueCodeableConcept"):
        vc = resource.get("valueCodeableConcept")
        vc_coding = vc.get("coding", [])
        # Look for SNOMED interpretation codes (260385009=Negative, 10828001=Positive)
        for vc_c in vc_coding:
            if vc_c.get("system") == "http://snomed.info/sct" and vc_c.get("code") in [
                "260385009",
                "10828001",
                "413444009",
            ]:  # Neg, Pos, Equivocal
                # Also check if text patterns match (HER2, ER, PR) - be more specific
                patterns = mapping.get("text_patterns", [])
                display_lower = display.lower()
                if any(pattern in display_lower for pattern in patterns):
                    # Additional check for ER/PR to avoid M0/N0 staging confusion
                    if "er" in mapping.get("key", "").lower() and any(
                        term in display_lower for term in ["estrogen", "er receptor"]
                    ):
                        return True
                    elif "pr" in mapping.get("key", "").lower() and any(
                        term in display_lower
                        for term in ["progesterone", "pr receptor"]
                    ):
                        return True
                    elif any(pattern in display_lower for pattern in patterns):
                        return True

    # For CancerCondition, specific cancer-related matches
    if mapping.get("key") == "CancerCondition":
        for c in coding:
            if c.get("system") == "http://snomed.info/sct":
                if c.get("code") in [
                    "254837009",
                    "108294007",
                    "254838001",
                ]:  # Breast cancer codes
                    return True
            if c.get(
                "system"
            ) == "http://hl7.org/fhir/sid/icd-10-cm" and "C50" in c.get("code", ""):
                return True
            if any(
                term in (c.get("display", "") or "").lower()
                for term in ["breast cancer", "malignant neoplasm of breast"]
            ):
                return True

    # Last resort: text pattern matching (only if no exact matches above)
    patterns = mapping.get("text_patterns", [])
    if patterns and any(pattern in display for pattern in patterns):
        # For procedures, ensure cancer context if required
        if mapping.get("require_cancer_context"):
            cancer_context = any(
                term in display
                for term in ["breast", "cancer", "tumor", "neoplasm", "malignant"]
            )
            if not cancer_context:
                return False

        # For text matches, also check if it's not a generic measurement
        if not (resource.get("valueQuantity") and mapping.get("is_biomarker")):
            return True

    return False


def extract_mcode_value(
    resource: Dict, mcode_key: str, mapping: Dict
) -> Optional[Dict]:
    """Extract structured value from resource for specific mCODE element."""
    value_data = {
        "system": None,
        "code": None,
        "display": None,
        "interpretation": None,
        "interpretation_system": None,
        "interpretation_code": None,
        "quantity": None,
        "date": resource.get("effectiveDateTime") or resource.get("issued"),
        "reference": resource.get("id"),
    }

    code = resource.get("code", {})
    coding = code.get("coding", [])

    # For biomarkers (HER2, ER, PR), prioritize valueCodeableConcept interpretation
    if mapping.get("is_biomarker"):
        if resource.get("valueCodeableConcept"):
            vc = resource.get("valueCodeableConcept")
            vc_coding = vc.get("coding", [])

            # Find the best interpretation coding (SNOMED preferred)
            best_vc = None
            snomed_map = mapping.get("value_mappings", {})
            for vc_c in vc_coding:
                if vc_c.get("system") == "http://snomed.info/sct":
                    # Map SNOMED codes to readable values
                    mapped_display = snomed_map.get(vc_c.get("code"))
                    if mapped_display:
                        value_data["interpretation"] = mapped_display
                    else:
                        value_data["interpretation"] = vc_c.get("display") or vc.get(
                            "text", "Unknown"
                        )
                    value_data["interpretation_system"] = vc_c.get("system")
                    value_data["interpretation_code"] = vc_c.get("code")
                    best_vc = vc_c
                    break  # Use first SNOMED match

            # If no SNOMED, use first coding
            if not best_vc and vc_coding:
                vc_c = vc_coding[0]
                value_data["interpretation"] = vc_c.get("display") or vc.get(
                    "text", "Unknown"
                )
                value_data["interpretation_system"] = vc_c.get("system")
                value_data["interpretation_code"] = vc_c.get("code")
                best_vc = vc_c

            # Set code from resource code (the test type, e.g., HER2 [Interpretation])
            if coding:
                # Prefer LOINC code for the test
                loinc_coding = next(
                    (c for c in coding if c.get("system") == "http://loinc.org"),
                    coding[0],
                )
                value_data["system"] = loinc_coding.get("system")
                value_data["code"] = loinc_coding.get("code")
                raw_display = loinc_coding.get("display") or code.get("text")
                # Clean display text of common qualifiers
                cleaned_display = (
                    raw_display.replace(" (disorder)", "")
                    .replace(" (procedure)", "")
                    .replace(" (finding)", "")
                    .replace(" (qualifier value)", "")
                    .replace(" (regime/therapy)", "")
                    .replace(" (morphologic abnormality)", "")
                    .strip()
                )
                value_data["display"] = cleaned_display

            # If we have an interpretation, that's our main value
            if (
                value_data["interpretation"]
                and value_data["interpretation"] != "Unknown"
            ):
                value_data["display"] = f"{value_data['interpretation']} ({mcode_key})"
                return value_data

        # If no valueCodeableConcept for biomarker, it might be a quantity (older format)
        elif resource.get("valueQuantity"):
            vq = resource.get("valueQuantity")
            value_data["quantity"] = {
                "value": vq.get("value"),
                "unit": vq.get("unit"),
                "code": vq.get("code"),
            }
            # Set code from LOINC if available
            if coding and any(c.get("system") == "http://loinc.org" for c in coding):
                loinc_c = next(
                    c for c in coding if c.get("system") == "http://loinc.org"
                )
                value_data["system"] = loinc_c.get("system")
                value_data["code"] = loinc_c.get("code")
                value_data["display"] = loinc_c.get("display")
            else:
                value_data["display"] = code.get(
                    "text", f"{vq.get('value')} {vq.get('unit')}"
                )
            return value_data

    # For non-biomarkers (procedures, conditions, etc.)
    else:
        # Find best coding match
        best_coding = None
        for c in coding:
            if (
                c.get("system") == "http://snomed.info/sct"
            ):  # Prefer SNOMED for procedures/conditions
                best_coding = c
                break
        if not best_coding:
            best_coding = coding[0] if coding else None

        if best_coding:
            value_data["system"] = best_coding.get("system")
            value_data["code"] = best_coding.get("code")
            raw_display = best_coding.get("display") or code.get("text")
            # Clean display text of common qualifiers
            cleaned_display = (
                raw_display.replace(" (disorder)", "")
                .replace(" (procedure)", "")
                .replace(" (finding)", "")
                .replace(" (qualifier value)", "")
                .replace(" (regime/therapy)", "")
                .replace(" (morphologic abnormality)", "")
                .strip()
            )
            value_data["display"] = cleaned_display

        # For quantities (tumor size, etc.)
        if resource.get("valueQuantity") and mapping.get("expect_quantity", False):
            vq = resource.get("valueQuantity")
            value_data["quantity"] = {
                "value": vq.get("value"),
                "unit": vq.get("unit"),
                "code": vq.get("code"),
            }
            value_data["display"] = f"{vq.get('value')} {vq.get('unit')}"
        # For staging and other observations with valueCodeableConcept
        elif resource.get("valueCodeableConcept"):
            vc = resource.get("valueCodeableConcept")
            vc_coding = vc.get("coding", [{}])[0]
            raw_interpretation = vc_coding.get("display") or vc.get("text")
            # Clean interpretation text of common qualifiers
            cleaned_interpretation = (
                raw_interpretation.replace(" (disorder)", "")
                .replace(" (procedure)", "")
                .replace(" (finding)", "")
                .replace(" (qualifier value)", "")
                .replace(" (regime/therapy)", "")
                .replace(" (morphologic abnormality)", "")
                .strip()
            )
            value_data["interpretation"] = cleaned_interpretation
            value_data["display"] = value_data["interpretation"]

        # Return if we have meaningful data
        if value_data["code"] or value_data["display"] or value_data["quantity"]:
            return value_data

    return None


def extract_demographics(patient_resource: Dict) -> Dict:
    """Extract basic demographics as mCODE elements."""
    demographics = {}

    # PatientSex
    gender = patient_resource.get("gender")
    if gender:
        demographics["PatientSex"] = {
            "value": gender,
            "system": "http://hl7.org/fhir/administrative-gender",
            "display": {
                "male": "Male",
                "female": "Female",
                "other": "Other",
                "unknown": "Unknown",
            }.get(gender, gender),
        }

    # Race/Ethnicity from US Core extensions
    extensions = patient_resource.get("extension", [])
    for ext in extensions:
        url = ext.get("url")
        if url == "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race":
            coding = ext.get("valueCoding", {})
            demographics["Race"] = {
                "system": coding.get("system"),
                "code": coding.get("code"),
                "display": coding.get("display", "Unknown"),
            }
        elif url == "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity":
            coding = ext.get("valueCoding", {})
            demographics["Ethnicity"] = {
                "system": coding.get("system"),
                "code": coding.get("code"),
                "display": coding.get("display", "Unknown"),
            }

    return demographics


def filter_patient_mcode_elements(
    patient_mcode_elements: Dict[str, Any],
    clinical_trial_mcode_elements: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Filters patient mCODE elements to keep only those present in clinical trial data.

    Args:
        patient_mcode_elements (Dict[str, Any]): mCODE elements extracted from patient data
        clinical_trial_mcode_elements (Dict[str, Any]): mCODE elements extracted from clinical trial data

    Returns:
        Dict[str, Any]: Filtered patient mCODE elements
    """
    logger = get_logger(__name__)
    filtered_elements = {}

    # Get the set of mCODE element types from clinical trials
    clinical_trial_element_types = set(clinical_trial_mcode_elements.keys())

    # Filter patient elements to keep only those types present in clinical trials
    for element_type, element_data in patient_mcode_elements.items():
        if element_type in clinical_trial_element_types:
            filtered_elements[element_type] = element_data
            logger.debug(f"Keeping patient mCODE element: {element_type}")
        else:
            logger.debug(f"Filtering out patient mCODE element: {element_type}")

    logger.info(
        f"Filtered {len(filtered_elements)}/{len(patient_mcode_elements)} patient mCODE elements"
    )
    return filtered_elements


# PatientGenerator handles all ZIP archive loading - no direct ZIP handling needed here


def extract_demographics(patient_resource: Dict) -> Dict:
    """Extract basic demographics as mCODE elements."""
    demographics = {}

    # PatientSex
    gender = patient_resource.get("gender")
    if gender:
        demographics["PatientSex"] = {
            "value": gender,
            "system": "http://hl7.org/fhir/administrative-gender",
            "display": {
                "male": "Male",
                "female": "Female",
                "other": "Other",
                "unknown": "Unknown",
            }.get(gender, gender),
        }

    # Race/Ethnicity from US Core extensions
    extensions = patient_resource.get("extension", [])
    for ext in extensions:
        url = ext.get("url")
        if url == "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race":
            coding = ext.get("valueCoding", {})
            demographics["Race"] = {
                "system": coding.get("system"),
                "code": coding.get("code"),
                "display": coding.get("display", "Unknown"),
            }
        elif url == "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity":
            coding = ext.get("valueCoding", {})
            demographics["Ethnicity"] = {
                "system": coding.get("system"),
                "code": coding.get("code"),
                "display": coding.get("display", "Unknown"),
            }

    return demographics


def store_filtered_patients_to_memory(
    filtered_data: Dict[str, Any], api_key: str
) -> None:
    """
    Store filtered patient summaries in CORE Memory, similar to mcode_fetcher.py.

    Args:
        filtered_data (Dict[str, Any]): The filtered patient Bundle data.
        api_key (str): CORE Memory API key.
    """
    logger = get_logger(__name__)
    try:
        client = CoreMemoryClient(api_key=api_key)

        # First, find Patient resource and create mapping
        patient_map = {}
        for entry in filtered_data.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Patient":
                patient_id = resource.get("identifier", [{}])[0].get("value", "unknown")
                name = (
                    " ".join(resource.get("name", [{}])[0].get("given", []))
                    + " "
                    + resource.get("name", [{}])[0].get("family", "")
                )
                dob = resource.get("birthDate")
                gender = resource.get("gender")
                race = next(
                    (
                        ext.get("extension", [{}])[0]
                        .get("valueCoding", {})
                        .get("display")
                        for ext in resource.get("extension", [])
                        if ext.get("url")
                        == "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race"
                    ),
                    "Unknown",
                )
                ethnicity = next(
                    (
                        ext.get("extension", [{}])[0]
                        .get("valueCoding", {})
                        .get("display")
                        for ext in resource.get("extension", [])
                        if ext.get("url")
                        == "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity"
                    ),
                    "Unknown",
                )
                patient_map[patient_id] = {
                    "name": name,
                    "dob": dob,
                    "gender": gender,
                    "race": race,
                    "ethnicity": ethnicity,
                    "mcode": filtered_data.get("filtered_mcode_elements", {}),
                    "procedures": [],
                    "vitals": {},
                    "conditions": [],
                    "id": patient_id,  # Store ID for reference matching
                }
                break  # Assume single patient for now

        patients = patient_map

        if not patients:
            logger.warning("No Patient resource found in filtered data")
            return

        patient_id = next(iter(patients))  # Get the single patient ID

        # Collect procedures, vitals, conditions for the patient
        for entry in filtered_data.get("entry", []):
            resource = entry.get("resource", {})
            rtype = resource.get("resourceType")
            subject_ref = resource.get("subject", {}).get("reference", "")
            if (
                f"Patient/{patient_id}" in subject_ref or patient_id in subject_ref
            ):  # Match by subject reference
                if rtype == "Procedure":
                    proc_code = resource.get("code", {}).get("text", "")
                    patients[patient_id]["procedures"].append(proc_code)
                elif rtype == "Observation" and "vital-signs" in resource.get(
                    "category", [{}]
                )[0].get("coding", [{}])[0].get("code", ""):
                    code_text = resource.get("code", {}).get("text", "")
                    value = resource.get("valueQuantity", {}).get("value")
                    unit = resource.get("valueQuantity", {}).get("unit", "")
                    patients[patient_id]["vitals"][code_text] = f"{value}{unit}"
                elif rtype == "Condition":
                    cond_code = resource.get("code", {}).get("text", "")
                    patients[patient_id]["conditions"].append(cond_code)

        # Deduplicate procedures
        patients[patient_id]["procedures"] = list(
            set(patients[patient_id]["procedures"])
        )

        # Build and ingest per-patient summaries
        for pid, data in patients.items():
            summary_parts = [
                f"Patient {data['name']} (PatientID: {pid}) is a {data['gender']} born on {data['dob']}.",
                f"The patient's race is {data['race']} and ethnicity is {data['ethnicity']}.",
            ]

            mcode = data["mcode"]
            if "CancerCondition" in mcode:
                cc = mcode["CancerCondition"]
                system_url = cc.get("system", "CodeSystem")
                code_system = (
                    "SNOMED"
                    if "snomed" in system_url
                    else "LOINC" if "loinc" in system_url else "CodeSystem"
                )
                display_text = cc.get("display", "").replace(" (disorder)", "")
                summary_parts.append(
                    f"The patient has a diagnosis of {display_text} (mCODE:CancerCondition, {code_system}: {cc.get('code', '')})"
                )
            if "TNMStage" in mcode:
                ts = mcode["TNMStage"]
                system_url = ts.get("system", "CodeSystem")
                code_system = (
                    "SNOMED"
                    if "snomed" in system_url
                    else "LOINC" if "loinc" in system_url else "CodeSystem"
                )
                stage_display = ts.get("display", "").replace(" (qualifier value)", "")
                summary_parts.append(
                    f"The cancer is staged as {stage_display} (mCODE:TNMStage, {code_system}: {ts.get('code', '')})"
                )
            if "HER2ReceptorStatus" in mcode:
                her2 = mcode["HER2ReceptorStatus"]
                # Prioritize interpretation, then display
                her2_status = her2.get("interpretation") or her2.get(
                    "display", "Unknown"
                )
                if her2_status == "Unknown" and her2.get("quantity"):
                    # Fallback to quantity if no interpretation
                    qty = her2["quantity"]
                    her2_status = f"{qty['value']} {qty['unit']}"
                system_url = her2.get("system", "CodeSystem")
                code_system = (
                    "SNOMED"
                    if "snomed" in system_url
                    else "LOINC" if "loinc" in system_url else "CodeSystem"
                )
                summary_parts.append(
                    f"HER2 receptor status is {her2_status} (mCODE:HER2ReceptorStatus, {code_system}: {her2.get('code', '')})"
                )
            if "ERReceptorStatus" in mcode:
                er = mcode["ERReceptorStatus"]
                er_status = er.get("interpretation") or er.get("display", "Unknown")
                if er_status == "Unknown" and er.get("quantity"):
                    qty = er["quantity"]
                    er_status = f"{qty['value']} {qty['unit']}"
                system_url = er.get("system", "CodeSystem")
                code_system = (
                    "SNOMED"
                    if "snomed" in system_url
                    else "LOINC" if "loinc" in system_url else "CodeSystem"
                )
                summary_parts.append(
                    f"Estrogen receptor status is {er_status} (mCODE:ERReceptorStatus, {code_system}: {er.get('code', '')})"
                )
            if "PRReceptorStatus" in mcode:
                pr = mcode["PRReceptorStatus"]
                pr_status = pr.get("interpretation") or pr.get("display", "Unknown")
                if pr_status == "Unknown" and pr.get("quantity"):
                    qty = pr["quantity"]
                    pr_status = f"{qty['value']} {qty['unit']}"
                system_url = pr.get("system", "CodeSystem")
                code_system = (
                    "SNOMED"
                    if "snomed" in system_url
                    else "LOINC" if "loinc" in system_url else "CodeSystem"
                )
                summary_parts.append(
                    f"Progesterone receptor status is {pr_status} (mCODE:PRReceptorStatus, {code_system}: {pr.get('code', '')})"
                )
            if "TumorSize" in mcode:
                size = mcode["TumorSize"]
                if size.get("quantity"):
                    system_url = size.get("system", "CodeSystem")
                    code_system = (
                        "SNOMED"
                        if "snomed" in system_url
                        else "LOINC" if "loinc" in system_url else "CodeSystem"
                    )
                    summary_parts.append(
                        f"The tumor size is {size['quantity']['value']} {size['quantity']['unit']} (mCODE:TumorSize, {code_system}: {size.get('code', '')})"
                    )
                else:
                    summary_parts.append(
                        f"Tumor size is {size.get('display', 'N/A')} (mCODE:TumorSize)."
                    )
            if "LymphNodeInvolvement" in mcode:
                lymph = mcode["LymphNodeInvolvement"]
                if lymph.get("quantity"):
                    system_url = lymph.get("system", "CodeSystem")
                    code_system = (
                        "SNOMED"
                        if "snomed" in system_url
                        else "LOINC" if "loinc" in system_url else "CodeSystem"
                    )
                    summary_parts.append(
                        f"There are {lymph['quantity']['value']} involved lymph nodes (mCODE:LymphNodeInvolvement, {code_system}: {lymph.get('code', '')})"
                    )
                else:
                    summary_parts.append(
                        f"Lymph node involvement is {lymph.get('display', 'N/A')} (mCODE:LymphNodeInvolvement)."
                    )
            # Use extracted procedure mCODE data instead of general procedures array
            procedures_mcode = mcode.get("CancerRelatedSurgicalProcedure", [])
            if isinstance(procedures_mcode, list):
                # Multiple procedures extracted
                proc_details = []
                for proc in procedures_mcode[:5]:  # Limit to 5 procedures
                    cleaned_proc = (
                        proc.get("display", "Unknown procedure")
                        .replace(" (procedure)", "")
                        .replace(" (finding)", "")
                        .strip()
                    )
                    system_url = proc.get("system", "CodeSystem")
                    code_system = (
                        "SNOMED"
                        if "snomed" in system_url.lower()
                        else "LOINC" if "loinc" in system_url.lower() else "CodeSystem"
                    )
                    code = proc.get("code", "Unknown")
                    proc_details.append(
                        f"{cleaned_proc} (mCODE:CancerRelatedSurgicalProcedure, {code_system}: {code})"
                    )

                if proc_details:
                    procs_str = ", ".join(proc_details)
                    summary_parts.append(
                        f"Key cancer-related procedures include: {procs_str}"
                    )
            elif isinstance(procedures_mcode, dict):
                # Single procedure
                cleaned_proc = (
                    procedures_mcode.get("display", "Unknown procedure")
                    .replace(" (procedure)", "")
                    .replace(" (finding)", "")
                    .strip()
                )
                system_url = procedures_mcode.get("system", "CodeSystem")
                code_system = (
                    "SNOMED"
                    if "snomed" in system_url.lower()
                    else "LOINC" if "loinc" in system_url.lower() else "CodeSystem"
                )
                code = procedures_mcode.get("code", "Unknown")
                summary_parts.append(
                    f"Key cancer-related procedure: {cleaned_proc} (mCODE:CancerRelatedSurgicalProcedure, {code_system}: {code})"
                )
            if data["conditions"]:
                cancer_conds = [
                    c
                    for c in data["conditions"]
                    if any(
                        term in c.lower()
                        for term in [
                            "cancer",
                            "breast",
                            "malignant",
                            "neoplasm",
                            "tumor",
                        ]
                    )
                ]
                if cancer_conds:
                    cond_details = []
                    for cond_text in cancer_conds:
                        # Clean condition text of common qualifiers
                        cleaned_cond = (
                            cond_text.replace(" (disorder)", "")
                            .replace(" (finding)", "")
                            .replace(" (morphologic abnormality)", "")
                            .strip()
                        )

                        cond_mcode = mcode.get("CancerCondition")
                        if (
                            cond_mcode
                            and cond_mcode.get("display")
                            and any(
                                cleaned_cond.lower()
                                in cond_mcode.get("display", "").lower()
                                for _ in [0]
                            )
                        ):
                            system_url = cond_mcode.get("system", "CodeSystem")
                            code_system = (
                                "SNOMED"
                                if "snomed" in system_url.lower()
                                else (
                                    "LOINC"
                                    if "loinc" in system_url.lower()
                                    else "CodeSystem"
                                )
                            )
                            cond_details.append(
                                f"{cleaned_cond} (mCODE:CancerCondition, {code_system}: {cond_mcode.get('code')})"
                            )
                        else:
                            cond_details.append(cleaned_cond)
                    conds_str = ", ".join(cond_details)
                    if cond_details:  # Only add if we have meaningful conditions
                        summary_parts.append(
                            f"Other cancer-related conditions include: {conds_str}"
                        )

            summary = " ".join(summary_parts)
            logger.debug(f"CORE Memory summary for patient {pid}: {summary}")
            client.ingest(summary)
            logger.info(
                f"Stored summary for patient {pid} ({data['name']}) in CORE Memory"
            )

        logger.info("Successfully stored patient summaries in CORE Memory")

    except Exception as e:
        if isinstance(e, CoreMemoryError):
            logger.error(f"Failed to store patient summaries to CORE Memory: {e}")
        else:
            logger.error(f"Unexpected error while storing to CORE Memory: {e}")
        # Continue without halting


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for mCODE Patients Preprocessor."""
    parser = argparse.ArgumentParser(
        description="mCODE Patients Preprocessor - Filter patient mCODE data based on clinical trial elements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file processing
  mcode_patients.py --input-file data/fetcher_output/deepseek-chat.results.json \
    --patient-file patient.pruned.for_deepseek.json --output patient.filtered.for_deepseek.json

  # Load from synthetic patient ZIP archive
  mcode_patients.py --archive-path data/synthetic_patients/breast_cancer/10_years/breast_cancer_10_years.zip \
    --output patient_from_archive_filtered.json

  # Batch processing with directory recursion
  mcode_patients.py --input-file data/fetcher_output/deepseek-chat.results.json \
    --input-dir data/mcode_downloads --output-dir data/mcode_filtered --workers 4

  # With CORE Memory storage
  mcode_patients.py --input-file data/fetcher_output/deepseek-chat.results.json \
    --patient-file patient.pruned.for_deepseek.json --store-in-core-memory --verbose
        """,
    )
    parser.add_argument(
        "--input-file",
        "-i",
        type=str,
        default="data/fetcher_output/deepseek-chat.results.json",
        help="Path to clinical trial data file (default: data/fetcher_output/deepseek-chat.results.json)",
    )
    parser.add_argument(
        "--patient-file",
        "-p",
        type=str,
        default=None,
        help="Path to single patient data file (JSON Bundle). Use --input-dir for batch or --archive-path for ZIP archives.",
    )
    parser.add_argument(
        "--archive-path",
        "-a",
        type=str,
        default=None,
        help="Path to synthetic patient ZIP archive. Overrides --patient-file for loading from archives.",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Specific patient ID to load from archive (if not specified, loads all patients).",
    )
    parser.add_argument(
        "--input-dir",
        "-d",
        type=str,
        default=None,
        help="Input directory to recurse for patient .json files. Overrides --patient-file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for single file mode. For batch, use --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        "-O",
        type=str,
        default=None,
        help="Output directory for batch mode. Filtered files saved with '_filtered' suffix.",
    )
    parser.add_argument(
        "--store-in-core-memory",
        action="store_true",
        help="Store filtered patient summaries to CORE Memory",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of parallel workers for batch processing (default: CPU count)",
    )
    # Universal flags
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose (DEBUG) logging"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )
    parser.add_argument(
        "--config", type=str, help="Path to configuration file (overrides default)"
    )
    return parser


def main() -> None:
    """Main function to preprocess patient mCODE data based on clinical trial mCODE elements."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else getattr(logging, args.log_level)
    setup_logging(level=log_level)
    logger = get_logger(__name__)

    logger.info("🚀 mCODE Patients Preprocessor starting")

    try:
        # Initialize configuration
        config = Config()
        if args.config:
            logger.info(f"Using custom config: {args.config}")

        # Handle input selection - allow standalone patient processing
        clinical_trial_file = None
        if args.input_file:
            clinical_trial_file = args.input_file

        # Determine patient input source
        patient_source = None
        if args.archive_path:
            patient_source = "archive"
            archive_path = args.archive_path
            patient_id = args.patient_id
            logger.info(f"Loading patients from synthetic archive: {archive_path}")
        elif args.patient_file:
            patient_source = "file"
            patient_file = args.patient_file
            logger.info(f"Loading patient from file: {patient_file}")
        elif args.input_dir:
            patient_source = "directory"
        else:
            if clinical_trial_file:
                logger.warning(
                    "No patient data source specified (--patient-file, --archive-path, or --input-dir). Running clinical trial extraction only."
                )
            else:
                logger.error(
                    "Must provide either --input-file (with patient source), --patient-file, --archive-path, or --input-dir"
                )
                sys.exit(1)

        # Read clinical trial data (if provided)
        if clinical_trial_file:
            logger.info(f"Reading clinical trial data from: {clinical_trial_file}")
            with open(clinical_trial_file, "r") as f:
                clinical_trial_data = json.load(f)

            # Extract mCODE elements from clinical trial data
            clinical_trial_mcode_elements = extract_clinical_trial_mcode_elements(
                clinical_trial_data
            )
            logger.info(
                f"Extracted {len(clinical_trial_mcode_elements)} mCODE elements from clinical trial data: {list(clinical_trial_mcode_elements.keys())}"
            )
        else:
            clinical_trial_mcode_elements = {}
            logger.info(
                "No clinical trial data provided - running in standalone patient extraction mode"
            )

        if args.input_dir:
            # Batch mode: Recurse through input_dir for .json files (unchanged)
            processed_count = 0
            import os

            for root, dirs, files in os.walk(args.input_dir):
                for file in files:
                    if file.endswith(".json"):
                        file_path = os.path.join(root, file)
                        logger.info(f"Processing batch file: {file_path}")
                        with open(file_path, "r") as f:
                            patient_data = json.load(f)

                        # Extract and filter
                        patient_mcode_elements = extract_patient_mcode_elements(
                            patient_data
                        )
                        logger.info(
                            f"Extracted {len(patient_mcode_elements)} mCODE elements from {file}: {list(patient_mcode_elements.keys())}"
                        )

                        filtered_patient_mcode_elements = filter_patient_mcode_elements(
                            patient_mcode_elements, clinical_trial_mcode_elements
                        )

                        # Create filtered data
                        filtered_patient_data = patient_data.copy()
                        filtered_patient_data["filtered_mcode_elements"] = (
                            filtered_patient_mcode_elements
                        )

                        # Mirror output structure
                        rel_path = os.path.relpath(file_path, args.input_dir)
                        output_path = os.path.join(
                            args.output_dir or args.output,
                            rel_path.replace(".json", "_filtered.json"),
                        )
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        with open(output_path, "w") as f:
                            json.dump(filtered_patient_data, f, indent=2)
                        logger.info(f"Saved filtered data to: {output_path}")

                        # Store to CORE Memory if requested
                        if args.store_in_core_memory:
                            api_key = config.get_core_memory_api_key()
                            store_filtered_patients_to_memory(
                                filtered_patient_data, api_key
                            )

                        processed_count += 1

            logger.info(
                f"Batch processing completed: {processed_count} files processed"
            )
        elif patient_source == "generator":
            # PatientGenerator mode - using new PatientGenerator class
            try:
                # Create generator with appropriate parameters
                generator = create_patient_generator(
                    archive_identifier=archive_identifier,
                    config=config,
                    shuffle=shuffle,
                    seed=seed,
                )

                if random_selection:
                    # Get single random patient
                    patient_data = generator.get_random_patient()
                    patient_bundles = [patient_data]
                    logger.info(
                        "Selected random patient from archive using PatientGenerator"
                    )
                elif patient_id:
                    # Get specific patient by ID
                    patient_data = generator.get_patient_by_id(patient_id)
                    patient_bundles = [patient_data]
                    logger.info(
                        f"Selected specific patient {patient_id} from archive using PatientGenerator"
                    )
                else:
                    # Get all patients or limited slice
                    if limit:
                        patient_bundles = generator.get_patients(limit=limit)
                        logger.info(
                            f"Loaded {len(patient_bundles)} patients from archive (limited to {limit})"
                        )
                    else:
                        patient_bundles = list(generator)  # Convert iterator to list
                        logger.info(
                            f"Loaded all {len(patient_bundles)} patients from archive"
                        )

                processed_count = 0
                for i, patient_data in enumerate(patient_bundles):
                    logger.info(
                        f"Processing patient bundle {i+1}/{len(patient_bundles)} from PatientGenerator"
                    )

                    # Extract mCODE elements from patient data
                    patient_mcode_elements = extract_patient_mcode_elements(
                        patient_data
                    )
                    logger.info(
                        f"Extracted {len(patient_mcode_elements)} mCODE elements from generator patient {i+1}: {list(patient_mcode_elements.keys())}"
                    )

                    # Use all elements if no clinical trial filtering, otherwise filter
                    if clinical_trial_mcode_elements:
                        filtered_patient_mcode_elements = filter_patient_mcode_elements(
                            patient_mcode_elements, clinical_trial_mcode_elements
                        )
                        logger.info(
                            f"Filtered to {len(filtered_patient_mcode_elements)} elements based on clinical trial criteria"
                        )
                    else:
                        filtered_patient_mcode_elements = patient_mcode_elements
                        logger.info(
                            "Using all extracted mCODE elements (no clinical trial filtering)"
                        )

                    # Create filtered patient data structure
                    filtered_patient_data = patient_data.copy()
                    filtered_patient_data["filtered_mcode_elements"] = (
                        filtered_patient_mcode_elements
                    )
                    filtered_patient_data["source_generator"] = {
                        "archive": archive_identifier,
                        "patient_id": patient_id
                        or generator._extract_patient_id(patient_data),
                        "selection_method": (
                            "random"
                            if random_selection
                            else "specific" if patient_id else "all/limited"
                        ),
                        "total_patients": len(generator),
                        "random_seed": seed,
                    }

                    # Generate output filename
                    if random_selection:
                        base_name = f"random_patient_from_{archive_identifier.replace('/', '_')}_filtered"
                    elif patient_id:
                        base_name = f"patient_{patient_id}_filtered"
                    else:
                        base_name = f"patient_from_{archive_identifier.replace('/', '_')}_{i+1}_filtered"

                    output_file = args.output or f"{base_name}.json"
                    logger.info(f"Writing filtered patient data to: {output_file}")
                    with open(output_file, "w") as f:
                        json.dump(filtered_patient_data, f, indent=2)

                    # Store to CORE Memory if requested
                    if args.store_in - core_memory:
                        logger.info(
                            f"Storing filtered patient {i+1} from generator to CORE Memory"
                        )
                        api_key = config.get_core_memory_api_key()
                        store_filtered_patients_to_memory(
                            filtered_patient_data, api_key
                        )

                    processed_count += 1

                logger.info(
                    f"PatientGenerator processing completed: {processed_count} patients processed from {archive_identifier}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to process archive with PatientGenerator: {str(e)}"
                )
                if args.verbose:
                    import traceback

                    traceback.print_exc()
                sys.exit(1)
        else:
            # Single file mode (unchanged)
            # Read patient data
            logger.info(f"Reading patient data from: {patient_file}")
            with open(patient_file, "r") as f:
                patient_data = json.load(f)

            # Extract mCODE elements from patient data
            patient_mcode_elements = extract_patient_mcode_elements(patient_data)
            logger.info(
                f"Extracted {len(patient_mcode_elements)} mCODE elements from patient data: {list(patient_mcode_elements.keys())}"
            )

            # Use all elements if no clinical trial filtering, otherwise filter
            if clinical_trial_mcode_elements:
                filtered_patient_mcode_elements = filter_patient_mcode_elements(
                    patient_mcode_elements, clinical_trial_mcode_elements
                )
                logger.info(
                    f"Filtered to {len(filtered_patient_mcode_elements)} elements based on clinical trial criteria"
                )
            else:
                filtered_patient_mcode_elements = patient_mcode_elements
                logger.info(
                    "Using all extracted mCODE elements (no clinical trial filtering)"
                )

            # Create filtered patient data structure
            filtered_patient_data = patient_data.copy()
            filtered_patient_data["filtered_mcode_elements"] = (
                filtered_patient_mcode_elements
            )

            # Write filtered patient data to output file
            output_file = args.output or patient_file.replace(".json", "_filtered.json")
            logger.info(f"Writing filtered patient data to: {output_file}")
            with open(output_file, "w") as f:
                json.dump(filtered_patient_data, f, indent=2)

            # Store to CORE Memory if requested
            if args.store_in_core_memory:
                logger.info("Storing filtered patient data to CORE Memory")
                api_key = config.get_core_memory_api_key()
                store_filtered_patients_to_memory(filtered_patient_data, api_key)

            logger.info("Patient mCODE preprocessing completed successfully")

        logger.info("✅ mCODE Patients Preprocessor completed successfully")

    except KeyboardInterrupt:
        logger.info("⏹️  Operation cancelled by user")
        sys.exit(130)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
