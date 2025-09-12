#!/usr/bin/env python3
"""
mCODE Patients Preprocessor - Filter patient mCODE data based on clinical trial elements

Author: mCODE Translation Team
Version: 2.0.0
License: MIT
"""
import json
import logging
import argparse
import sys
import os
from typing import Dict, Any, List, Optional

from src.utils.logging_config import get_logger, setup_logging
from src.utils.config import Config
from src.utils.core_memory_client import CoreMemoryClient
from dotenv import load_dotenv

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
            "snomed_codes": ["254837009", "108294007", "254838001"],  # Breast cancer, malignant neoplasm
            "loinc_codes": ["72113-7"],  # Cancer diagnosis
            "text_patterns": ["breast cancer", "malignant neoplasm", "carcinoma", "neoplasm"],
            "priority_codes": ["C50"],  # ICD-10 breast cancer
            "require_histology": True
        },
        "CancerStage": {
            "resource_types": ["Observation"],
            "loinc_codes": ["21908-9", "59453-7"],  # Stage group, TNM
            "snomed_codes": ["258232001", "385355005"],  # TNM categories
            "text_patterns": ["cancer stage", "tnm stage", "ajcc stage"],
            "related": ["TNMStage"]
        },
        "TNMStage": {
            "resource_types": ["Observation"],
            "loinc_codes": ["59453-7", "21908-9"],  # TNM, Stage group
            "snomed_codes": ["258232001", "261615002", "261634002", "258228008"],  # Stage 1A-4
            "text_patterns": ["tnm", "stage group", "t category", "n category", "m category"],
            "components": ["TumorSize", "LymphNodeInvolvement"]
        },
        
        # Biomarkers & Tumor Markers
        "HER2ReceptorStatus": {
            "resource_types": ["Observation"],
            "loinc_codes": ["48676-1", "85319-2", "85318-4"],  # IHC, FISH
            "snomed_codes": ["108283007"],  # HER2 receptor status
            "text_patterns": ["her2", "HER2", "neu"],
            "value_mappings": {
                "260385009": "Negative", "10828001": "Positive", "413444009": "Equivocal"
            },
            "is_biomarker": True
        },
        "ERReceptorStatus": {
            "resource_types": ["Observation"],
            "loinc_codes": ["85313-5", "48607-9"],  # ER IHC
            "snomed_codes": ["445281001"],  # ER status
            "text_patterns": ["er", "estrogen receptor"],
            "is_biomarker": True
        },
        "PRReceptorStatus": {
            "resource_types": ["Observation"],
            "loinc_codes": ["85314-3"],  # PR IHC
            "snomed_codes": ["445282008"],  # PR status
            "text_patterns": ["pr", "progesterone receptor"],
            "is_biomarker": True
        },
        "TumorMarkerTest": {
            "resource_types": ["Observation"],
            "mcode_profile": "mcode-tumor-marker-test",
            "text_patterns": ["tumor marker", "biomarker", "molecular test"],
            "exclude": ["HER2", "ER", "PR"],  # Handle specifically above
            "is_biomarker": True
        },
        "GenomicVariant": {
            "resource_types": ["Observation"],
            "loinc_codes": ["53037-8", "81347-8"],  # Variant, gene study
            "snomed_codes": ["363344003", "363346002"],  # Genetic analysis
            "text_patterns": ["genomic variant", "mutation", "genetic test"],
            "mcode_profile": "mcode-genomic-variant"
        },
        "HistologyMorphologyBehavior": {
            "resource_types": ["Observation", "Condition"],
            "loinc_codes": ["43986-5"],  # Histology
            "snomed_codes": ["419377000"],  # Histological type
            "text_patterns": ["histology", "morphology", "grade", "differentiation"],
            "mcode_profile": "mcode-histology-morphology"
        },
        
        # Treatments & Procedures
        "CancerTreatment": {
            "resource_types": ["MedicationStatement", "Procedure"],
            "snomed_codes": ["386261006", "367336001", "278850018"],  # Chemo, radiation, immunotherapy
            "text_patterns": ["chemotherapy", "radiation", "immunotherapy", "targeted therapy"],
            "multiple": True  # Array of treatments
        },
        "CancerRelatedMedication": {
            "resource_types": ["MedicationStatement"],
            "snomed_codes": ["763158003"],  # Cancer chemotherapy regimen
            "text_patterns": ["cancer medication", "antineoplastic", "antineoplastic agent"],
            "multiple": True
        },
        "CancerRelatedSurgicalProcedure": {
            "resource_types": ["Procedure"],
            "snomed_codes": ["387713003", "232347006", "712946003"],  # Cancer surgery, mastectomy, reconstruction
            "text_patterns": ["cancer surgery", "mastectomy", "lumpectomy", "resection"],
            "multiple": True
        },
        "RadiationDose": {
            "resource_types": ["Observation"],
            "loinc_codes": ["55335-6"],  # Radiation dose
            "text_patterns": ["radiation dose", "radiation therapy", "rt dose"],
            "expect_quantity": True,
            "multiple": True
        },
        
        # Adverse Events & Safety
        "AdverseEvent": {
            "resource_types": ["Observation"],
            "loinc_codes": ["75276-9"],  # Adverse event
            "snomed_codes": ["62315001"],  # Adverse reaction
            "text_patterns": ["adverse event", "toxicity", "side effect", "ae"],
            "mcode_profile": "mcode-adverse-event",
            "multiple": True
        },
        
        # Comorbidities & Performance
        "CancerComorbidity": {
            "resource_types": ["Condition"],
            "snomed_codes": ["64572001"],  # Comorbidity
            "text_patterns": ["comorbidity", "concurrent condition"],
            "related_to": "CancerCondition"
        },
        "ECOGPerformanceStatus": {
            "resource_types": ["Observation"],
            "loinc_codes": ["89250-6"],  # ECOG score
            "text_patterns": ["ecog", "performance status"],
            "snomed_codes": ["288528006"]
        },
        
        # Patient Demographics
        "Patient": {
            "resource_types": ["Patient"],
            "fields": ["gender", "birthDate", "deceasedBoolean", "multipleBirthBoolean"]
        },
        "PatientAgeGroup": {
            "resource_types": ["Patient"],
            "calculated": True,
            "age_ranges": {
                "<18": "Pediatric",
                "18-39": "Young Adult",
                "40-64": "Adult",
                "65+": "Senior"
            }
        },
        "PatientSex": {
            "resource_types": ["Patient"],
            "path": "gender"
        },
        "Race": {
            "resource_types": ["Patient"],
            "extension_url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race"
        },
        "Ethnicity": {
            "resource_types": ["Patient"],
            "extension_url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity"
        }
    }
    
    # Process each entry in the Bundle
    for entry in patient_data.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")
        
        # Special handling for Patient demographics (always first)
        if resource_type == "Patient" and not any(k in mcode_profile for k in ["PatientSex", "Race", "Ethnicity"]):
            demographics = extract_demographics(resource)
            mcode_profile.update(demographics)
            continue
        
        # Check each mCODE mapping for other resource types
        for mcode_key, mapping in mcode_mappings.items():
            if resource_type in mapping.get("resource_types", []) and mcode_key not in mcode_profile:
                matched = match_mcode_resource(resource, mapping)
                if matched:
                    value = extract_mcode_value(resource, mcode_key, mapping)
                    if value:
                        mcode_profile[mcode_key] = value
                        logger.debug(f"Extracted {mcode_key}: {value}")
                        break  # One match per element
    
    logger.info(f"Extracted {len(mcode_profile)} comprehensive mCODE elements from patient data: {list(mcode_profile.keys())}")
    return mcode_profile


def match_mcode_resource(resource: Dict, mapping: Dict) -> bool:
    """Determine if resource matches mCODE mapping criteria."""
    code = resource.get("code", {})
    coding = code.get("coding", [])
    display = code.get("text", "").lower()
    
    # Check LOINC codes
    for c in coding:
        if c.get("system") == "http://loinc.org" and c.get("code") in mapping.get("loinc_codes", []):
            return True
    
    # Check SNOMED codes
    for c in coding:
        if c.get("system") == "http://snomed.info/sct" and c.get("code") in mapping.get("snomed_codes", []):
            return True
    
    # Check mCODE profile
    if mapping.get("mcode_profile") and mapping.get("mcode_profile") in resource.get("meta", {}).get("profile", []):
        return True
    
    # Text pattern matching
    patterns = mapping.get("text_patterns", [])
    if patterns and any(pattern in display for pattern in patterns):
        return True
    
    # For CancerCondition, check if it's cancer-related
    if mapping.get("key") == "CancerCondition":
        for c in coding:
            # Breast cancer codes
            if c.get("system") == "http://snomed.info/sct" and c.get("code") in ["254837009", "108294007"]:
                return True
            # ICD-10 breast cancer
            if "C50" in c.get("code", ""):
                return True
            # Text indicates cancer
            if any(term in c.get("display", "").lower() for term in ["breast cancer", "malignant neoplasm"]):
                return True
    
    return False


def extract_mcode_value(resource: Dict, mcode_key: str, mapping: Dict) -> Optional[Dict]:
    """Extract structured value from resource for specific mCODE element."""
    value_data = {
        "system": None, "code": None, "display": None,
        "interpretation": None, "quantity": None,
        "date": resource.get("effectiveDateTime") or resource.get("issued"),
        "reference": resource.get("id")
    }
    
    code = resource.get("code", {})
    coding = code.get("coding", [])
    
    # Find the best coding match for identification
    best_coding = None
    for c in coding:
        if match_mcode_resource({"code": {"coding": [c]}}, mapping):
            best_coding = c
            break
    
    if best_coding:
        value_data["system"] = best_coding.get("system")
        value_data["code"] = best_coding.get("code")
        value_data["display"] = best_coding.get("display") or code.get("text")
    
    # For biomarkers, prioritize valueCodeableConcept (interpretation)
    if mapping.get("is_biomarker") and resource.get("valueCodeableConcept"):
        vc = resource.get("valueCodeableConcept")
        vc_coding = vc.get("coding", [{}])[0]
        value_data["interpretation_system"] = vc_coding.get("system")
        value_data["interpretation_code"] = vc_coding.get("code")
        value_data["interpretation"] = vc_coding.get("display") or vc.get("text")
        
        # Map SNOMED interpretations
        if vc_coding.get("system") == "http://snomed.info/sct":
            snomed_map = mapping.get("value_mappings", {})
            mapped = snomed_map.get(vc_coding.get("code"))
            if mapped:
                value_data["interpretation"] = mapped
    
    # For measurements, use valueQuantity
    elif mapping.get("expect_quantity") and resource.get("valueQuantity"):
        vq = resource.get("valueQuantity")
        value_data["quantity"] = {
            "value": vq.get("value"),
            "unit": vq.get("unit"),
            "code": vq.get("code")
        }
        value_data["display"] = f"{vq.get('value')} {vq.get('unit')}"
    
    # Default display from code text if no specific value
    elif not value_data["display"] and code.get("text"):
        value_data["display"] = code.get("text")
    
    # Return if we have meaningful data
    if (value_data["code"] or value_data["display"] or
        value_data["interpretation"] or value_data["quantity"]):
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
            "display": {"male": "Male", "female": "Female", "other": "Other", "unknown": "Unknown"}.get(gender, gender)
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
                "display": coding.get("display", "Unknown")
            }
        elif url == "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity":
            coding = ext.get("valueCoding", {})
            demographics["Ethnicity"] = {
                "system": coding.get("system"),
                "code": coding.get("code"),
                "display": coding.get("display", "Unknown")
            }
    
    return demographics

def filter_patient_mcode_elements(patient_mcode_elements: Dict[str, Any],
                                 clinical_trial_mcode_elements: Dict[str, Any]) -> Dict[str, Any]:
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
    
    logger.info(f"Filtered {len(filtered_elements)}/{len(patient_mcode_elements)} patient mCODE elements")
    return filtered_elements


def extract_mcode_value(resource: Dict, mcode_key: str, mapping: Dict) -> Optional[Dict]:
    """Extract structured value from resource for specific mCODE element."""
    value_data = {
        "system": None, "code": None, "display": None,
        "interpretation": None, "quantity": None,
        "date": resource.get("effectiveDateTime") or resource.get("issued"),
        "reference": resource.get("id")
    }
    
    code = resource.get("code", {})
    coding = code.get("coding", [{}])[0] if code.get("coding") else {}
    
    # Set basic coding info
    value_data["system"] = coding.get("system")
    value_data["code"] = coding.get("code")
    value_data["display"] = coding.get("display") or code.get("text")
    
    # ValueCodeableConcept (interpretations like Positive/Negative)
    if resource.get("valueCodeableConcept"):
        vc = resource.get("valueCodeableConcept")
        coding = vc.get("coding", [{}])[0]
        value_data["system"] = coding.get("system") or value_data["system"]
        value_data["code"] = coding.get("code") or value_data["code"]
        value_data["display"] = coding.get("display") or vc.get("text")
        
        # Map SNOMED interpretations
        if value_data["system"] == "http://snomed.info/sct":
            snomed_map = mapping.get("value_mappings", {})
            value_data["interpretation"] = snomed_map.get(value_data["code"], value_data["display"])
    
    # Quantity (measurements like tumor size)
    elif resource.get("valueQuantity"):
        vq = resource.get("valueQuantity")
        value_data["quantity"] = {
            "value": vq.get("value"),
            "unit": vq.get("unit"),
            "code": vq.get("code")
        }
        value_data["display"] = f"{vq.get('value')} {vq.get('unit')}"
    
    # Reference (procedures, medications)
    elif code.get("text"):
        value_data["display"] = code.get("text")
    
    # For Patient demographics, special handling
    elif mcode_key in ["PatientSex", "Race", "Ethnicity"]:
        pass  # Handled in extract_demographics
    
    return value_data if any([value_data["code"], value_data["display"], value_data["quantity"], value_data["interpretation"]]) else None


def extract_demographics(patient_resource: Dict) -> Dict:
    """Extract basic demographics as mCODE elements."""
    demographics = {}
    
    # PatientSex
    gender = patient_resource.get("gender")
    if gender:
        demographics["PatientSex"] = {
            "value": gender,
            "system": "http://hl7.org/fhir/administrative-gender",
            "display": {"male": "Male", "female": "Female", "other": "Other", "unknown": "Unknown"}.get(gender, gender)
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
                "display": coding.get("display", "Unknown")
            }
        elif url == "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity":
            coding = ext.get("valueCoding", {})
            demographics["Ethnicity"] = {
                "system": coding.get("system"),
                "code": coding.get("code"),
                "display": coding.get("display", "Unknown")
            }
    
    return demographics

def store_filtered_patients_to_memory(filtered_data: Dict[str, Any], api_key: str) -> None:
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
                name = " ".join(resource.get("name", [{}])[0].get("given", [])) + " " + resource.get("name", [{}])[0].get("family", "")
                dob = resource.get("birthDate")
                gender = resource.get("gender")
                race = next((ext.get("extension", [{}])[0].get("valueCoding", {}).get("display") for ext in resource.get("extension", []) if ext.get("url") == "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race"), "Unknown")
                ethnicity = next((ext.get("extension", [{}])[0].get("valueCoding", {}).get("display") for ext in resource.get("extension", []) if ext.get("url") == "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity"), "Unknown")
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
                    "id": patient_id  # Store ID for reference matching
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
            if f"Patient/{patient_id}" in subject_ref or patient_id in subject_ref:  # Match by subject reference
                if rtype == "Procedure":
                    proc_code = resource.get("code", {}).get("text", "")
                    patients[patient_id]["procedures"].append(proc_code)
                elif rtype == "Observation" and "vital-signs" in resource.get("category", [{}])[0].get("coding", [{}])[0].get("code", ""):
                    code_text = resource.get("code", {}).get("text", "")
                    value = resource.get("valueQuantity", {}).get("value")
                    unit = resource.get("valueQuantity", {}).get("unit", "")
                    patients[patient_id]["vitals"][code_text] = f"{value}{unit}"
                elif rtype == "Condition":
                    cond_code = resource.get("code", {}).get("text", "")
                    patients[patient_id]["conditions"].append(cond_code)

        # Deduplicate procedures
        patients[patient_id]["procedures"] = list(set(patients[patient_id]["procedures"]))
        
        # Build and ingest per-patient summaries
        for pid, data in patients.items():
            summary_lines = [
                f"Patient {data['name']}:",
                f"Gender: {data['gender']}. Date of birth: {data['dob']}. Race: {data['race']}. Ethnicity: {data['ethnicity']}.",
                "MCODE elements:"
            ]
            mcode = data["mcode"]
            if "CancerCondition" in mcode:
                cc = mcode["CancerCondition"]
                summary_lines.append(f"  CancerCondition: {cc.get('display', '')} ({cc.get('code', '')})")
            if "TNMStage" in mcode:
                ts = mcode["TNMStage"]
                summary_lines.append(f"  TNMStage: {ts.get('display', '')} ({ts.get('code', '')})")
            if "TumorMarker" in mcode:
                tm = mcode["TumorMarker"]
                summary_lines.append(f"  TumorMarker: {tm.get('display', '')} ({tm.get('code', '')})")
            if "HER2ReceptorStatus" in mcode:
                her2 = mcode["HER2ReceptorStatus"]
                her2_status = her2.get('interpretation', her2.get('display', 'Unknown'))
                summary_lines.append(f"  HER2ReceptorStatus: {her2_status} ({her2.get('code', '')})")
            if "TumorSize" in mcode:
                size = mcode["TumorSize"]
                if size.get('quantity'):
                    summary_lines.append(f"  TumorSize: {size['quantity']['value']} {size['quantity']['unit']} ({size.get('code', '')})")
                else:
                    summary_lines.append(f"  TumorSize: {size.get('display', 'N/A')}")
            if "LymphNodeInvolvement" in mcode:
                lymph = mcode["LymphNodeInvolvement"]
                if lymph.get('quantity'):
                    summary_lines.append(f"  LymphNodeInvolvement: {lymph['quantity']['value']} ({lymph.get('code', '')})")
                else:
                    summary_lines.append(f"  LymphNodeInvolvement: {lymph.get('display', 'N/A')}")
            if data["procedures"]:
                # Limit procedures to cancer-related ones
                cancer_procs = [p for p in data["procedures"] if any(term in p.lower() for term in ["cancer", "breast", "tumor", "chemo", "radiation", "surgery", "biopsy", "her2"])]
                if cancer_procs:
                    procs_str = ", ".join(cancer_procs[:10])  # Top 10
                    if len(cancer_procs) > 10:
                        procs_str += f" + {len(cancer_procs)-10} more"
                    summary_lines.append(f"Key Cancer Procedures: {procs_str}")
                else:
                    procs = ", ".join(data["procedures"][:5])  # Top 5 if no cancer-specific
                    summary_lines.append(f"Procedures: {procs}")
            if data["conditions"]:
                # Prioritize cancer-related conditions
                cancer_conds = [c for c in data["conditions"] if any(term in c.lower() for term in ["cancer", "breast", "malignant", "neoplasm", "tumor"])]
                if cancer_conds:
                    conds_str = ", ".join(cancer_conds)
                    summary_lines.append(f"Cancer Conditions: {conds_str}")
                else:
                    conds = ", ".join(data["conditions"][:5])
                    summary_lines.append(f"Conditions: {conds}")
            if data["vitals"]:
                # Key cancer-relevant vitals
                key_vitals = {k: v for k, v in data["vitals"].items() if any(term in k.lower() for term in ["weight", "bmi", "height"])}
                if key_vitals:
                    vitals_str = ", ".join([f"{k}: {v}" for k, v in key_vitals.items()])
                    summary_lines.append(f"Key Vitals: {vitals_str}")
                else:
                    vitals_str = ", ".join([f"{k}: {v}" for k, v in list(data["vitals"].items())[:5]])  # Top 5
                    summary_lines.append(f"Vitals: {vitals_str}")
            
            summary = " ".join(summary_lines) + "."
            logger.debug(f"CORE Memory summary for patient {pid}: {summary}")
            client.ingest(summary)
            logger.info(f"Stored summary for patient {pid} ({data['name']}) in CORE Memory")
        
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
  mcode_patients.py --clinical-trial-file data/fetcher_output/deepseek-chat.results.json \
    --patient-file patient.pruned.for_deepseek.json --output-file patient.filtered.for_deepseek.json

  # Batch processing with directory recursion
  mcode_patients.py --clinical-trial-file data/fetcher_output/deepseek-chat.results.json \
    --root-dir data/mcode_downloads --output-dir data/mcode_filtered --workers 4

  # With CORE Memory storage
  mcode_patients.py --clinical-trial-file data/fetcher_output/deepseek-chat.results.json \
    --patient-file patient.pruned.for_deepseek.json --store-in-core-memory --verbose
        """
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--clinical-trial-file", "-c",
        type=str, default="data/fetcher_output/deepseek-chat.results.json",
        help="Path to clinical trial data file (default: data/fetcher_output/deepseek-chat.results.json)"
    )
    input_group.add_argument(
        "input_file",
        nargs="?",
        help="Patient data file (alternative to --patient-file)"
    )
    parser.add_argument(
        "--patient-file", "-p",
        type=str, default="patient.pruned.for_deepseek.json",
        help="Path to single patient data file (default: patient.pruned.for_deepseek.json). Use --input-dir for batch."
    )
    parser.add_argument(
        "--input-dir", "-d",
        type=str, default=None,
        help="Input directory to recurse for patient .json files. Overrides --patient-file."
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="Output file for single file mode. For batch, use --output-dir."
    )
    parser.add_argument(
        "--output-dir", "-O",
        type=str, default=None,
        help="Output directory for batch mode. Filtered files saved with '_filtered' suffix."
    )
    parser.add_argument(
        "--store-in-core-memory",
        action="store_true",
        help="Store filtered patient summaries to CORE Memory"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int, default=os.cpu_count() or 4,
        help="Number of parallel workers for batch processing (default: CPU count)"
    )
    # Universal flags
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (overrides default)"
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
        
        # Handle input selection
        if args.input_file:
            clinical_trial_file = args.input_file
        else:
            clinical_trial_file = args.clinical_trial_file
        
        patient_file = args.patient_file  # Always set patient_file for single mode
        
        # Read clinical trial data (single file)
        logger.info(f"Reading clinical trial data from: {clinical_trial_file}")
        with open(clinical_trial_file, 'r') as f:
            clinical_trial_data = json.load(f)
        
        # Extract mCODE elements from clinical trial data
        clinical_trial_mcode_elements = extract_clinical_trial_mcode_elements(clinical_trial_data)
        logger.info(f"Extracted {len(clinical_trial_mcode_elements)} mCODE elements from clinical trial data: {list(clinical_trial_mcode_elements.keys())}")
        
        if args.input_dir:
            # Batch mode: Recurse through input_dir for .json files
            processed_count = 0
            import os
            for root, dirs, files in os.walk(args.input_dir):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        logger.info(f"Processing batch file: {file_path}")
                        with open(file_path, 'r') as f:
                            patient_data = json.load(f)
                        
                        # Extract and filter
                        patient_mcode_elements = extract_patient_mcode_elements(patient_data)
                        logger.info(f"Extracted {len(patient_mcode_elements)} mCODE elements from {file}: {list(patient_mcode_elements.keys())}")
                        
                        filtered_patient_mcode_elements = filter_patient_mcode_elements(patient_mcode_elements, clinical_trial_mcode_elements)
                        
                        # Create filtered data
                        filtered_patient_data = patient_data.copy()
                        filtered_patient_data["filtered_mcode_elements"] = filtered_patient_mcode_elements
                        
                        # Mirror output structure
                        rel_path = os.path.relpath(file_path, args.input_dir)
                        output_path = os.path.join(args.output_dir or args.output, rel_path.replace('.json', '_filtered.json'))
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        with open(output_path, 'w') as f:
                            json.dump(filtered_patient_data, f, indent=2)
                        logger.info(f"Saved filtered data to: {output_path}")
                        
                        # Store to CORE Memory if requested (parallelizable with --workers, but sequential for simplicity)
                        if args.store_in_core_memory:
                            api_key = config.get_core_memory_api_key()
                            store_filtered_patients_to_memory(filtered_patient_data, api_key)
                        
                        processed_count += 1
            
            logger.info(f"Batch processing completed: {processed_count} files processed")
        else:
            # Single file mode
            # Read patient data
            logger.info(f"Reading patient data from: {patient_file}")
            with open(patient_file, 'r') as f:
                patient_data = json.load(f)
            
            # Extract mCODE elements from patient data
            patient_mcode_elements = extract_patient_mcode_elements(patient_data)
            logger.info(f"Extracted {len(patient_mcode_elements)} mCODE elements from patient data: {list(patient_mcode_elements.keys())}")
            
            # Filter patient mCODE elements based on clinical trial elements
            filtered_patient_mcode_elements = filter_patient_mcode_elements(patient_mcode_elements, clinical_trial_mcode_elements)
            
            # Create filtered patient data structure
            filtered_patient_data = patient_data.copy()
            filtered_patient_data["filtered_mcode_elements"] = filtered_patient_mcode_elements
            
            # Write filtered patient data to output file
            output_file = args.output or patient_file.replace('.json', '_filtered.json')
            logger.info(f"Writing filtered patient data to: {output_file}")
            with open(output_file, 'w') as f:
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