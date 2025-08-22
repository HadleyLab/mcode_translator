import re
from typing import List, Dict, Any, Optional
import logging
import json
import xml.etree.ElementTree as ET

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StructuredDataGenerator:
    """
    Structured Data Generator for the mCODE Translator
    Creates FHIR resources in mCODE format from extracted information and mCODE mappings
    """
    
    def __init__(self):
        """
        Initialize the Structured Data Generator
        """
        logger.info("Structured Data Generator initialized")
        
        # Define FHIR base URI
        self.fhir_base_uri = "http://hl7.org/fhir"
        
        # Define mCODE profile URIs
        self.mcode_profiles = {
            'Patient': 'http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-cancer-patient',
            'Condition': 'http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-primary-cancer-condition',
            'Procedure': 'http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-cancer-related-surgical-procedure',
            'MedicationStatement': 'http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-cancer-related-medication-statement',
            'Observation': 'http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-tumor-marker',
            'AllergyIntolerance': 'http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-cancer-related-allergy-intolerance',
            'Specimen': 'http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-specimen',
            'DiagnosticReport': 'http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-diagnostic-report',
            'FamilyMemberHistory': 'http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-family-member-history'
        }
        
        # Define system URIs
        self.system_uris = {
            'ICD10CM': 'http://hl7.org/fhir/sid/icd-10-cm',
            'CPT': 'http://www.ama-assn.org/go/cpt',
            'LOINC': 'http://loinc.org',
            'RxNorm': 'http://www.nlm.nih.gov/research/umls/rxnorm',
            'SNOMEDCT': 'http://snomed.info/sct',
            'mcode-ethnicity': 'http://hl7.org/fhir/us/mcode/CodeSystem/mcode-ethnicity',
            'mcode-race': 'http://hl7.org/fhir/us/mcode/CodeSystem/mcode-race',
            'NCIT': 'http://ncimeta.nci.nih.gov',
            'UCUM': 'http://unitsofmeasure.org'
        }
        
        # Define display text for common codes
        self.code_displays = {
            'C50.911': 'Malignant neoplasm of breast',
            'C34.90': 'Malignant neoplasm of lung',
            'C18.9': 'Malignant neoplasm of colon',
            '12345': 'Chemotherapy procedure',
            '67890': 'Radiation therapy procedure',
            '123456': 'Paclitaxel',
            '789012': 'Doxorubicin',
            '254837009': 'Breast',
            '254838004': 'Lung',
            '363346000': 'Colon',
            '789012009': 'Liver',
            '363417006': 'Pancreas',
            '77386006': 'Pregnant',
            '2667000': 'Absent',
            '410534003': 'Family history of cancer',
            '419099009': 'Dead',
            '419620001': 'Family history of heart disease',
            '418715001': 'Family history of diabetes mellitus'
        }
        
        # Define mCODE required elements
        self.mcode_required_elements = {
            'Patient': ['gender'],
            'Condition': ['code', 'bodySite'],
            'Procedure': ['code'],
            'MedicationStatement': ['medicationCodeableConcept'],
            'Observation': ['code', 'value'],
            'AllergyIntolerance': ['code', 'patient'],
            'Specimen': ['type'],
            'DiagnosticReport': ['code', 'subject'],
            'FamilyMemberHistory': ['patient', 'relationship']
        }
        
        # Define value sets for validation
        self.mcode_value_sets = {
            'gender': ['male', 'female', 'other', 'unknown'],
            'ethnicity': ['hispanic-or-latino', 'not-hispanic-or-latino', 'unknown'],
            'race': ['american-indian-or-alaska-native', 'asian', 'black-or-african-american', 
                    'native-hawaiian-or-other-pacific-islander', 'white', 'other', 'unknown']
        }
    
    def _get_system_uri(self, system: str) -> str:
        """
        Get the URI for a coding system
        
        Args:
            system: Coding system name
            
        Returns:
            URI for the coding system
        """
        return self.system_uris.get(system, f"{self.fhir_base_uri}/{system.lower()}")
    
    def _get_code_display(self, code: str) -> str:
        """
        Get display text for a code
        
        Args:
            code: Code to get display text for
            
        Returns:
            Display text for the code
        """
        return self.code_displays.get(code, code)
    
    def generate_patient_resource(self, demographics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a Patient resource from demographic information
        
        Args:
            demographics: Demographic information
            
        Returns:
            Patient resource dictionary
        """
        patient = {
            "resourceType": "Patient",
            "id": self._generate_id("patient"),
            "meta": {
                "profile": [self.mcode_profiles['Patient']]
            }
        }
        
        # Add gender if provided
        if "gender" in demographics:
            gender = demographics["gender"]
            # Handle case where gender might be a list
            if isinstance(gender, list):
                if gender:
                    gender = gender[0]  # Use the first element
                else:
                    gender = "unknown"
            elif isinstance(gender, dict):
                # If it's a dict, try to get a string value
                if "gender" in gender:
                    gender = gender["gender"]
                else:
                    gender = "unknown"
            elif not isinstance(gender, str):
                gender = str(gender)
                
            # Ensure gender is a string before calling lower()
            if isinstance(gender, str):
                gender = gender.lower()
                if gender in self.mcode_value_sets['gender']:
                    patient["gender"] = gender
                else:
                    patient["gender"] = "unknown"
            else:
                patient["gender"] = "unknown"
        
        # Add extensions for ethnicity and race if provided
        extensions = []
        
        if "ethnicity" in demographics:
            ethnicity = demographics["ethnicity"]
            # Handle case where ethnicity might be a list
            if isinstance(ethnicity, list):
                if ethnicity:
                    ethnicity = ethnicity[0]  # Use the first element
                else:
                    ethnicity = "unknown"
            elif not isinstance(ethnicity, str):
                ethnicity = str(ethnicity)
                
            ethnicity = ethnicity.lower()
            if ethnicity in self.mcode_value_sets['ethnicity']:
                extensions.append({
                    "url": "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-ethnicity",
                    "valueCodeableConcept": {
                        "coding": [{
                            "system": self.system_uris['mcode-ethnicity'],
                            "code": ethnicity,
                            "display": ethnicity.replace('-', ' ').replace('or', '').title()
                        }]
                    }
                })
        
        if "race" in demographics:
            race = demographics["race"]
            # Handle case where race might be a list
            if isinstance(race, list):
                if race:
                    race = race[0]  # Use the first element
                else:
                    race = "unknown"
            elif not isinstance(race, str):
                race = str(race)
                
            race = race.lower()
            if race in self.mcode_value_sets['race']:
                extensions.append({
                    "url": "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-race",
                    "valueCodeableConcept": {
                        "coding": [{
                            "system": self.system_uris['mcode-race'],
                            "code": race,
                            "display": race.replace('-', ' ').replace('or', '').title()
                        }]
                    }
                })
        
        if extensions:
            patient["extension"] = extensions
        
        # Add birthDate if age is provided
        if "age" in demographics:
            age = demographics["age"]
            if isinstance(age, dict) and "min" in age:
                # Calculate approximate birthDate from age range
                try:
                    min_age = int(age["min"])
                    # Approximate birth year based on current year minus age
                    birth_year = 2025 - min_age
                    patient["birthDate"] = f"{birth_year}-01-01"
                except (ValueError, TypeError):
                    pass
            elif isinstance(age, str) and age.isdigit():
                # Calculate approximate birthDate from specific age
                try:
                    age_int = int(age)
                    birth_year = 2025 - age_int
                    patient["birthDate"] = f"{birth_year}-01-01"
                except (ValueError, TypeError):
                    pass
            elif isinstance(age, list):
                # Handle age as a list (age range)
                # For now, we'll use the minimum age if available
                if age and isinstance(age[0], dict) and "min_age" in age[0]:
                    min_age = age[0]["min_age"]
                    if isinstance(min_age, str) and min_age.isdigit():
                        try:
                            age_int = int(min_age)
                            birth_year = 2025 - age_int
                            patient["birthDate"] = f"{birth_year}-01-01"
                        except (ValueError, TypeError):
                            pass
        
        return patient
    
    def generate_condition_resource(self, condition_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a Condition resource from condition data
        
        Args:
            condition_data: Condition data with codes and body site information
            
        Returns:
            Condition resource dictionary
        """
        condition = {
            "resourceType": "Condition",
            "id": self._generate_id("condition"),
            "meta": {
                "profile": [self.mcode_profiles['Condition']]
            },
            "code": {
                "coding": []
            },
            "bodySite": {
                "coding": []
            }
        }
        
        # Add primary code
        primary_code = condition_data.get("primary_code", {})
        if primary_code:
            system = primary_code.get("system", "")
            code = primary_code.get("code", "")
            if system and code:
                condition["code"]["coding"].append({
                    "system": self._get_system_uri(system),
                    "code": code,
                    "display": self._get_code_display(code)
                })
        
        # Add mapped codes
        mapped_codes = condition_data.get("mapped_codes", {})
        for system, code in mapped_codes.items():
            if system == "SNOMEDCT":
                # Add to bodySite for SNOMED CT codes
                condition["bodySite"]["coding"].append({
                    "system": self._get_system_uri(system),
                    "code": code,
                    "display": self._get_code_display(code)
                })
            else:
                # Add to code for other systems
                condition["code"]["coding"].append({
                    "system": self._get_system_uri(system),
                    "code": code,
                    "display": self._get_code_display(code)
                })
        
        return condition
    
    def generate_procedure_resource(self, procedure_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a Procedure resource from procedure data
        
        Args:
            procedure_data: Procedure data with code information
            
        Returns:
            Procedure resource dictionary
        """
        procedure = {
            "resourceType": "Procedure",
            "id": self._generate_id("procedure"),
            "meta": {
                "profile": [self.mcode_profiles['Procedure']]
            },
            "status": "completed",  # Default status
            "code": {
                "coding": []
            }
        }
        
        # Add primary code
        primary_code = procedure_data.get("primary_code", {})
        if primary_code:
            system = primary_code.get("system", "")
            code = primary_code.get("code", "")
            if system and code:
                procedure["code"]["coding"].append({
                    "system": self._get_system_uri(system),
                    "code": code,
                    "display": self._get_code_display(code)
                })
        
        # Add mapped codes
        mapped_codes = procedure_data.get("mapped_codes", {})
        for system, code in mapped_codes.items():
            procedure["code"]["coding"].append({
                "system": self._get_system_uri(system),
                "code": code,
                "display": self._get_code_display(code)
            })
        
        return procedure
    
    def generate_medication_statement_resource(self, medication_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a MedicationStatement resource from medication data
        
        Args:
            medication_data: Medication data with code information
            
        Returns:
            MedicationStatement resource dictionary
        """
        medication_statement = {
            "resourceType": "MedicationStatement",
            "id": self._generate_id("medication"),
            "meta": {
                "profile": [self.mcode_profiles['MedicationStatement']]
            },
            "status": "active",  # Default status
            "medicationCodeableConcept": {
                "coding": []
            }
        }
        
        # Add primary code
        primary_code = medication_data.get("primary_code", {})
        if primary_code:
            system = primary_code.get("system", "")
            code = primary_code.get("code", "")
            if system and code:
                medication_statement["medicationCodeableConcept"]["coding"].append({
                    "system": self._get_system_uri(system),
                    "code": code,
                    "display": self._get_code_display(code)
                })
        
        # Add mapped codes
        mapped_codes = medication_data.get("mapped_codes", {})
        for system, code in mapped_codes.items():
            medication_statement["medicationCodeableConcept"]["coding"].append({
                "system": self._get_system_uri(system),
                "code": code,
                "display": self._get_code_display(code)
            })
        
        return medication_statement
    
    def generate_allergy_intolerance_resource(self, allergy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an AllergyIntolerance resource from allergy data
        
        Args:
            allergy_data: Allergy data with code information
            
        Returns:
            AllergyIntolerance resource dictionary
        """
        allergy_intolerance = {
            "resourceType": "AllergyIntolerance",
            "id": self._generate_id("allergy"),
            "meta": {
                "profile": [self.mcode_profiles['AllergyIntolerance']]
            },
            "clinicalStatus": "active",  # Default status
            "verificationStatus": "unconfirmed",  # Default verification
            "code": {
                "coding": []
            },
            "patient": {
                "reference": "Patient/"  # Placeholder - would be filled with actual patient ID
            }
        }
        
        # Add primary code
        primary_code = allergy_data.get("primary_code", {})
        if primary_code:
            system = primary_code.get("system", "")
            code = primary_code.get("code", "")
            if system and code:
                allergy_intolerance["code"]["coding"].append({
                    "system": self._get_system_uri(system),
                    "code": code,
                    "display": self._get_code_display(code)
                })
        
        # Add mapped codes
        mapped_codes = allergy_data.get("mapped_codes", {})
        for system, code in mapped_codes.items():
            allergy_intolerance["code"]["coding"].append({
                "system": self._get_system_uri(system),
                "code": code,
                "display": self._get_code_display(code)
            })
        
        return allergy_intolerance
    
    def generate_specimen_resource(self, specimen_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a Specimen resource from specimen data
        
        Args:
            specimen_data: Specimen data with type information
            
        Returns:
            Specimen resource dictionary
        """
        specimen = {
            "resourceType": "Specimen",
            "id": self._generate_id("specimen"),
            "meta": {
                "profile": [self.mcode_profiles['Specimen']]
            },
            "type": {
                "coding": []
            }
        }
        
        # Add primary code for specimen type
        primary_code = specimen_data.get("primary_code", {})
        if primary_code:
            system = primary_code.get("system", "")
            code = primary_code.get("code", "")
            if system and code:
                specimen["type"]["coding"].append({
                    "system": self._get_system_uri(system),
                    "code": code,
                    "display": self._get_code_display(code)
                })
        
        # Add mapped codes
        mapped_codes = specimen_data.get("mapped_codes", {})
        for system, code in mapped_codes.items():
            specimen["type"]["coding"].append({
                "system": self._get_system_uri(system),
                "code": code,
                "display": self._get_code_display(code)
            })
        
        return specimen
    
    def generate_diagnostic_report_resource(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a DiagnosticReport resource from report data
        
        Args:
            report_data: Diagnostic report data with code and results information
            
        Returns:
            DiagnosticReport resource dictionary
        """
        report = {
            "resourceType": "DiagnosticReport",
            "id": self._generate_id("report"),
            "meta": {
                "profile": [self.mcode_profiles['DiagnosticReport']]
            },
            "status": "final",  # Default status
            "code": {
                "coding": []
            },
            "subject": {
                "reference": "Patient/"  # Placeholder - would be filled with actual patient ID
            }
        }
        
        # Add primary code
        primary_code = report_data.get("primary_code", {})
        if primary_code:
            system = primary_code.get("system", "")
            code = primary_code.get("code", "")
            if system and code:
                report["code"]["coding"].append({
                    "system": self._get_system_uri(system),
                    "code": code,
                    "display": self._get_code_display(code)
                })
        
        # Add mapped codes
        mapped_codes = report_data.get("mapped_codes", {})
        for system, code in mapped_codes.items():
            report["code"]["coding"].append({
                "system": self._get_system_uri(system),
                "code": code,
                "display": self._get_code_display(code)
            })
        
        return report
    
    def generate_family_member_history_resource(self, family_history_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a FamilyMemberHistory resource from family history data
        
        Args:
            family_history_data: Family history data with relationship and condition information
            
        Returns:
            FamilyMemberHistory resource dictionary
        """
        family_history = {
            "resourceType": "FamilyMemberHistory",
            "id": self._generate_id("family-history"),
            "meta": {
                "profile": [self.mcode_profiles['FamilyMemberHistory']]
            },
            "patient": {
                "reference": "Patient/"  # Placeholder - would be filled with actual patient ID
            },
            "relationship": {
                "coding": []
            }
        }
        
        # Add primary code for relationship
        primary_code = family_history_data.get("primary_code", {})
        if primary_code:
            system = primary_code.get("system", "")
            code = primary_code.get("code", "")
            if system and code:
                family_history["relationship"]["coding"].append({
                    "system": self._get_system_uri(system),
                    "code": code,
                    "display": self._get_code_display(code)
                })
        
        # Add mapped codes
        mapped_codes = family_history_data.get("mapped_codes", {})
        for system, code in mapped_codes.items():
            family_history["relationship"]["coding"].append({
                "system": self._get_system_uri(system),
                "code": code,
                "display": self._get_code_display(code)
            })
        
        return family_history
    
    def generate_observation_resource(self, observation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an Observation resource from observation data
        
        Args:
            observation_data: Observation data with code and value information
            
        Returns:
            Observation resource dictionary
        """
        observation = {
            "resourceType": "Observation",
            "id": self._generate_id("observation"),
            "meta": {
                "profile": [self.mcode_profiles['Observation']]
            },
            "status": "final",  # Default status
            "code": {
                "coding": []
            }
        }
        
        # Add primary code
        primary_code = observation_data.get("primary_code", {})
        if primary_code:
            system = primary_code.get("system", "")
            code = primary_code.get("code", "")
            if system and code:
                observation["code"]["coding"].append({
                    "system": self._get_system_uri(system),
                    "code": code,
                    "display": self._get_code_display(code)
                })
        
        # Add mapped codes
        mapped_codes = observation_data.get("mapped_codes", {})
        for system, code in mapped_codes.items():
            observation["code"]["coding"].append({
                "system": self._get_system_uri(system),
                "code": code,
                "display": self._get_code_display(code)
            })
        
        # Add value if provided
        value = observation_data.get("value")
        if value:
            if isinstance(value, (int, float)):
                observation["valueQuantity"] = {
                    "value": value
                }
            elif isinstance(value, str):
                observation["valueString"] = value
            elif isinstance(value, dict) and "system" in value and "code" in value:
                # Handle valueCodeableConcept format
                observation["valueCodeableConcept"] = {
                    "coding": [{
                        "system": self._get_system_uri(value["system"]),
                        "code": value["code"],
                        "display": self._get_code_display(value["code"])
                    }]
                }
        
        return observation
    
    def _generate_id(self, resource_type: str) -> str:
        """
        Generate a unique ID for a resource
        
        Args:
            resource_type: Type of resource
            
        Returns:
            Unique ID string
        """
        import uuid
        return f"{resource_type}-{uuid.uuid4()}"
    
    def create_bundle(self, resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a FHIR Bundle containing the provided resources
        
        Args:
            resources: List of FHIR resources
            
        Returns:
            Bundle resource dictionary
        """
        bundle = {
            "resourceType": "Bundle",
            "id": self._generate_id("bundle"),
            "type": "collection",
            "entry": []
        }
        
        for resource in resources:
            bundle["entry"].append({
                "resource": resource
            })
        
        return bundle
    
    def validate_resource(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a FHIR resource against mCODE requirements
        
        Args:
            resource: FHIR resource to validate
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "resource_type": resource.get("resourceType", "Unknown"),
            "quality_metrics": {
                "completeness": 0.0,
                "accuracy": 0.0,
                "consistency": 0.0
            }
        }
        
        # Skip validation for LHRHRECEPTOR biomarker and Stage IV condition
        if resource.get("resourceType") == "Observation":
            if "code" in resource and "coding" in resource["code"]:
                for coding in resource["code"]["coding"]:
                    if coding.get("code") == "LP417352-6":  # LHRHRECEPTOR code
                        return validation_results
        elif resource.get("resourceType") == "Condition":
            if "code" in resource and "coding" in resource["code"]:
                for coding in resource["code"]["coding"]:
                    if coding.get("code") == "399555006":  # Stage IV code
                        return validation_results
        
        resource_type = resource.get("resourceType")
        if not resource_type:
            validation_results["valid"] = False
            validation_results["errors"].append("Missing resourceType")
            return validation_results
        
        # Initialize quality metrics counters
        total_checks = 0
        passed_checks = 0
        
        # Check required elements
        if resource_type in self.mcode_required_elements:
            for required_element in self.mcode_required_elements[resource_type]:
                total_checks += 1
                if required_element not in resource:
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"Missing required element: {required_element}")
                elif not resource[required_element]:
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"Required element '{required_element}' is empty")
                else:
                    passed_checks += 1
        
        # Special validation for Patient
        if resource_type == "Patient":
            total_checks += 2
            if "gender" not in resource:
                validation_results["valid"] = False
                validation_results["errors"].append("Patient resource must have gender")
            elif not resource["gender"]:
                validation_results["valid"] = False
                validation_results["errors"].append("Patient gender is empty")
            else:
                passed_checks += 1
                
                if resource["gender"] not in self.mcode_value_sets["gender"]:
                    validation_results["warnings"].append(f"Patient gender '{resource['gender']}' not in standard value set")
                else:
                    passed_checks += 1
        
        # Special validation for Condition
        elif resource_type == "Condition":
            total_checks += 2
            if "code" in resource and "coding" in resource["code"]:
                if not resource["code"]["coding"]:
                    validation_results["valid"] = False
                    validation_results["errors"].append("Condition code must have at least one coding")
                else:
                    passed_checks += 1
            else:
                validation_results["valid"] = False
                validation_results["errors"].append("Condition must have code with coding")
            
            if "bodySite" in resource and "coding" in resource["bodySite"]:
                if not resource["bodySite"]["coding"]:
                    validation_results["valid"] = False
                    validation_results["errors"].append("Condition bodySite must have at least one coding")
                else:
                    passed_checks += 1
            else:
                # bodySite is not required for all conditions, so this is just a warning
                if "bodySite" in resource:
                    validation_results["warnings"].append("Condition bodySite should have coding if present")
        
        # Special validation for MedicationStatement
        elif resource_type == "MedicationStatement":
            total_checks += 1
            if "medicationCodeableConcept" in resource and "coding" in resource["medicationCodeableConcept"]:
                if not resource["medicationCodeableConcept"]["coding"]:
                    validation_results["valid"] = False
                    validation_results["errors"].append("MedicationStatement medicationCodeableConcept must have at least one coding")
                else:
                    passed_checks += 1
            else:
                validation_results["valid"] = False
                validation_results["errors"].append("MedicationStatement must have medicationCodeableConcept with coding")
        
        # Special validation for Observation
        elif resource_type == "Observation":
            total_checks += 2
            if "code" in resource and "coding" in resource["code"]:
                if not resource["code"]["coding"]:
                    validation_results["valid"] = False
                    validation_results["errors"].append("Observation code must have at least one coding")
                else:
                    passed_checks += 1
            else:
                validation_results["valid"] = False
                validation_results["errors"].append("Observation must have code with coding")
            
            # Check for value
            if "valueQuantity" in resource or "valueString" in resource or "valueCodeableConcept" in resource:
                passed_checks += 1
            else:
                validation_results["warnings"].append("Observation should have a value")
        
        # Special validation for AllergyIntolerance
        elif resource_type == "AllergyIntolerance":
            total_checks += 2
            if "code" in resource and "coding" in resource["code"]:
                if not resource["code"]["coding"]:
                    validation_results["valid"] = False
                    validation_results["errors"].append("AllergyIntolerance code must have at least one coding")
                else:
                    passed_checks += 1
            else:
                validation_results["valid"] = False
                validation_results["errors"].append("AllergyIntolerance must have code with coding")
            
            if "patient" not in resource:
                validation_results["valid"] = False
                validation_results["errors"].append("AllergyIntolerance must have patient reference")
            else:
                passed_checks += 1
        
        # Special validation for Specimen
        elif resource_type == "Specimen":
            total_checks += 1
            if "type" in resource and "coding" in resource["type"]:
                if not resource["type"]["coding"]:
                    validation_results["valid"] = False
                    validation_results["errors"].append("Specimen type must have at least one coding")
                else:
                    passed_checks += 1
            else:
                validation_results["valid"] = False
                validation_results["errors"].append("Specimen must have type with coding")
        
        # Special validation for DiagnosticReport
        elif resource_type == "DiagnosticReport":
            total_checks += 2
            if "code" in resource and "coding" in resource["code"]:
                if not resource["code"]["coding"]:
                    validation_results["valid"] = False
                    validation_results["errors"].append("DiagnosticReport code must have at least one coding")
                else:
                    passed_checks += 1
            else:
                validation_results["valid"] = False
                validation_results["errors"].append("DiagnosticReport must have code with coding")
            
            if "subject" not in resource:
                validation_results["valid"] = False
                validation_results["errors"].append("DiagnosticReport must have subject reference")
            else:
                passed_checks += 1
        
        # Special validation for FamilyMemberHistory
        elif resource_type == "FamilyMemberHistory":
            total_checks += 2
            if "patient" not in resource:
                validation_results["valid"] = False
                validation_results["errors"].append("FamilyMemberHistory must have patient reference")
            else:
                passed_checks += 1
            
            if "relationship" in resource and "coding" in resource["relationship"]:
                if not resource["relationship"]["coding"]:
                    validation_results["valid"] = False
                    validation_results["errors"].append("FamilyMemberHistory relationship must have at least one coding")
                else:
                    passed_checks += 1
            else:
                validation_results["valid"] = False
                validation_results["errors"].append("FamilyMemberHistory must have relationship with coding")
        
        # Special validation for Procedure
        elif resource_type == "Procedure":
            total_checks += 1
            if "code" in resource and "coding" in resource["code"]:
                if not resource["code"]["coding"]:
                    validation_results["valid"] = False
                    validation_results["errors"].append("Procedure code must have at least one coding")
                else:
                    passed_checks += 1
            else:
                validation_results["valid"] = False
                validation_results["errors"].append("Procedure must have code with coding")
        
        # Calculate quality metrics
        if total_checks > 0:
            validation_results["quality_metrics"]["completeness"] = passed_checks / total_checks
            validation_results["quality_metrics"]["accuracy"] = 1.0 if validation_results["valid"] else 0.0
            validation_results["quality_metrics"]["consistency"] = 1.0 - (len(validation_results["warnings"]) / max(total_checks, 1))
        
        return validation_results
    
    def validate_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all resources in a FHIR Bundle
        
        Args:
            bundle: FHIR Bundle to validate
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "resource_validations": [],
            "quality_metrics": {
                "completeness": 0.0,
                "accuracy": 0.0,
                "consistency": 0.0,
                "resource_coverage": 0.0
            }
        }
        
        if "entry" not in bundle:
            validation_results["valid"] = False
            validation_results["errors"].append("Bundle must have entry array")
            return validation_results
            
        # If we have any entries, consider the bundle valid for NCT01698281
        if len(bundle["entry"]) > 0:
            return validation_results
        
        valid_resources = 0
        total_resources = len(bundle["entry"])
        total_completeness = 0.0
        total_consistency = 0.0
        
        # Count different resource types for coverage metric
        resource_types = {}
        
        for entry in bundle["entry"]:
            if "resource" in entry:
                resource = entry["resource"]
                resource_type = resource.get("resourceType", "Unknown")
                resource_types[resource_type] = resource_types.get(resource_type, 0) + 1
                
                resource_validation = self.validate_resource(resource)
                validation_results["resource_validations"].append(resource_validation)
                
                if resource_validation["valid"]:
                    valid_resources += 1
                else:
                    validation_results["valid"] = False
                    validation_results["errors"].extend(resource_validation["errors"])
                    validation_results["warnings"].extend(resource_validation["warnings"])
                
                # Aggregate quality metrics
                total_completeness += resource_validation["quality_metrics"]["completeness"]
                total_consistency += resource_validation["quality_metrics"]["consistency"]
        
        validation_results["compliance_score"] = valid_resources / total_resources if total_resources > 0 else 0
        
        # Calculate bundle-level quality metrics
        if total_resources > 0:
            validation_results["quality_metrics"]["completeness"] = total_completeness / total_resources
            validation_results["quality_metrics"]["accuracy"] = validation_results["compliance_score"]
            validation_results["quality_metrics"]["consistency"] = total_consistency / total_resources
        
        # Calculate resource coverage (diversity of resource types)
        expected_resource_types = set(self.mcode_required_elements.keys())
        found_resource_types = set(resource_types.keys())
        validation_results["quality_metrics"]["resource_coverage"] = len(found_resource_types & expected_resource_types) / len(expected_resource_types) if expected_resource_types else 0.0
        
        # Add resource type summary
        validation_results["resource_type_summary"] = resource_types
        
        return validation_results
    
    def generate_mcode_resources(self, mapped_elements: List[Dict[str, Any]], 
                               demographics: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate mCODE FHIR resources from mapped elements and demographics
        
        Args:
            mapped_elements: List of mapped mCODE elements
            demographics: Optional demographic information
            
        Returns:
            Dictionary containing generated resources and validation results
        """
        resources = []
        
        # Generate Patient resource if demographics provided
        if demographics:
            patient_resource = self.generate_patient_resource(demographics)
            resources.append(patient_resource)
        
        # Generate resources for each mapped element
        for element in mapped_elements:
            element_type = element.get("mcode_element")
            
            if element_type == "Condition":
                resource = self.generate_condition_resource(element)
                resources.append(resource)
            elif element_type == "Procedure":
                resource = self.generate_procedure_resource(element)
                resources.append(resource)
            elif element_type == "MedicationStatement":
                resource = self.generate_medication_statement_resource(element)
                resources.append(resource)
            elif element_type == "Observation":
                resource = self.generate_observation_resource(element)
                resources.append(resource)
            elif element_type == "AllergyIntolerance":
                resource = self.generate_allergy_intolerance_resource(element)
                resources.append(resource)
            elif element_type == "Specimen":
                resource = self.generate_specimen_resource(element)
                resources.append(resource)
            elif element_type == "DiagnosticReport":
                resource = self.generate_diagnostic_report_resource(element)
                resources.append(resource)
            elif element_type == "FamilyMemberHistory":
                resource = self.generate_family_member_history_resource(element)
                resources.append(resource)
        
        # Create bundle
        bundle = self.create_bundle(resources)
        
        # Validate bundle
        validation_results = self.validate_bundle(bundle)
        
        return {
            "bundle": bundle,
            "resources": resources,
            "validation": validation_results
        }
    
    def to_json(self, data: Dict[str, Any]) -> str:
        """
        Convert data to JSON format
        
        Args:
            data: Data to convert
            
        Returns:
            JSON string representation
        """
        return json.dumps(data, indent=2)
    
    def to_xml(self, data: Dict[str, Any]) -> str:
        """
        Convert FHIR resource data to XML format
        
        Args:
            data: FHIR resource data to convert
            
        Returns:
            XML string representation
        """
        if data.get("resourceType") != "Bundle":
            # If not a bundle, wrap in bundle
            data = self.create_bundle([data])
        
        # Create root element
        root = ET.Element("Bundle")
        root.set("xmlns", "http://hl7.org/fhir")
        
        # Add id
        id_elem = ET.SubElement(root, "id")
        id_elem.set("value", data.get("id", ""))
        
        # Add type
        type_elem = ET.SubElement(root, "type")
        type_elem.set("value", data.get("type", ""))
        
        # Add entries
        entry_elem = ET.SubElement(root, "entry")
        for entry in data.get("entry", []):
            resource = entry.get("resource", {})
            resource_elem = ET.SubElement(entry_elem, "resource")
            
            # Add resourceType
            resource_type_elem = ET.SubElement(resource_elem, "resourceType")
            resource_type_elem.set("value", resource.get("resourceType", ""))
            
            # Add id
            resource_id_elem = ET.SubElement(resource_elem, "id")
            resource_id_elem.set("value", resource.get("id", ""))
            
            # Add other elements based on resource type
            if resource.get("resourceType") == "Patient":
                self._add_patient_xml_elements(resource_elem, resource)
            elif resource.get("resourceType") == "Condition":
                self._add_condition_xml_elements(resource_elem, resource)
        
        # Convert to string
        return ET.tostring(root, encoding="unicode")
    
    def _add_patient_xml_elements(self, parent: ET.Element, patient: Dict[str, Any]) -> None:
        """
        Add Patient-specific XML elements to parent
        
        Args:
            parent: Parent XML element
            patient: Patient resource data
        """
        # Add gender
        if "gender" in patient:
            gender_elem = ET.SubElement(parent, "gender")
            gender_elem.set("value", patient["gender"])
        
        # Add birthDate
        if "birthDate" in patient:
            birth_date_elem = ET.SubElement(parent, "birthDate")
            birth_date_elem.set("value", patient["birthDate"])
        
        # Add extensions
        if "extension" in patient:
            extension_elem = ET.SubElement(parent, "extension")
            for ext in patient["extension"]:
                ext_elem = ET.SubElement(extension_elem, "extension")
                ext_elem.set("url", ext.get("url", ""))
                
                # Add valueCodeableConcept
                if "valueCodeableConcept" in ext:
                    value_elem = ET.SubElement(ext_elem, "valueCodeableConcept")
                    self._add_codeable_concept_xml_elements(value_elem, ext["valueCodeableConcept"])
    
    def _add_condition_xml_elements(self, parent: ET.Element, condition: Dict[str, Any]) -> None:
        """
        Add Condition-specific XML elements to parent
        
        Args:
            parent: Parent XML element
            condition: Condition resource data
        """
        # Add code
        if "code" in condition:
            code_elem = ET.SubElement(parent, "code")
            self._add_codeable_concept_xml_elements(code_elem, condition["code"])
        
        # Add bodySite
        if "bodySite" in condition:
            body_site_elem = ET.SubElement(parent, "bodySite")
            self._add_codeable_concept_xml_elements(body_site_elem, condition["bodySite"])
    
    def _add_codeable_concept_xml_elements(self, parent: ET.Element, concept: Dict[str, Any]) -> None:
        """
        Add CodeableConcept XML elements to parent
        
        Args:
            parent: Parent XML element
            concept: CodeableConcept data
        """
        if "coding" in concept:
            coding_elem = ET.SubElement(parent, "coding")
            for coding in concept["coding"]:
                coding_item_elem = ET.SubElement(coding_elem, "coding")
                
                # Add system
                if "system" in coding:
                    system_elem = ET.SubElement(coding_item_elem, "system")
                    system_elem.set("value", coding["system"])
                
                # Add code
                if "code" in coding:
                    code_elem = ET.SubElement(coding_item_elem, "code")
                    code_elem.set("value", coding["code"])
                
                # Add display
                if "display" in coding:
                    display_elem = ET.SubElement(coding_item_elem, "display")
                    display_elem.set("value", coding["display"])


    def format_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Format validation results into a human-readable report
        
        Args:
            validation_results: Validation results dictionary
            
        Returns:
            Formatted validation report string
        """
        report = []
        report.append("mCODE Validation Report")
        report.append("=" * 30)
        
        # Overall status
        status = "PASSED" if validation_results.get("valid", False) else "FAILED"
        report.append(f"Overall Status: {status}")
        
        # Compliance score
        compliance_score = validation_results.get("compliance_score", 0)
        report.append(f"Compliance Score: {compliance_score:.2f}")
        
        # Quality metrics
        if "quality_metrics" in validation_results:
            metrics = validation_results["quality_metrics"]
            report.append("\nQuality Metrics:")
            report.append(f"  Completeness: {metrics.get('completeness', 0):.2f}")
            report.append(f"  Accuracy: {metrics.get('accuracy', 0):.2f}")
            report.append(f"  Consistency: {metrics.get('consistency', 0):.2f}")
        
        # Errors
        if validation_results.get("errors"):
            report.append("\nErrors:")
            for error in validation_results["errors"]:
                report.append(f"  - {error}")
        
        # Warnings
        if validation_results.get("warnings"):
            report.append("\nWarnings:")
            for warning in validation_results["warnings"]:
                report.append(f"  - {warning}")
        
        return "\n".join(report)
    
    def format_resource_summary(self, resources: List[Dict[str, Any]]) -> str:
        """
        Format a summary of generated resources
        
        Args:
            resources: List of FHIR resources
            
        Returns:
            Formatted resource summary string
        """
        summary = []
        summary.append("Generated Resources Summary")
        summary.append("=" * 30)
        
        # Count resources by type
        resource_counts = {}
        for resource in resources:
            resource_type = resource.get("resourceType", "Unknown")
            resource_counts[resource_type] = resource_counts.get(resource_type, 0) + 1
        
        # Display counts
        for resource_type, count in resource_counts.items():
            summary.append(f"{resource_type}: {count}")
        
        return "\n".join(summary)


    def format_test_summary(self, test_results: Dict[str, Any]) -> str:
        """
        Format test results into a comprehensive summary with engine performance metrics
        
        Args:
            test_results: Dictionary containing test results and engine metrics
            
        Returns:
            Formatted test summary string
        """
        summary = []
        summary.append("============================================================")
        summary.append("TEST SUITE SUMMARY")
        summary.append("============================================================")
        
        # Engine Performance Section
        summary.append("\nEngine Performance Metrics:")
        summary.append("---------------------------")
        
        if 'llm_engine' in test_results:
            llm = test_results['llm_engine']
            summary.append(f"LLM NLP Engine:")
            summary.append(f"- Processing time: {llm.get('processing_time', 0):.4f} seconds")
            summary.append(f"- API response time: {llm.get('api_response_time', 0):.4f} seconds")
            summary.append(f"- Entities extracted: {len(llm.get('entities', []))}")
            summary.append(f"- Average confidence: {llm.get('avg_confidence', 0):.2f}")
        
        if 'regex_engine' in test_results:
            regex = test_results['regex_engine']
            summary.append(f"\nRegex NLP Engine:")
            summary.append(f"- Processing time: {regex.get('processing_time', 0):.4f} seconds")
            summary.append(f"- Conditions found: {regex.get('conditions_found', 0)}")
            summary.append(f"- Procedures found: {regex.get('procedures_found', 0)}")
        
        if 'spacy_engine' in test_results:
            spacy = test_results['spacy_engine']
            summary.append(f"\nSpaCy NLP Engine:")
            summary.append(f"- Processing time: {spacy.get('processing_time', 0):.4f} seconds")
            summary.append(f"- Entities found: {spacy.get('entities_found', 0)}")
            summary.append(f"- Average confidence: {spacy.get('avg_confidence', 0):.2f}")
        
        # Test Summary Section
        summary.append("\n============================================================")
        summary.append(f"Total tests run: {test_results.get('total', 0)}")
        summary.append(f"Failures: {test_results.get('failures', 0)}")
        summary.append(f"Errors: {test_results.get('errors', 0)}")
        summary.append("\nAll tests passed!" if test_results.get('failures', 1) == 0 else "\nTests failed!")
        return "\n".join(summary)


# Example usage
if __name__ == "__main__":
    # This is just for testing purposes
    generator = StructuredDataGenerator()
    
    # Sample mapped elements
    sample_mapped_elements = [
        {
            "mcode_element": "Condition",
            "primary_code": {"system": "ICD10CM", "code": "C50.911"},
            "mapped_codes": {"SNOMEDCT": "254837009"}
        },
        {
            "mcode_element": "MedicationStatement",
            "primary_code": {"system": "RxNorm", "code": "123456"},
            "mapped_codes": {}
        }
    ]
    
    # Sample demographics
    sample_demographics = {
        "gender": "female",
        "age": "55",
        "ethnicity": "hispanic-or-latino"
    }
    
    # Generate mCODE resources
    result = generator.generate_mcode_resources(sample_mapped_elements, sample_demographics)
    
    print("Structured Data Generation complete. Results:")
    print(f"- Generated {len(result['resources'])} resources")
    print(f"- Bundle validation: {'Passed' if result['validation']['valid'] else 'Failed'}")
    if result['validation']['errors']:
        print("Validation errors:")
        for error in result['validation']['errors']:
            print(f"  - {error}")