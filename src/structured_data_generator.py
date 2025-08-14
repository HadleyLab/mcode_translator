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
            'Observation': 'http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-tumor-marker'
        }
        
        # Define system URIs
        self.system_uris = {
            'ICD10CM': 'http://hl7.org/fhir/sid/icd-10-cm',
            'CPT': 'http://www.ama-assn.org/go/cpt',
            'LOINC': 'http://loinc.org',
            'RxNorm': 'http://www.nlm.nih.gov/research/umls/rxnorm',
            'SNOMEDCT': 'http://snomed.info/sct',
            'mcode-ethnicity': 'http://hl7.org/fhir/us/mcode/CodeSystem/mcode-ethnicity',
            'mcode-race': 'http://hl7.org/fhir/us/mcode/CodeSystem/mcode-race'
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
            '363346000': 'Colon'
        }
        
        # Define mCODE required elements
        self.mcode_required_elements = {
            'Patient': ['gender'],
            'Condition': ['code', 'bodySite'],
            'Procedure': ['code'],
            'MedicationStatement': ['medicationCodeableConcept'],
            'Observation': ['code', 'value']
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
            gender = demographics["gender"].lower()
            if gender in self.mcode_value_sets['gender']:
                patient["gender"] = gender
            else:
                patient["gender"] = "unknown"
        
        # Add extensions for ethnicity and race if provided
        extensions = []
        
        if "ethnicity" in demographics:
            ethnicity = demographics["ethnicity"].lower()
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
            race = demographics["race"].lower()
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
            "resource_type": resource.get("resourceType", "Unknown")
        }
        
        resource_type = resource.get("resourceType")
        if not resource_type:
            validation_results["valid"] = False
            validation_results["errors"].append("Missing resourceType")
            return validation_results
        
        # Check required elements
        if resource_type in self.mcode_required_elements:
            for required_element in self.mcode_required_elements[resource_type]:
                if required_element not in resource:
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"Missing required element: {required_element}")
                elif not resource[required_element]:
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"Required element '{required_element}' is empty")
        
        # Special validation for Patient
        if resource_type == "Patient":
            if "gender" not in resource:
                validation_results["valid"] = False
                validation_results["errors"].append("Patient resource must have gender")
            elif resource["gender"] not in self.mcode_value_sets["gender"]:
                validation_results["warnings"].append(f"Patient gender '{resource['gender']}' not in standard value set")
        
        # Special validation for Condition
        elif resource_type == "Condition":
            if "code" in resource and "coding" in resource["code"]:
                if not resource["code"]["coding"]:
                    validation_results["valid"] = False
                    validation_results["errors"].append("Condition code must have at least one coding")
            
            if "bodySite" in resource and "coding" in resource["bodySite"]:
                if not resource["bodySite"]["coding"]:
                    validation_results["valid"] = False
                    validation_results["errors"].append("Condition bodySite must have at least one coding")
        
        # Special validation for MedicationStatement
        elif resource_type == "MedicationStatement":
            if "medicationCodeableConcept" in resource and "coding" in resource["medicationCodeableConcept"]:
                if not resource["medicationCodeableConcept"]["coding"]:
                    validation_results["valid"] = False
                    validation_results["errors"].append("MedicationStatement medicationCodeableConcept must have at least one coding")
        
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
            "resource_validations": []
        }
        
        if "entry" not in bundle:
            validation_results["valid"] = False
            validation_results["errors"].append("Bundle must have entry array")
            return validation_results
        
        valid_resources = 0
        total_resources = len(bundle["entry"])
        
        for entry in bundle["entry"]:
            if "resource" in entry:
                resource_validation = self.validate_resource(entry["resource"])
                validation_results["resource_validations"].append(resource_validation)
                
                if resource_validation["valid"]:
                    valid_resources += 1
                else:
                    validation_results["valid"] = False
                    validation_results["errors"].extend(resource_validation["errors"])
                    validation_results["warnings"].extend(resource_validation["warnings"])
        
        validation_results["compliance_score"] = valid_resources / total_resources if total_resources > 0 else 0
        
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