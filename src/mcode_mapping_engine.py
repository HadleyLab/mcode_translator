import re
from typing import List, Dict, Any, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCODEMappingEngine:
    """
    mCODE Mapping Engine for the mCODE Translator
    Maps extracted concepts to mCODE data elements and standard codes
    """
    
    def __init__(self):
        """
        Initialize the mCODE Mapping Engine
        """
        logger.info("mCODE Mapping Engine initialized")
        
        # Define mCODE required elements
        self.mcode_elements = {
            'Patient': {
                'required': ['gender', 'birthDate'],
                'optional': ['extension']
            },
            'Condition': {
                'required': ['code', 'bodySite'],
                'optional': ['onsetDateTime', 'recordedDate']
            },
            'Procedure': {
                'required': ['code'],
                'optional': ['performedDateTime', 'status']
            },
            'MedicationStatement': {
                'required': ['medicationCodeableConcept'],
                'optional': ['effectiveDateTime', 'status']
            }
        }
        
        # Define mCODE required codes
        self.mcode_required_codes = {
            'ICD10CM': ['C50.911', 'C34.90', 'C18.9', 'C22.0', 'C25.9'],
            'CPT': ['12345', '67890'],
            'LOINC': ['12345-6', '78901-2'],
            'RxNorm': ['123456', '789012']
        }
        
        # Define cross-walk mappings between coding systems
        self.cross_walks = {
            # ICD-10-CM to SNOMED CT mappings
            ('C50.911', 'ICD10CM', 'SNOMEDCT'): '254837009',  # Breast cancer
            ('C34.90', 'ICD10CM', 'SNOMEDCT'): '254838004',  # Lung cancer
            ('C18.9', 'ICD10CM', 'SNOMEDCT'): '363346000',   # Colorectal cancer
            ('C22.0', 'ICD10CM', 'SNOMEDCT'): '109838000',   # Liver cancer
            ('C25.9', 'ICD10CM', 'SNOMEDCT'): '363417006',   # Pancreatic cancer
            
            # ICD-10-CM to LOINC mappings
            ('C50.911', 'ICD10CM', 'LOINC'): 'LP12345-6',    # Breast cancer
            ('C34.90', 'ICD10CM', 'LOINC'): 'LP78901-2',     # Lung cancer
            
            # RxNorm to SNOMED CT mappings
            ('123456', 'RxNorm', 'SNOMEDCT'): '386906001',   # Paclitaxel
            ('789012', 'RxNorm', 'SNOMEDCT'): '386907005',   # Doxorubicin
        }
        
        # Define mCODE element mappings
        self.mcode_element_mappings = {
            'breast cancer': {
                'mcode_element': 'Condition',
                'primary_code': {'system': 'ICD10CM', 'code': 'C50.911'},
                'mapped_codes': {
                    'SNOMEDCT': '254837009',
                    'LOINC': 'LP12345-6'
                }
            },
            'lung cancer': {
                'mcode_element': 'Condition',
                'primary_code': {'system': 'ICD10CM', 'code': 'C34.90'},
                'mapped_codes': {
                    'SNOMEDCT': '254838004',
                    'LOINC': 'LP78901-2'
                }
            },
            'colorectal cancer': {
                'mcode_element': 'Condition',
                'primary_code': {'system': 'ICD10CM', 'code': 'C18.9'},
                'mapped_codes': {
                    'SNOMEDCT': '363346000'
                }
            },
            'paclitaxel': {
                'mcode_element': 'MedicationStatement',
                'primary_code': {'system': 'RxNorm', 'code': '123456'},
                'mapped_codes': {
                    'SNOMEDCT': '386906001'
                }
            },
            'doxorubicin': {
                'mcode_element': 'MedicationStatement',
                'primary_code': {'system': 'RxNorm', 'code': '789012'},
                'mapped_codes': {
                    'SNOMEDCT': '386907005'
                }
            },
            'chemotherapy': {
                'mcode_element': 'Procedure',
                'primary_code': {'system': 'CPT', 'code': '12345'},
                'mapped_codes': {}
            },
            'radiation therapy': {
                'mcode_element': 'Procedure',
                'primary_code': {'system': 'CPT', 'code': '67890'},
                'mapped_codes': {}
            }
        }
        
        # Define value sets for mCODE compliance
        self.mcode_value_sets = {
            'gender': ['male', 'female', 'other', 'unknown'],
            'ethnicity': ['hispanic-or-latino', 'not-hispanic-or-latino', 'unknown'],
            'race': ['american-indian-or-alaska-native', 'asian', 'black-or-african-american', 
                    'native-hawaiian-or-other-pacific-islander', 'white', 'other', 'unknown']
        }
        
        # Define mapping rules
        self.mapping_rules = {
            'condition_mapping': {
                'min_confidence': 0.7,
                'required_fields': ['code', 'bodySite']
            },
            'procedure_mapping': {
                'min_confidence': 0.6,
                'required_fields': ['code']
            },
            'medication_mapping': {
                'min_confidence': 0.7,
                'required_fields': ['medicationCodeableConcept']
            }
        }
    
    def map_concept_to_mcode(self, concept: str, confidence: float = 0.8) -> Optional[Dict[str, Any]]:
        """
        Map a concept to an mCODE element
        
        Args:
            concept: Concept to map
            confidence: Confidence score for the mapping
            
        Returns:
            Dictionary with mCODE element information or None if no mapping found
        """
        concept_lower = concept.lower()
        
        # Check if concept exists in mappings
        if concept_lower in self.mcode_element_mappings:
            mapping = self.mcode_element_mappings[concept_lower].copy()
            mapping['confidence'] = confidence
            mapping['mapped_from'] = concept
            return mapping
        
        return None
    
    def map_code_to_mcode(self, code: str, system: str) -> Optional[Dict[str, Any]]:
        """
        Map a code to mCODE elements and cross-walks
        
        Args:
            code: Code to map
            system: Coding system of the code
            
        Returns:
            Dictionary with mCODE mapping information or None if no mapping found
        """
        # Check if code is mCODE required
        is_mcode_required = False
        if system in self.mcode_required_codes:
            is_mcode_required = code in self.mcode_required_codes[system]
        
        # Get cross-walk mappings
        mapped_codes = {}
        for (source_code, source_system, target_system), target_code in self.cross_walks.items():
            if source_code == code and source_system == system:
                mapped_codes[target_system] = target_code
        
        # Determine mCODE element type based on system and code
        mcode_element = self._determine_mcode_element(code, system)
        
        return {
            'code': code,
            'system': system,
            'mcode_required': is_mcode_required,
            'mapped_codes': mapped_codes,
            'mcode_element': mcode_element
        }
    
    def _determine_mcode_element(self, code: str, system: str) -> str:
        """
        Determine the appropriate mCODE element for a code
        
        Args:
            code: Code to determine element for
            system: Coding system of the code
            
        Returns:
            mCODE element type
        """
        # Simple mapping logic based on system
        element_mapping = {
            'ICD10CM': 'Condition',
            'CPT': 'Procedure',
            'LOINC': 'Observation',
            'RxNorm': 'MedicationStatement'
        }
        
        return element_mapping.get(system, 'Observation')
    
    def validate_mcode_compliance(self, mcode_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate mCODE data for compliance with mCODE standards
        
        Args:
            mcode_data: Dictionary containing mCODE data
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'compliance_score': 0.0
        }
        
        # Check required elements only for elements that are present in the data
        total_required = 0
        satisfied_required = 0
        
        for element_type, element_info in self.mcode_elements.items():
            # Only check elements that are actually present in the data
            if element_type in mcode_data:
                total_required += len(element_info['required'])
                for required_field in element_info['required']:
                    if required_field in mcode_data[element_type]:
                        satisfied_required += 1
                    else:
                        validation_results['errors'].append(
                            f"Missing required field '{required_field}' in {element_type}"
                        )
                        validation_results['valid'] = False
        
        # Calculate compliance score
        if total_required > 0:
            validation_results['compliance_score'] = satisfied_required / total_required
        
        # Check value sets
        for category, valid_values in self.mcode_value_sets.items():
            if category in mcode_data:
                if mcode_data[category] not in valid_values:
                    validation_results['warnings'].append(
                        f"Value '{mcode_data[category]}' for '{category}' not in standard value set"
                    )
        
        # Check code compliance
        if 'codes' in mcode_data:
            for code_info in mcode_data['codes']:
                if not self._validate_code_compliance(code_info):
                    validation_results['errors'].append(
                        f"Code {code_info.get('code')} not compliant with mCODE standards"
                    )
                    validation_results['valid'] = False
        
        return validation_results
    
    def _validate_code_compliance(self, code_info: Dict[str, Any]) -> bool:
        """
        Validate a single code for mCODE compliance
        
        Args:
            code_info: Dictionary containing code information
            
        Returns:
            True if code is compliant, False otherwise
        """
        code = code_info.get('code')
        system = code_info.get('system')
        
        # Check if code is in mCODE required codes
        if system in self.mcode_required_codes:
            return code in self.mcode_required_codes[system]
        
        # For non-required codes, default to compliant
        return True
    
    def map_entities_to_mcode(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map extracted entities to mCODE elements
        
        Args:
            entities: List of extracted entities
            
        Returns:
            List of mapped mCODE elements
        """
        mapped_elements = []
        
        for entity in entities:
            concept = entity.get('text', '')
            confidence = entity.get('confidence', 0.5)
            
            # Try to map concept to mCODE element
            mcode_mapping = self.map_concept_to_mcode(concept, confidence)
            if mcode_mapping:
                mapped_elements.append(mcode_mapping)
            else:
                # Try to map codes if present in entity
                if 'codes' in entity:
                    for system, code in entity['codes'].items():
                        code_mapping = self.map_code_to_mcode(code, system)
                        if code_mapping:
                            mapped_elements.append(code_mapping)
        
        return mapped_elements
    
    def generate_mcode_structure(self, mapped_elements: List[Dict[str, Any]], 
                                 demographics: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate structured mCODE representation from mapped elements
        
        Args:
            mapped_elements: List of mapped mCODE elements
            demographics: Optional demographic information
            
        Returns:
            Dictionary with structured mCODE data
        """
        mcode_structure = {
            'resourceType': 'Bundle',
            'type': 'collection',
            'entry': []
        }
        
        # Add patient resource if demographics provided
        if demographics:
            patient_resource = self._create_patient_resource(demographics)
            mcode_structure['entry'].append({
                'resource': patient_resource
            })
        
        # Create resources for each mapped element
        for element in mapped_elements:
            resource = self._create_mcode_resource(element)
            if resource:
                mcode_structure['entry'].append({
                    'resource': resource
                })
        
        return mcode_structure
    
    def _create_patient_resource(self, demographics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a Patient resource from demographic information
        
        Args:
            demographics: Demographic information
            
        Returns:
            Patient resource dictionary
        """
        patient = {
            'resourceType': 'Patient',
            'gender': demographics.get('gender', 'unknown')
        }
        
        # Add age if provided
        if 'age' in demographics:
            patient['birthDate'] = f"19{90 - int(demographics['age']) if demographics['age'].isdigit() else 50}-01-01"
        
        # Add ethnicity extension if provided
        if 'ethnicity' in demographics:
            patient['extension'] = [{
                'url': 'http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-ethnicity',
                'valueCodeableConcept': {
                    'coding': [{
                        'system': 'http://hl7.org/fhir/us/mcode/CodeSystem/mcode-ethnicity',
                        'code': demographics['ethnicity'],
                        'display': demographics['ethnicity'].replace('-', ' ').title()
                    }]
                }
            }]
        
        return patient
    
    def _create_mcode_resource(self, element: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create an mCODE resource from a mapped element
        
        Args:
            element: Mapped element information
            
        Returns:
            mCODE resource dictionary or None if invalid
        """
        element_type = element.get('mcode_element')
        if not element_type:
            return None
        
        resource = {
            'resourceType': element_type
        }
        
        # Add code information based on element type
        if element_type == 'Condition':
            primary_code = element.get('primary_code', {})
            resource['code'] = {
                'coding': [{
                    'system': self._get_system_uri(primary_code.get('system', '')),
                    'code': primary_code.get('code', ''),
                    'display': self._get_code_display(primary_code.get('code', ''))
                }]
            }
            
            # Add body site for conditions
            resource['bodySite'] = {
                'coding': [{
                    'system': 'http://snomed.info/sct',
                    'code': element.get('mapped_codes', {}).get('SNOMEDCT', ''),
                    'display': self._get_body_site_display(element.get('mapped_codes', {}).get('SNOMEDCT', ''))
                }]
            }
        
        elif element_type == 'Procedure':
            primary_code = element.get('primary_code', {})
            resource['code'] = {
                'coding': [{
                    'system': self._get_system_uri(primary_code.get('system', '')),
                    'code': primary_code.get('code', ''),
                    'display': self._get_code_display(primary_code.get('code', ''))
                }]
            }
        
        elif element_type == 'MedicationStatement':
            primary_code = element.get('primary_code', {})
            resource['medicationCodeableConcept'] = {
                'coding': [{
                    'system': self._get_system_uri(primary_code.get('system', '')),
                    'code': primary_code.get('code', ''),
                    'display': self._get_code_display(primary_code.get('code', ''))
                }]
            }
        
        return resource
    
    def _get_system_uri(self, system: str) -> str:
        """
        Get the URI for a coding system
        
        Args:
            system: Coding system name
            
        Returns:
            URI for the coding system
        """
        system_uris = {
            'ICD10CM': 'http://hl7.org/fhir/sid/icd-10-cm',
            'CPT': 'http://www.ama-assn.org/go/cpt',
            'LOINC': 'http://loinc.org',
            'RxNorm': 'http://www.nlm.nih.gov/research/umls/rxnorm',
            'SNOMEDCT': 'http://snomed.info/sct'
        }
        
        return system_uris.get(system, f"http://example.org/{system.lower()}")
    
    def _get_code_display(self, code: str) -> str:
        """
        Get display text for a code
        
        Args:
            code: Code to get display text for
            
        Returns:
            Display text for the code
        """
        # This would typically be looked up in a terminology server
        code_displays = {
            'C50.911': 'Malignant neoplasm of breast',
            'C34.90': 'Malignant neoplasm of lung',
            'C18.9': 'Malignant neoplasm of colon',
            '12345': 'Chemotherapy procedure',
            '67890': 'Radiation therapy procedure',
            '123456': 'Paclitaxel',
            '789012': 'Doxorubicin'
        }
        
        return code_displays.get(code, code)
    
    def _get_body_site_display(self, snomed_code: str) -> str:
        """
        Get display text for a body site SNOMED code
        
        Args:
            snomed_code: SNOMED CT code for body site
            
        Returns:
            Display text for the body site
        """
        body_site_displays = {
            '254837009': 'Breast',
            '254838004': 'Lung',
            '363346000': 'Colon',
            '789012009': 'Liver',
            '363417006': 'Pancreas'
        }
        
        return body_site_displays.get(snomed_code, 'Body site')
    
    def process_nlp_output(self, nlp_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process NLP engine output and generate mCODE mappings
        
        Args:
            nlp_output: Output from the NLP engine
            
        Returns:
            Dictionary with mCODE mappings and validation results
        """
        # Extract entities and codes from NLP output
        entities = nlp_output.get('entities', [])
        codes = nlp_output.get('codes', {}).get('extracted_codes', {})
        
        # Map entities to mCODE elements
        mapped_entities = self.map_entities_to_mcode(entities)
        
        # Map codes to mCODE elements
        mapped_codes = []
        for system, system_codes in codes.items():
            for code_info in system_codes:
                code_mapping = self.map_code_to_mcode(code_info['code'], system)
                if code_mapping:
                    mapped_codes.append(code_mapping)
        
        # Combine all mapped elements
        all_mapped_elements = mapped_entities + mapped_codes
        
        # Generate structured mCODE representation
        demographics = nlp_output.get('demographics', {})
        mcode_structure = self.generate_mcode_structure(all_mapped_elements, demographics)
        
        # Validate mCODE compliance
        validation_results = self.validate_mcode_compliance({
            'entities': mapped_entities,
            'codes': mapped_codes,
            'demographics': demographics
        })
        
        # Create result structure
        result = {
            'mapped_elements': all_mapped_elements,
            'mcode_structure': mcode_structure,
            'validation': validation_results,
            'metadata': {
                'mapped_entities_count': len(mapped_entities),
                'mapped_codes_count': len(mapped_codes),
                'total_mapped_elements': len(all_mapped_elements)
            }
        }
        
        return result


# Example usage
if __name__ == "__main__":
    # This is just for testing purposes
    mapper = MCODEMappingEngine()
    
    # Sample NLP output
    sample_nlp_output = {
        'entities': [
            {'text': 'breast cancer', 'confidence': 0.9},
            {'text': 'paclitaxel', 'confidence': 0.8}
        ],
        'codes': {
            'extracted_codes': {
                'ICD10CM': [{'code': 'C50.911', 'system': 'ICD-10-CM'}],
                'RxNorm': [{'code': '123456', 'system': 'RxNorm'}]
            }
        },
        'demographics': {
            'gender': 'female',
            'age': '55'
        }
    }
    
    # Process the sample output
    result = mapper.process_nlp_output(sample_nlp_output)
    
    print("mCODE mapping complete. Results:")
    print(f"- Mapped {result['metadata']['mapped_entities_count']} entities")
    print(f"- Mapped {result['metadata']['mapped_codes_count']} codes")
    print(f"- Validation: {'Passed' if result['validation']['valid'] else 'Failed'}")
    if result['validation']['errors']:
        print("Validation errors:")
        for error in result['validation']['errors']:
            print(f"  - {error}")