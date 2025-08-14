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
            },
            'GenomicVariant': {
                'required': ['geneStudied', 'dnaChange'],
                'optional': ['aminoAcidChange', 'variantAlleleFrequency']
            },
            'Biomarker': {
                'required': ['code', 'value'],
                'optional': ['interpretation', 'method']
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
                },
                'biomarkers': {
                    'ER': {'system': 'LOINC', 'code': '16112-5'},
                    'PR': {'system': 'LOINC', 'code': '16113-3'},
                    'HER2': {'system': 'LOINC', 'code': '48676-1'},
                    'BRCA1': {'system': 'HGNC', 'code': '1100'},
                    'BRCA2': {'system': 'HGNC', 'code': '1101'},
                    'PD-L1': {'system': 'LOINC', 'code': '82397-3'},
                    'Ki-67': {'system': 'LOINC', 'code': '85337-4'}
                },
                'genomic_variants': {
                    'PIK3CA': {'system': 'HGNC', 'code': '8985'},
                    'TP53': {'system': 'HGNC', 'code': '11998'},
                    'ESR1': {'system': 'HGNC', 'code': '3467'}
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
            },
            'genomic testing': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'CPT', 'code': '81479'},
                'mapped_codes': {
                    'LOINC': '48018-6'
                }
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
        
        # Handle different data structures
        if 'mapped_elements' in mcode_data:
            # Process mapped elements structure
            mapped_elements = mcode_data['mapped_elements']
            for element in mapped_elements:
                element_type = element.get('mcode_element')
                if element_type and element_type in self.mcode_elements:
                    total_required += len(self.mcode_elements[element_type]['required'])
                    # Check if required fields are present in the element
                    required_fields = self.mcode_elements[element_type]['required']
                    for required_field in required_fields:
                        # For mapped elements, we check if they have the necessary data
                        if element_type == 'Condition':
                            if required_field == 'code' and element.get('primary_code'):
                                satisfied_required += 1
                            elif required_field == 'bodySite' and element.get('mapped_codes', {}).get('SNOMEDCT'):
                                satisfied_required += 1
                        elif element_type in ['Procedure', 'MedicationStatement']:
                            if required_field == 'code' and element.get('primary_code'):
                                satisfied_required += 1
                        else:
                            satisfied_required += 1  # Default to satisfied for other types
        elif 'entities' in mcode_data or 'codes' in mcode_data:
            # Process legacy structure
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
        else:
            # For demographics-only data, we still consider it valid
            if 'demographics' in mcode_data and mcode_data['demographics']:
                validation_results['valid'] = True
                validation_results['compliance_score'] = 1.0
                return validation_results
        
        # Calculate compliance score
        if total_required > 0:
            validation_results['compliance_score'] = satisfied_required / total_required
        elif 'mapped_elements' in mcode_data and len(mcode_data['mapped_elements']) > 0:
            # If we have mapped elements but no required fields, consider it partially valid
            validation_results['compliance_score'] = 0.5
        elif 'demographics' in mcode_data and mcode_data['demographics']:
            # If we only have demographics, consider it valid
            validation_results['compliance_score'] = 1.0
        
        # Check value sets
        demographics = mcode_data.get('demographics', {})
        if isinstance(demographics, dict):
            for category, valid_values in self.mcode_value_sets.items():
                if category in demographics:
                    value = demographics[category]
                    # Handle different data structures for demographics
                    if isinstance(value, list) and value:
                        # Use the first value if it's a list
                        value = value[0] if isinstance(value[0], (str, dict)) else str(value[0])
                    elif isinstance(value, dict) and 'gender' in value:
                        # Extract gender from dict structure
                        value = value['gender']
                    elif not isinstance(value, str):
                        value = str(value)
                    
                    # Check if value is in valid values
                    if isinstance(value, str) and value.lower() not in valid_values:
                        validation_results['warnings'].append(
                            f"Value '{value}' for '{category}' not in standard value set"
                        )
        
        # Check code compliance - only if codes are provided
        codes_list = []
        if 'codes' in mcode_data:
            codes_data = mcode_data['codes']
            if isinstance(codes_data, dict) and 'extracted_codes' in codes_data:
                # Handle extracted_codes structure
                for system_codes in codes_data['extracted_codes'].values():
                    codes_list.extend(system_codes)
            elif isinstance(codes_data, list):
                # Handle list of codes
                codes_list = codes_data
        
        for code_info in codes_list:
            if isinstance(code_info, dict) and not self._validate_code_compliance(code_info):
                code_value = code_info.get('code', 'unknown')
                validation_results['errors'].append(
                    f"Code {code_value} not compliant with mCODE standards"
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
            'gender': 'unknown'  # Default value
        }
        
        # Extract gender from gender criteria
        if 'gender' in demographics:
            gender_criteria = demographics['gender']
            if isinstance(gender_criteria, list) and gender_criteria:
                # Extract gender from the first gender criterion
                first_criterion = gender_criteria[0]
                if isinstance(first_criterion, dict) and 'gender' in first_criterion:
                    gender = first_criterion['gender']
                    if gender in ['male', 'female']:
                        patient['gender'] = gender
                # If we have text, try to extract gender from it
                elif isinstance(first_criterion, dict) and 'text' in first_criterion:
                    text = first_criterion['text'].lower()
                    if 'male' in text or 'men' in text:
                        patient['gender'] = 'male'
                    elif 'female' in text or 'women' in text:
                        patient['gender'] = 'female'
            elif isinstance(gender_criteria, str):
                # Direct gender string
                patient['gender'] = gender_criteria.lower()
        
        # Add age if provided
        if 'age' in demographics:
            age = demographics['age']
            if isinstance(age, str) and age.isdigit():
                patient['birthDate'] = f"19{90 - int(age)}-01-01"
            elif isinstance(age, list) and age:
                # Handle age range - use the minimum age
                if isinstance(age[0], dict) and "min_age" in age[0]:
                    min_age = age[0]["min_age"]
                    if isinstance(min_age, str) and min_age.isdigit():
                        patient['birthDate'] = f"19{90 - int(min_age)}-01-01"
                    else:
                        patient['birthDate'] = "1975-01-01"  # Default
                else:
                    patient['birthDate'] = "1975-01-01"  # Default
            else:
                patient['birthDate'] = "1975-01-01"  # Default
        
        # Add ethnicity extension if provided
        # Note: Ethnicity extraction is not implemented in the current NLP engine
        # This would need to be added in a future version
        
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
            nlp_output: Output from the NLP engine or LLM interface
            
        Returns:
            Dictionary with mCODE mappings and validation results
        """
        # Extract entities, codes and genomic features from input
        entities = nlp_output.get('entities', [])
        codes = nlp_output.get('codes', {}).get('extracted_codes', {})
        
        # Handle LLM genomic features if present
        if 'genomic_features' in nlp_output:
            for feature in nlp_output['genomic_features']:
                entities.append({
                    'text': f"{feature['gene']} {feature['variant']}",
                    'confidence': 0.9,
                    'type': 'genomic_variant'
                })
        
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
            'mapped_elements': all_mapped_elements,
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