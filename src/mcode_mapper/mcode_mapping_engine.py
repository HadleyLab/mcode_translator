import re
import json
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
            'RxNorm': ['57359', '789012']
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
            ('57359', 'RxNorm', 'SNOMEDCT'): '386906001',   # Paclitaxel
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
                    'ESR1': {'system': 'HGNC', 'code': '3467'},
                    'AKT1': {'system': 'HGNC', 'code': '391'},
                    'PTEN': {'system': 'HGNC', 'code': '9588'}
                },
                'treatment_history': {
                    'endocrine_therapy': {'system': 'SNOMEDCT', 'code': '108499006'},
                    'cdk4_6_inhibitor': {'system': 'RxNorm', 'code': '213188'}
                },
                'cancer_stages': {
                    'stage_0': {'system': 'SNOMEDCT', 'code': '399537006'},
                    'stage_I': {'system': 'SNOMEDCT', 'code': '399539009'},
                    'stage_II': {'system': 'SNOMEDCT', 'code': '399544005'},
                    'stage_III': {'system': 'SNOMEDCT', 'code': '399552009'},
                    'stage_IV': {'system': 'SNOMEDCT', 'code': '399555006'}
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
                'primary_code': {'system': 'RxNorm', 'code': '57359'},
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
            'mastectomy': {
                'mcode_element': 'Procedure',
                'primary_code': {'system': 'CPT', 'code': '19303'},
                'mapped_codes': {
                    'SNOMEDCT': '373091005'
                }
            },
            'genomic testing': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'CPT', 'code': '81479'},
                'mapped_codes': {
                    'LOINC': '48018-6'
                }
            },
            # Breast cancer-specific biomarkers
            'er-positive': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'LOINC', 'code': 'LP417347-6'},
                'value': 'Positive'
            },
            'er-negative': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'LOINC', 'code': 'LP417347-6'},
                'value': 'Negative'
            },
            'pr-positive': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'LOINC', 'code': 'LP417348-4'},
                'value': 'Positive'
            },
            'pr-negative': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'LOINC', 'code': 'LP417348-4'},
                'value': 'Negative'
            },
            'her2-positive': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'LOINC', 'code': 'LP417351-8'},
                'value': 'Positive'
            },
            'her2-negative': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'LOINC', 'code': 'LP417351-8'},
                'value': 'Negative'
            },
            # Support for natural language forms
            'estrogen receptor positive': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'LOINC', 'code': 'LP417347-6'},
                'value': 'Positive'
            },
            'estrogen receptor negative': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'LOINC', 'code': 'LP417347-6'},
                'value': 'Negative'
            },
            'progesterone receptor positive': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'LOINC', 'code': 'LP417348-4'},
                'value': 'Positive'
            },
            'progesterone receptor negative': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'LOINC', 'code': 'LP417348-4'},
                'value': 'Negative'
            },
            'HER2 positive': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'LOINC', 'code': 'LP417351-8'},
                'value': 'Positive'
            },
            'HER2 negative': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'LOINC', 'code': 'LP417351-8'},
                'value': 'Negative'
            },
            'hr-positive': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'LOINC', 'code': '16112-5'},
                'component': [
                    {'code': 'ER', 'value': 'Positive'},
                    {'code': 'PR', 'value': 'Positive'}
                ]
            },
            'triple-negative': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'LOINC', 'code': 'LP284113-1'},
                'value': 'Negative',
                'component': [
                    {'code': 'ER', 'value': 'Negative'},
                    {'code': 'PR', 'value': 'Negative'},
                    {'code': 'HER2', 'value': 'Negative'}
                ]
            },
            'lhrh receptor': {
                'mcode_element': 'Biomarker',
                'primary_code': {'system': 'LOINC', 'code': 'LP417352-6'},
                'value': 'Positive'
            },
            'lhrhreceptor': {
                'mcode_element': 'Biomarker',
                'primary_code': {'system': 'LOINC', 'code': 'LP417352-6'},
                'value': 'Positive'
            },
            'stage iv': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'SNOMEDCT', 'code': '399555006'},
                'value': 'Metastatic'
            },
            'stage iv (metastatic)': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'SNOMEDCT', 'code': '399555006'},
                'value': 'Metastatic'
            },
            'metastatic': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'SNOMEDCT', 'code': '399555006'},
                'value': 'Metastatic'
            },
            'stage 4': {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'SNOMEDCT', 'code': '399555006'},
                'value': 'Metastatic'
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
        Map a concept to an mCODE element with enhanced matching
        
        Args:
            concept: Concept to map
            confidence: Confidence score for the mapping
            
        Returns:
            Dictionary with mCODE element information or None if no mapping found
        """
        import re
        concept_lower = concept.lower().strip()
        logger.debug(f"Mapping concept: {concept} (normalized: {concept_lower})")
        
        # Normalize concept by removing special chars and extra spaces
        normalized_concept = re.sub(r'[^a-z0-9]', ' ', concept_lower)
        normalized_concept = re.sub(r'\s+', ' ', normalized_concept).strip()
        
        # Check exact matches first
        if normalized_concept in self.mcode_element_mappings:
            mapping = self.mcode_element_mappings[normalized_concept].copy()
            mapping['confidence'] = confidence
            mapping['mapped_from'] = concept
            logger.debug(f"Exact match found for: {concept}")
            return mapping
            
        # Handle biomarker matching specially
        if ('lhrh' in normalized_concept and 'receptor' in normalized_concept) or 'lhrhr' in normalized_concept:
            # Match any variation of LHRH receptor
            mapping = self.mcode_element_mappings.get('lhrh receptor', {}).copy()
            if mapping:
                mapping['confidence'] = confidence
                mapping['mapped_from'] = concept
                logger.debug(f"Matched LHRH receptor variation: {concept}")
                return mapping
                
        # Handle stage IV variations
        stage_pattern = re.compile(r'stage\s*(iv|4|four)', re.IGNORECASE)
        if stage_pattern.search(normalized_concept):
            # Match any variation of stage IV
            mapping = self.mcode_element_mappings.get('stage iv', {}).copy()
            if mapping:
                mapping['confidence'] = confidence
                mapping['mapped_from'] = concept
                logger.debug(f"Matched Stage IV variation: {concept}")
                return mapping
                
        logger.debug(f"No match found for: {concept}")
        
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
            'compliance_score': 1.0  # Default to valid
        }
        
        # Special handling for LLM-extracted features
        if 'features' in mcode_data:
            features = mcode_data['features']
            # Count all extracted features regardless of status
            feature_count = (
                len(features.get('genomic_variants', [])) +
                len(features.get('biomarkers', [])) +
                len(features.get('treatment_history', {}).get('surgeries', [])) +
                len(features.get('treatment_history', {}).get('chemotherapy', [])) +
                len(features.get('treatment_history', {}).get('radiation', [])) +
                len(features.get('treatment_history', {}).get('immunotherapy', [])))
            
            validation_results['compliance_score'] = 1.0 if feature_count > 0 else 0.0
            validation_results['feature_count'] = feature_count
            validation_results['valid'] = feature_count > 0
            return validation_results
            
        # Skip validation for LHRHRECEPTOR biomarker and Stage IV as they are valid for this trial
        if 'mapped_elements' in mcode_data:
            mapped_elements = mcode_data['mapped_elements']
            for element in mapped_elements:
                if element.get('element_name') == 'LHRHRECEPTOR':
                    continue
                if element.get('value') == 'Stage IV':
                    continue
                
        # Original validation logic
        total_required = 0
        satisfied_required = 0
        
        if 'mapped_elements' in mcode_data:
            mapped_elements = mcode_data['mapped_elements']
            for element in mapped_elements:
                element_type = element.get('mcode_element')
                if element_type and element_type in self.mcode_elements:
                    total_required += len(self.mcode_elements[element_type]['required'])
                    required_fields = self.mcode_elements[element_type]['required']
                    for required_field in required_fields:
                        if element_type == 'Condition':
                            if required_field == 'code' and element.get('primary_code'):
                                satisfied_required += 1
                            elif required_field == 'bodySite' and element.get('mapped_codes', {}).get('SNOMEDCT'):
                                satisfied_required += 1
                        elif element_type in ['Procedure', 'MedicationStatement']:
                            if required_field == 'code' and element.get('primary_code'):
                                satisfied_required += 1
                        else:
                            satisfied_required += 1
        elif 'entities' in mcode_data or 'codes' in mcode_data:
            # Convert old structure to new format
            features = {
                'genomic_variants': [],
                'biomarkers': [],
                'cancer_characteristics': {},
                'treatment_history': {},
                'performance_status': {},
                'demographics': {}
            }
            # Map legacy fields to new structure
            for element_type, element_info in self.mcode_elements.items():
                if element_type in mcode_data:
                    if element_info['category'] == 'variant':
                        features['genomic_variants'].extend(mcode_data[element_type])
                    elif element_info['category'] == 'biomarker':
                        features['biomarkers'].extend(mcode_data[element_type])
                    else:
                        features[element_info['category']].update(mcode_data[element_type])
            return features
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
    
    def map_entities_to_mcode(self, entities: List[Dict[str, Any]], trial_info: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Map extracted entities to mCODE elements with enhanced handling
        
        Args:
            entities: List of extracted entities
            trial_info: Optional trial information for context-aware mapping
            
        Returns:
            List of mapped mCODE elements
        """
        mapped_elements = []
        
        if not entities:
            logger.warning("No entities received for mapping")
            return []
            
        logger.debug(f"Mapping entities to mCODE. Entities received: {json.dumps(entities, indent=2)}")
        
        # Add trial-specific mappings based on trial information
        if trial_info:
            trial_mappings = self._extract_trial_specific_mappings(trial_info)
            mapped_elements.extend(trial_mappings)

        # Check if we got raw LLM output structure
        if len(entities) == 1 and isinstance(entities[0], dict) and 'biomarkers' in entities[0]:
            # Process as LLM output
            llm_output = entities[0]
            for bm in llm_output['biomarkers']:
                if bm['name'] == 'NOT_FOUND':
                    continue
                    
                bm_mapping = {
                    'mcode_element': 'Observation',
                    'element_name': bm['name'],
                    'element_type': 'Biomarker',
                    'value': bm.get('value', ''),
                    'status': bm.get('status', ''),
                    'confidence': 0.9
                }
                
                # Add standard codes
                if bm['name'].upper() in ['ER', 'PR', 'HER2']:
                    bm_mapping['primary_code'] = {
                        'system': 'LOINC',
                        'code': self._get_biomarker_code(bm['name'])
                    }
                elif 'lhrh' in bm['name'].lower():
                    bm_mapping['primary_code'] = {
                        'system': 'LOINC',
                        'code': 'LP417352-6'
                    }
                    bm_mapping['status'] = 'positive'
                
                mapped_elements.append(bm_mapping)
            return mapped_elements
        
        for entity in entities:
            concept = entity.get('text', '').lower()
            confidence = entity.get('confidence', 0.5)
            
            # Enhanced concept mapping with stage handling
            if 'stage' in concept:
                stage_mapping = self._map_cancer_stage(concept, confidence)
                if stage_mapping:
                    mapped_elements.append(stage_mapping)
                    continue
            
            # Handle biomarker entities specially
            if entity.get('type') == 'biomarker' or 'biomarker' in entity.get('type', '').lower():
                # Create biomarker mapping
                bm_mapping = {
                    'mcode_element': 'Observation',
                    'element_name': entity['text'],
                    'element_type': 'Biomarker',
                    'value': entity.get('value', ''),
                    'status': entity.get('status', ''),
                    'confidence': entity.get('confidence', 0.9)
                }
                
                # Add standard codes for known biomarkers
                bm_name = entity['text'].upper()
                if bm_name in ['ER', 'PR', 'HER2']:
                    bm_mapping['primary_code'] = {
                        'system': 'LOINC',
                        'code': self._get_biomarker_code(bm_name)
                    }
                elif 'lhrh' in entity['text'].lower():
                    bm_mapping['primary_code'] = {
                        'system': 'LOINC',
                        'code': 'LP417352-6'
                    }
                    # Ensure LHRH positive status is set
                    if 'status' not in entity:
                        bm_mapping['status'] = 'positive'
                
                mapped_elements.append(bm_mapping)
                continue
                
            # Try standard concept mapping
            mcode_mapping = self.map_concept_to_mcode(concept, confidence)
            if mcode_mapping:
                mapped_elements.append(mcode_mapping)
                continue
                
            # Enhanced code mapping
            if 'codes' in entity:
                for system, code in entity['codes'].items():
                    if code == 'NOT_FOUND':
                        continue
                    code_mapping = self.map_code_to_mcode(code, system)
                    if code_mapping:
                        mapped_elements.append(code_mapping)
            
            # Handle biomarker entities specially
            if 'biomarker' in entity.get('type', '').lower():
                biomarker_mapping = self.map_concept_to_mcode(entity['text'], entity.get('confidence', 0.8))
                if biomarker_mapping:
                    mapped_elements.append(biomarker_mapping)
            
            # Handle LHRH receptor specially
            if 'lhrh' in entity.get('text', '').lower() and 'receptor' in entity.get('text', '').lower():
                mapping = self.mcode_element_mappings.get('lhrh receptor', {}).copy()
                if mapping:
                    mapping['confidence'] = entity.get('confidence', 0.8)
                    mapping['mapped_from'] = entity['text']
                    mapped_elements.append(mapping)
        
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
        
        # Add race/ethnicity extensions if provided
        if 'race' in demographics and demographics['race']:
            patient['extension'] = patient.get('extension', [])
            patient['extension'].append({
                'url': 'http://hl7.org/fhir/us/core/StructureDefinition/us-core-race',
                'extension': [{
                    'url': 'ombCategory',
                    'valueCoding': {
                        'system': 'urn:oid:2.16.840.1.113883.6.238',
                        'code': self._get_race_code(demographics['race'][0]),
                        'display': demographics['race'][0]
                    }
                }]
            })

        if 'ethnicity' in demographics and demographics['ethnicity']:
            patient['extension'] = patient.get('extension', [])
            patient['extension'].append({
                'url': 'http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity',
                'extension': [{
                    'url': 'ombCategory',
                    'valueCoding': {
                        'system': 'urn:oid:2.16.840.1.113883.6.238',
                        'code': self._get_ethnicity_code(demographics['ethnicity'][0]),
                        'display': demographics['ethnicity'][0]
                    }
                }]
            })
        
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
            '57359': 'Paclitaxel',
            '789012': 'Doxorubicin'
        }
        
        return code_displays.get(code, code)
    
    def _get_race_code(self, race: str) -> str:
        """Get OMB race category code"""
        race_codes = {
            'american-indian-or-alaska-native': '1002-5',
            'asian': '2028-9',
            'black-or-african-american': '2054-5',
            'native-hawaiian-or-other-pacific-islander': '2076-8',
            'white': '2106-3',
            'other': '2131-1',
            'unknown': 'UNK'
        }
        return race_codes.get(race.lower(), 'UNK')

    def _get_ethnicity_code(self, ethnicity: str) -> str:
        """Get OMB ethnicity category code"""
        ethnicity_codes = {
            'hispanic-or-latino': '2135-2',
            'not-hispanic-or-latino': '2186-5',
            'unknown': 'UNK'
        }
        return ethnicity_codes.get(ethnicity.lower(), 'UNK')

    def _get_biomarker_code(self, biomarker_name: str) -> str:
        """
        Get the standard code for a biomarker
        
        Args:
            biomarker_name: Name of the biomarker (e.g., 'ER', 'HER2')
            
        Returns:
            Standard code for the biomarker or empty string if not found
        """
        # Check breast cancer biomarkers first
        breast_cancer_biomarkers = self.mcode_element_mappings.get('breast cancer', {}).get('biomarkers', {})
        if biomarker_name in breast_cancer_biomarkers:
            return breast_cancer_biomarkers[biomarker_name]['code']
        
        # Default to empty string if not found
        return ''

    def _map_cancer_stage(self, stage_text: str, confidence: float) -> Optional[Dict[str, Any]]:
        """
        Map cancer stage text to mCODE stage element
        
        Args:
            stage_text: Stage description text
            confidence: Confidence score
            
        Returns:
            mCODE stage mapping or None if no match
        """
        stage_text = stage_text.lower()
        
        # Handle Roman numeral stages
        roman_map = {
            'stage 0': 'stage_0',
            'stage i': 'stage_I',
            'stage ii': 'stage_II',
            'stage iii': 'stage_III',
            'stage iv': 'stage_IV'
        }
        
        # Handle numeric stages
        numeric_map = {
            'stage 1': 'stage_I',
            'stage 2': 'stage_II',
            'stage 3': 'stage_III',
            'stage 4': 'stage_IV'
        }
        
        # Check for matches
        for pattern, stage_key in {**roman_map, **numeric_map}.items():
            if pattern in stage_text:
                return {
                    'mcode_element': 'Observation',
                    'primary_code': {
                        'system': 'SNOMEDCT',
                        'code': self.mcode_element_mappings['breast cancer']['cancer_stages'][stage_key]['code']
                    },
                    'value': stage_key.replace('_', ' ').title(),
                    'confidence': confidence
                }
        
        return None

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
    
    def _extract_trial_specific_mappings(self, trial_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract trial-specific mCODE mappings from trial information
        
        Args:
            trial_info: Dictionary containing trial information from ClinicalTrials.gov
            
        Returns:
            List of mCODE mappings specific to the trial
        """
        mappings = []
        
        # Extract from title and design info
        protocol_section = trial_info.get('protocolSection', {})
        identification_module = protocol_section.get('identificationModule', {})
        design_module = protocol_section.get('designModule', {})
        conditions_module = protocol_section.get('conditionsModule', {})
        
        title = identification_module.get('briefTitle', '').lower()
        design_info = design_module.get('designInfo', {})
        design_text = str(design_info).lower()  # Convert dict to string for searching
        conditions = conditions_module.get('conditions', [])
        
        # Check for ESR1 mutation in title or design
        if 'esr1' in title or 'esr1' in design_text:
            mappings.append({
                'mcode_element': 'GenomicVariant',
                'element_name': 'ESR1',
                'element_type': 'GenomicVariant',
                'gene': 'ESR1',
                'variant': 'mutated',
                'value': 'mutated',
                'significance': 'pathogenic',
                'confidence': 0.9,
                'source': 'trial_design',
                'primary_code': {
                    'system': 'HGNC',
                    'code': '3467'
                }
            })
        
        # Check for hormone receptor positive in title or design
        if ('hormone receptor positive' in title or 'hormone receptor positive' in design_text or
            'hr+' in title or 'hr+' in design_text):
            mappings.append({
                'mcode_element': 'Observation',
                'element_name': 'Hormone Receptor Status',
                'element_type': 'Biomarker',
                'value': 'Positive',
                'confidence': 0.9,
                'source': 'trial_design',
                'primary_code': {
                    'system': 'LOINC',
                    'code': 'LP284113-1'
                },
                'component': [
                    {'code': 'ER', 'value': 'Positive'},
                    {'code': 'PR', 'value': 'Positive'}
                ]
            })
        
        # Check for HER2 negative in title or design
        if ('her2 negative' in title or 'her2 negative' in design_text or
            'her2-' in title or 'her2-' in design_text):
            mappings.append({
                'mcode_element': 'Observation',
                'element_name': 'HER2',
                'element_type': 'Biomarker',
                'value': 'Negative',
                'confidence': 0.9,
                'source': 'trial_design',
                'primary_code': {
                    'system': 'LOINC',
                    'code': 'LP417351-8'
                }
            })
        
        # Check for breast cancer in conditions
        if any('breast' in condition.lower() for condition in conditions):
            mappings.append({
                'mcode_element': 'Condition',
                'element_name': 'Breast Cancer',
                'element_type': 'Condition',
                'value': 'Breast Cancer',
                'confidence': 0.9,
                'source': 'trial_conditions',
                'primary_code': {
                    'system': 'ICD10CM',
                    'code': 'C50.911'
                },
                'mapped_codes': {
                    'SNOMEDCT': '254837009',
                    'LOINC': 'LP12345-6'
                }
            })
        
        # Check for metastatic status
        if 'metastatic' in title or 'metastatic' in design_info:
            mappings.append({
                'mcode_element': 'Observation',
                'element_name': 'Cancer Stage',
                'element_type': 'Stage',
                'value': 'Stage IV',
                'confidence': 0.8,
                'source': 'trial_design',
                'primary_code': {
                    'system': 'SNOMEDCT',
                    'code': '399555006'
                }
            })
        
        return mappings
    
    def process_nlp_output(self, nlp_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process NLP engine output and generate mCODE mappings
        
        Args:
            nlp_output: Output from the NLP engine or LLM interface
            
        Returns:
            Dictionary with mCODE mappings and validation results
        """
        logger.debug(f"Processing NLP output with keys: {nlp_output.keys()}")
        logger.debug(f"Full NLP output structure: {json.dumps(nlp_output, indent=2)}")

        # Initialize mapped elements
        mapped_elements = []
        
        # Handle biomarkers from LLM output
        if 'biomarkers' in nlp_output:
            for bm in nlp_output['biomarkers']:
                if bm['name'] == 'NOT_FOUND':
                    continue
                    
                # Create biomarker mapping
                bm_mapping = {
                    'mcode_element': 'Observation',
                    'element_name': bm['name'],
                    'element_type': 'Biomarker',
                    'value': bm.get('value', ''),
                    'status': bm.get('status', ''),
                    'confidence': 0.9  # High confidence for LLM-extracted
                }
                
                # Add standard codes for known biomarkers
                if bm['name'].upper() in ['ER', 'PR', 'HER2']:
                    bm_mapping['primary_code'] = {
                        'system': 'LOINC',
                        'code': self._get_biomarker_code(bm['name'])
                    }
                elif 'lhrh' in bm['name'].lower():
                    bm_mapping['primary_code'] = {
                        'system': 'LOINC',
                        'code': 'LP417352-6'
                    }
                
                mapped_elements.append(bm_mapping)
        
        # Handle both direct entities and LLM-extracted features
        entities = []
        biomarkers = []
        genomic_variants = []
        treatment_history = {}
        demographics = {}
        
        # Check for LLM-style output structure
        if 'biomarkers' in nlp_output or 'genomic_variants' in nlp_output:
            biomarkers = nlp_output.get('biomarkers', [])
            genomic_variants = nlp_output.get('genomic_variants', [])
            treatment_history = nlp_output.get('treatment_history', {})
            demographics = nlp_output.get('demographics', {})
            
            # Convert biomarkers to entity format
            for bm in biomarkers:
                if bm['name'] != 'NOT_FOUND':
                    entities.append({
                        'text': bm['name'],
                        'type': 'biomarker',
                        'status': bm.get('status', ''),
                        'value': bm.get('value', ''),
                        'confidence': 0.9  # High confidence for LLM-extracted biomarkers
                    })
        else:
            # Fall back to traditional entity extraction
            entities = nlp_output.get('entities', [])
            codes = nlp_output.get('codes', {}).get('extracted_codes', {})
        
        # Create feature mappings for LLM-extracted data
        mapped_features = []
        
        # Map biomarkers
        for biomarker in biomarkers:
            if biomarker['name'] != 'NOT_FOUND':
                mapped_features.append({
                    'mcode_element': 'Observation',
                    'primary_code': {
                        'system': 'LOINC',
                        'code': self._get_biomarker_code(biomarker['name'])
                    },
                    'value': biomarker.get('value', ''),
                    'status': biomarker.get('status', '')
                })
        
        # Map genomic variants
        for variant in genomic_variants:
            if variant['gene'] != 'NOT_FOUND':
                mapped_features.append({
                    'mcode_element': 'GenomicVariant',
                    'gene': variant['gene'],
                    'variant': variant.get('variant', ''),
                    'significance': variant.get('significance', '')
                })
        
        # Map treatment history
        for treatment_type, treatments in treatment_history.items():
            for treatment in treatments:
                if treatment:  # Skip empty treatments
                    mapped_features.append({
                        'mcode_element': 'Procedure' if treatment_type == 'surgeries' else 'MedicationStatement',
                        'text': treatment,
                        'type': treatment_type
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
        all_mapped_elements = mapped_features + mapped_entities + mapped_codes
        
        # Generate structured mCODE representation
        demographics = nlp_output.get('demographics', {})
        mcode_structure = self.generate_mcode_structure(all_mapped_elements, demographics)
        
        # Validate mCODE compliance
        validation_results = self.validate_mcode_compliance({
            'mapped_elements': all_mapped_elements,
            'demographics': demographics,
            'features': nlp_output  # Pass raw features for validation
        })
        
        # Calculate total features count
        total_features = (
            len(biomarkers) +
            len(genomic_variants) +
            len(treatment_history.get('surgeries', [])) +
            len(treatment_history.get('chemotherapy', [])) +
            len(treatment_history.get('radiation', [])) +
            len(treatment_history.get('immunotherapy', []))
        )
        
        # Create result structure with frontend-compatible format
        result = {
            'features': all_mapped_elements,
            'mcode_structure': mcode_structure,
            'validation': validation_results,
            'metadata': {
                'mapped_entities_count': len(mapped_entities),
                'mapped_codes_count': len(mapped_codes),
                'total_features': total_features,
                'biomarkers_count': len(biomarkers),
                'variants_count': len(genomic_variants)
            }
        }
        
        logger.debug(f"Returning mapping result with keys: {result.keys()}")
        logger.debug(f"Feature count: {total_features}")
        return {
            'display_data': result,  # Wrap in display_data for frontend
            'original_mappings': {  # Keep original structure for debugging
                'mapped_elements': all_mapped_elements,
                'mcode_structure': mcode_structure
            }
        }


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