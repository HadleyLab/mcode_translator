import sys
import os

# Add src directory to Python path - this needs to be done before other imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import asyncio
import logging
import traceback
import colorlog
import time
import hashlib
from typing import Dict, List, Optional, Union, Callable, Set
from pytrials.client import ClinicalTrials
from src.utils.config import Config
from src.utils.cache import CacheManager
from src.data_fetcher.fetcher import search_trials as fetch_search_trials

# Configure colored logging FIRST
logger = colorlog.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers = []

handler = colorlog.StreamHandler(sys.stdout)
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s:%(lineno)d%(reset)s %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))
logger.addHandler(handler)

# Now import NiceGUI after logging is configured
from nicegui import ui, run

from src.pipeline.extraction_pipeline import ExtractionPipeline
from src.nlp_engine.regex_nlp_engine import RegexNLPEngine
from src.nlp_engine.spacy_nlp_engine import SpacyNLPEngine
from src.nlp_engine.llm_nlp_engine import LLMNLPEngine

# Import mCODE modules from fetcher_demo.py
from src.code_extraction.code_extraction import CodeExtractionModule
from src.mcode_mapper.mcode_mapping_engine import MCODEMappingEngine
from src.utils.feature_utils import standardize_features

class ClinicalTrialsAPIError(Exception):
    """Base exception for ClinicalTrialsAPI errors"""
    pass

def search_trials(search_expr: str, fields=None, max_results: int = 100, page_token: str = None, use_cache: bool = True):
    """
    Search for clinical trials matching the expression with pagination support
    
    Args:
        search_expr: Search expression (e.g., "breast cancer")
        fields: List of fields to retrieve (default: None for all fields)
        max_results: Maximum number of results to return (default: 100)
        page_token: Page token for pagination (default: None)
        
    Returns:
        Dictionary containing search results with pagination metadata
        
    Raises:
        ClinicalTrialsAPIError: If there's an error with the API request
    """
    try:
        # Use the refactored fetcher implementation
        result = fetch_search_trials(
            search_expr=search_expr,
            fields=fields,
            max_results=max_results,
            page_token=page_token,
            use_cache=use_cache
        )
        
        # Ensure consistent return format
        if 'studies' not in result:
            result['studies'] = []
            
        # Add pagination metadata if not present
        if 'pagination' not in result:
            result['pagination'] = {
                'max_results': max_results,
                'page_token': page_token
            }
            
        return result
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise ClinicalTrialsAPIError(f"API request failed: {str(e)}")

def get_full_study(nct_id: str):
    """
    Get complete study record for a specific trial
    
    Args:
        nct_id: NCT ID of the clinical trial (e.g., "NCT00000000")
        
    Returns:
        Dictionary containing the full study record
        
    Raises:
        ClinicalTrialsAPIError: If there's an error with the API request
    """
    # Initialize config and cache manager
    config = Config()
    cache_manager = CacheManager(config)
    
    # Create cache key
    cache_key = f"full_study:{nct_id}"
    
    # Try to get from cache first
    cached_result = cache_manager.get(cache_key)
    if cached_result:
        return cached_result
    
    try:
        # Rate limiting
        time.sleep(config.rate_limit_delay)
        
        # Initialize pytrials client
        ct = ClinicalTrials()
        
        # Use get_full_studies with the NCT ID as search expression
        result = ct.get_full_studies(
            search_expr=nct_id,
            max_studies=1,
            fmt="json"
        )
        
        # Extract the study from the response
        if result and 'studies' in result and len(result['studies']) > 0:
            study = result['studies'][0]
            # Cache the result
            cache_manager.set(cache_key, study)
            return study
        else:
            raise ClinicalTrialsAPIError(f"No study found for NCT ID {nct_id}")
            
    except Exception as e:
        raise ClinicalTrialsAPIError(f"API request failed: {str(e)}")

# Initialize NLP engines with caching
extraction_cache = {}
engines = {
    'Regex': RegexNLPEngine(),
    'SpaCy': SpacyNLPEngine(),
    'LLM': LLMNLPEngine()
}

# Initialize mCODE modules
code_extractor = CodeExtractionModule()
mcode_mapper = MCODEMappingEngine()
nlp_engine = LLMNLPEngine()

# Global task tracking for cleanup
active_tasks: Set[asyncio.Task] = set()

# Add mCODE extraction function from fetcher_demo.py
async def extract_mcode_data(study):
    """Extract mCODE data from a study with comprehensive error handling."""
    try:
        # Extract data from study
        protocol_section = study.get('protocolSection', {})
        identification_module = protocol_section.get('identificationModule', {})
        status_module = protocol_section.get('statusModule', {})
        description_module = protocol_section.get('descriptionModule', {})
        conditions_module = protocol_section.get('conditionsModule', {})
        eligibility_module = protocol_section.get('eligibilityModule', {})
        
        # Extract key information
        nct_id = identification_module.get('nctId', 'N/A')
        title = identification_module.get('briefTitle', 'N/A')
        status = status_module.get('overallStatus', 'N/A')
        conditions = conditions_module.get('conditions', [])
        brief_summary = description_module.get('briefSummary', '')
        eligibility_criteria = eligibility_module.get('eligibilityCriteria', '')
        
        # Process eligibility criteria through the full mCODE pipeline
        # Step 1: NLP processing
        # Handle empty or invalid eligibility criteria
        if not eligibility_criteria or not isinstance(eligibility_criteria, str) or len(eligibility_criteria.strip()) == 0:
            logger.warning(f"No eligibility criteria found for study {nct_id}")
            nlp_result = None
        else:
            try:
                # Use run.io_bound to run the synchronous LLM processing asynchronously with timeout
                nlp_result = await asyncio.wait_for(
                    run.io_bound(nlp_engine.process_text, eligibility_criteria),
                    timeout=30.0
                )
                # Check if processing failed
                if nlp_result and hasattr(nlp_result, 'error') and nlp_result.error:
                    logger.warning(f"LLM NLP processing failed for study {nct_id}: {nlp_result.error}")
                    nlp_result = None
            except asyncio.TimeoutError:
                logger.warning(f"Timeout processing criteria with LLM NLP engine for study {nct_id}")
                nlp_result = None
            except Exception as e:
                logger.error(f"Error processing criteria with LLM NLP engine for study {nct_id}: {str(e)}")
                nlp_result = None
        
        # Step 2: Code extraction with timeout
        try:
            extracted_codes = await asyncio.wait_for(
                run.io_bound(
                    code_extractor.process_criteria_for_codes,
                    eligibility_criteria if eligibility_criteria else "",
                    nlp_result.entities if nlp_result and not nlp_result.error else None
                ),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout during code extraction for study {nct_id}")
            extracted_codes = {'extracted_codes': {}}
        except Exception as e:
            logger.error(f"Error during code extraction for study {nct_id}: {str(e)}")
            extracted_codes = {'extracted_codes': {}}
        
        # Step 3: mCODE mapping with timeout
        try:
            # Combine NLP entities and extracted codes for mapping
            all_entities = []
            if nlp_result and not nlp_result.error and hasattr(nlp_result, 'entities'):
                all_entities.extend(nlp_result.entities)
            
            # Add codes as entities for mapping
            if extracted_codes and 'extracted_codes' in extracted_codes:
                for system, codes in extracted_codes['extracted_codes'].items():
                    for code_info in codes:
                        all_entities.append({
                            'text': code_info.get('text', ''),
                            'confidence': code_info.get('confidence', 0.8),
                            'codes': {system: code_info.get('code', '')}
                        })
            
            mapped_mcode = await asyncio.wait_for(
                run.io_bound(mcode_mapper.map_entities_to_mcode, all_entities),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout during mCODE mapping for study {nct_id}")
            mapped_mcode = []
        except Exception as e:
            logger.error(f"Error during mCODE mapping for study {nct_id}: {str(e)}")
            mapped_mcode = []
        
        # Step 4: Generate structured data with timeout
        try:
            demographics = {}
            if nlp_result and not nlp_result.error and hasattr(nlp_result, 'features'):
                demographics = nlp_result.features.get('demographics', {})
            
            structured_data = await asyncio.wait_for(
                run.io_bound(mcode_mapper.generate_mcode_structure, mapped_mcode, demographics),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout generating structured data for study {nct_id}")
            structured_data = {}
        except Exception as e:
            logger.error(f"Error generating structured data for study {nct_id}: {str(e)}")
            structured_data = {}
        
        # Step 5: Validate mCODE compliance with timeout
        try:
            validation_result = await asyncio.wait_for(
                run.io_bound(
                    mcode_mapper.validate_mcode_compliance,
                    {
                        'mapped_elements': mapped_mcode,
                        'demographics': demographics
                    }
                ),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout during mCODE validation for study {nct_id}")
            validation_result = {'is_valid': False, 'error': 'Timeout during validation'}
        except Exception as e:
            logger.error(f"Error during mCODE validation for study {nct_id}: {str(e)}")
            validation_result = {'is_valid': False, 'error': str(e)}
        
        return {
            'nct_id': nct_id,
            'title': title,
            'status': status,
            'conditions': conditions,
            'brief_summary': brief_summary,
            'eligibility_criteria': eligibility_criteria,
            'nlp_result': {
                'entities': nlp_result.entities if nlp_result and hasattr(nlp_result, 'entities') else [],
                'features': nlp_result.features if nlp_result and hasattr(nlp_result, 'features') else {}
            } if nlp_result else {},
            'extracted_codes': extracted_codes,
            'mapped_mcode': mapped_mcode,
            'structured_data': structured_data,
            'validation': validation_result
        }
    except Exception as e:
        logger.error(f"Error extracting mCODE data: {str(e)}")
        return None

# Add mCODE visualization functions from fetcher_demo.py
def display_mcode_visualization(mcode_data):
    """Display mCODE data visualization - simplified version."""
    if not mcode_data:
        ui.label('No mCODE data available for visualization').classes('text-red-500')
        return
    
    with ui.card().classes('w-full p-4 bg-white shadow-md rounded-lg'):
        ui.label('mCODE VISUALIZATION').classes('text-xl font-bold mb-4 text-primary')
        
        # Simple grid with key metrics
        with ui.grid(columns=2).classes('w-full gap-4'):
            # Mapped elements
            with ui.card().classes('p-3 bg-blue-50 rounded-lg'):
                ui.label('MAPPED ELEMENTS').classes('font-bold text-blue-700 text-sm')
                mapped_count = len(mcode_data.get('mapped_mcode', []))
                ui.label(f'{mapped_count}').classes('text-xl font-bold text-blue-600')
            
            # Validation status
            with ui.card().classes('p-3 bg-green-50 rounded-lg'):
                ui.label('VALIDATION').classes('font-bold text-green-700 text-sm')
                validation = mcode_data.get('validation', {})
                if validation.get('is_valid', False):
                    ui.label('VALID').classes('text-xl font-bold text-green-600')
                else:
                    ui.label('INVALID').classes('text-xl font-bold text-red-600')
            
            # NLP entities
            with ui.card().classes('p-3 bg-purple-50 rounded-lg'):
                ui.label('NLP ENTITIES').classes('font-bold text-purple-700 text-sm')
                entities_count = len(mcode_data.get('nlp_result', {}).get('entities', []))
                ui.label(f'{entities_count}').classes('text-xl font-bold text-purple-600')
            
            # Extracted codes
            with ui.card().classes('p-3 bg-orange-50 rounded-lg'):
                ui.label('EXTRACTED CODES').classes('font-bold text-orange-700 text-sm')
                codes = mcode_data.get('extracted_codes', {})
                codes_count = sum(len(codes_list) for codes_list in codes.get('extracted_codes', {}).values())
                ui.label(f'{codes_count}').classes('text-xl font-bold text-orange-600')
        
        # Show mapped elements in a simple list
        mapped_elements = mcode_data.get('mapped_mcode', [])
        if mapped_elements:
            with ui.column().classes('w-full gap-2 mt-4'):
                ui.label('MAPPED mCODE ELEMENTS:').classes('font-bold')
                for element in mapped_elements:
                    with ui.row().classes('items-center gap-2 p-2 bg-gray-50 rounded'):
                        ui.label(element.get('element_name', 'Unknown')).classes('font-medium')
                        ui.label(element.get('element_type', 'Unknown')).classes('text-xs text-gray-600')
                        if element.get('confidence'):
                            ui.label(f'({element.get("confidence"):.2f})').classes('text-xs text-gray-500')

def display_mcode_features_tab(mcode_data):
    """Display mCODE features in a simple structured format."""
    if not mcode_data:
        ui.label('No mCODE data available').classes('text-red-500')
        return
    
    structured_data = mcode_data.get('structured_data', {})
    
    with ui.card().classes('w-full p-4 bg-white shadow-md rounded-lg'):
        ui.label('mCODE STRUCTURED DATA').classes('text-xl font-bold mb-4 text-primary')
        
        # Simple column layout instead of tabs
        with ui.column().classes('w-full gap-4'):
            # Patient data
            patient_data = structured_data.get('patient', {})
            if patient_data:
                with ui.card().classes('w-full p-3 bg-blue-50'):
                    ui.label('PATIENT DATA').classes('font-bold text-blue-700 mb-2')
                    for key, value in patient_data.items():
                        if value and value != 'Not specified':
                            with ui.row().classes('items-center justify-between'):
                                ui.label(key.replace('_', ' ').title()).classes('text-sm')
                                ui.label(str(value)).classes('text-sm text-blue-600 font-medium')
            
            # Cancer condition
            cancer_data = structured_data.get('cancer_condition', {})
            if cancer_data:
                with ui.card().classes('w-full p-3 bg-green-50'):
                    ui.label('CANCER CONDITION').classes('font-bold text-green-700 mb-2')
                    for key, value in cancer_data.items():
                        if value and value != 'Not specified':
                            with ui.row().classes('items-center justify-between'):
                                ui.label(key.replace('_', ' ').title()).classes('text-sm')
                                ui.label(str(value)).classes('text-sm text-green-600 font-medium')
            
            # Genomics
            genomics_data = structured_data.get('genomics', {})
            if genomics_data:
                with ui.card().classes('w-full p-3 bg-purple-50'):
                    ui.label('GENOMICS').classes('font-bold text-purple-700 mb-2')
                    for category, items in genomics_data.items():
                        if items:
                            with ui.column().classes('ml-2 gap-1'):
                                ui.label(category.replace('_', ' ').title()).classes('font-medium text-sm')
                                for item in items:
                                    ui.label(f'• {item}').classes('text-xs text-purple-600')
            
            # Treatment
            treatment_data = structured_data.get('treatment', {})
            if treatment_data:
                with ui.card().classes('w-full p-3 bg-orange-50'):
                    ui.label('TREATMENT').classes('font-bold text-orange-700 mb-2')
                    for key, value in treatment_data.items():
                        if value and value != 'Not specified':
                            with ui.row().classes('items-center justify-between'):
                                ui.label(key.replace('_', ' ').title()).classes('text-sm')
                                ui.label(str(value)).classes('text-sm text-orange-600 font-medium')
            
            # Staging
            staging_data = structured_data.get('staging', {})
            if staging_data:
                with ui.card().classes('w-full p-3 bg-red-50'):
                    ui.label('STAGING').classes('font-bold text-red-700 mb-2')
                    for key, value in staging_data.items():
                        if value and value != 'Not specified':
                            with ui.row().classes('items-center justify-between'):
                                ui.label(key.replace('_', ' ').title()).classes('text-sm')
                                ui.label(str(value)).classes('text-sm text-red-600 font-medium')

# Patient profile section with full mCODE fields
patient_profile = {
    # Demographics
    'cancer_type': 'breast cancer',
    'age': 55,
    'gender': 'female',
    'birth_date': '1970-01-01',
    
    # Cancer Condition
    'stage': 'II',
    'histology': 'Invasive ductal carcinoma',
    'grade': '2',
    'tnm_staging': {
        't': '2',
        'n': '1',
        'm': '0'
    },
    'primary_site': 'Left breast',
    'laterality': 'Left',
    
    # Biomarkers
    'biomarkers': [
        {'name': 'ER', 'status': 'Positive', 'value': '90%'},
        {'name': 'PR', 'status': 'Positive', 'value': '80%'},
        {'name': 'HER2', 'status': 'Negative', 'value': '0'},
        {'name': 'Ki67', 'status': 'Positive', 'value': '20%'}
    ],
    
    # Genomic Variants
    'genomic_variants': [
        {'gene': 'PIK3CA', 'variant': 'H1047R', 'significance': 'Pathogenic'},
        {'gene': 'TP53', 'variant': 'R175H', 'significance': 'Pathogenic'}
    ],
    
    # Treatment History
    'treatments': [
        {
            'type': 'Chemotherapy',
            'name': 'AC-T',
            'start_date': '2023-01-15',
            'end_date': '2023-05-20'
        }
    ],
    
    # Performance Status
    'performance_status': {
        'system': 'ECOG',
        'score': '1',
        'date': '2023-06-01'
    },
    
    # Comorbidities
    'comorbidities': [
        {'condition': 'Hypertension', 'severity': 'Moderate'},
        {'condition': 'Type 2 Diabetes', 'severity': 'Mild'}
    ]
}

# Patient profile UI with expansion sections
patient_profile_expansion = ui.expansion('PATIENT PROFILE', icon='person', value=False).classes('w-full')
with patient_profile_expansion:
    with ui.card().classes('w-full p-4'):
        # Patient Matching Toggle
        with ui.row().classes('items-center w-full mb-4'):
            ui.label('Patient Matching:').classes('font-medium')
            matching_toggle = ui.switch(
                'Patient Matching',
                value=True
            ).props(
                'color=secondary '
                'tooltip="Enable to match trials based on:\\n- Cancer type\\n- Biomarkers\\n- Genomic variants\\n- Stage/grade"'
            ).classes('ml-2')
            
        # Simplified - always keep profile enabled
        matching_toggle.on('change', lambda: ui.notify(f"Patient matching {'enabled' if matching_toggle.value else 'disabled'}"))
        with ui.column().classes('w-full gap-6'):
            # Patient Information Section
            with ui.expansion('PATIENT INFORMATION', icon='person').classes('w-full'):
                with ui.card().classes('w-full p-4'):
                    with ui.grid(columns=1).classes('w-full gap-4').props('sm:columns=2'):
                    # Basic Demographics
                        with ui.column().classes('gap-4'):
                            ui.label('BASIC INFORMATION').classes('text-lg font-bold')
                            ui.label('Age*')
                            age_input = ui.number(
                                value=patient_profile['age'],
                                min=18,
                                max=100,
                                validation={'Please enter age between 18-100': lambda value: 18 <= value <= 100}
                            ).props('''
                                tooltip="Patient age in years (18-100)"
                                required
                            ''')
                            gender_select = ui.select(
                                ['female', 'male', 'other', 'unknown'],
                                value=patient_profile['gender'],
                                label='Sex'
                            ).props('tooltip="Patient\'s biological sex"')
                            ui.label('Date of Birth')
                            birth_date_input = ui.input(
                                value=patient_profile['birth_date']
                            ).props('type=date tooltip="Patient\'s date of birth (YYYY-MM-DD)"')
                    
                    # Cancer Diagnosis
                    with ui.column().classes('gap-4'):
                        ui.label('Cancer Diagnosis').classes('text-lg font-medium')
                        cancer_type_select = ui.select(
                            ['breast cancer', 'lung cancer', 'colorectal cancer', 'prostate cancer'],
                            value=patient_profile['cancer_type'],
                            label='Primary Cancer Type*',
                            validation={'Please select a cancer type': lambda value: value in ['breast cancer', 'lung cancer', 'colorectal cancer', 'prostate cancer']}
                        ).props('tooltip="Select the patient\'s primary cancer diagnosis" required')
        
        # Tumor Characteristics Section
        with ui.expansion('TUMOR CHARACTERISTICS', icon='medical_services', value=False).classes('w-full'):
            with ui.card().classes('w-full p-6 bg-gray-50 shadow-sm rounded-lg gap-4 hover:bg-white transition-colors'):
                with ui.grid(columns=1).classes('w-full gap-4').props('sm:columns=2'):
                    # Pathology Details
                    with ui.column().classes('gap-4'):
                        ui.label('Pathology').classes('text-lg font-medium')
                        histology_input = ui.input(
                            label='Histology*',
                            value=patient_profile['histology'],
                            validation={'Histology is required': lambda value: bool(value.strip())}
                        ).props('tooltip="Cancer histology/morphology type" required')
                        grade_select = ui.select(
                            ['1', '2', '3', '4'],
                            value=patient_profile['grade'],
                            label='Grade*',
                            validation={'Please select a grade': lambda value: value in ['1', '2', '3', '4']}
                        ).props('tooltip="Cancer differentiation grade (1-4)" required')
                        stage_select = ui.select(
                            ['I', 'II', 'III', 'IV'],
                            value=patient_profile['stage'],
                            label='Clinical Stage'
                        ).props('tooltip="Overall cancer stage"')
                    
                    # Tumor Location & Staging
                    with ui.column().classes('gap-4'):
                        ui.label('Location & Staging').classes('text-lg font-medium')
                        primary_site_input = ui.input(
                            label='Primary Site',
                            value=patient_profile['primary_site']
                        ).props('tooltip="Anatomical location of primary tumor"')
                        laterality_select = ui.select(
                            ['Left', 'Right', 'Bilateral', 'Midline'],
                            value=patient_profile['laterality'],
                            label='Laterality'
                        ).props('tooltip="Side of body where tumor is located"')
                        ui.label('TNM Staging').classes('font-medium')
                        with ui.row().classes('items-center gap-4'):
                            t_stage_select = ui.select(['1', '2', '3', '4'], value=patient_profile['tnm_staging']['t'], label='T') \
                                .props('tooltip="Primary tumor size/extent"')
                            n_stage_select = ui.select(['0', '1', '2', '3'], value=patient_profile['tnm_staging']['n'], label='N') \
                                .props('tooltip="Lymph node involvement"')
                            m_stage_select = ui.select(['0', '1'], value=patient_profile['tnm_staging']['m'], label='M') \
                                .props('tooltip="Distant metastasis presence"')
        
        # Biomarkers Section - Refactored to handle all engine formats
        with ui.expansion('BIOMARKERS', icon='science', value=False).classes('w-full'):
            with ui.card().classes('w-full p-6 bg-gray-50 shadow-sm rounded-lg gap-4 hover:bg-white transition-colors'):
                def create_biomarker_row(name, default_status='Unknown', default_value=''):
                    """Create standardized biomarker input row"""
                    with ui.row().classes('items-center gap-2'):
                        ui.label(f'{name}:').classes('font-medium w-16')
                        status_select = ui.select(
                            ['Positive', 'Negative', 'Unknown'],
                            value=default_status,
                        ).classes('flex-grow')
                        value_input = ui.input(value=default_value).classes('w-24')
                    return status_select, value_input

                # Standard biomarkers with fallback to patient profile
                biomarkers_to_show = ['ER', 'PR', 'HER2', 'Ki67']
                biomarker_controls = {}
                
                for biomarker in biomarkers_to_show:
                    # Find in patient profile or use defaults
                    patient_bm = next((b for b in patient_profile['biomarkers'] if b['name'] == biomarker), None)
                    default_status = patient_bm['status'] if patient_bm else 'Unknown'
                    default_value = patient_bm['value'] if patient_bm else ''
                    
                    # Create UI controls and store references
                    status_select, value_input = create_biomarker_row(
                        biomarker, default_status, default_value
                    )
                    biomarker_controls[biomarker] = {
                        'status': status_select,
                        'value': value_input
                    }

                # Dynamic biomarker adder
                with ui.row().classes('items-center gap-2 mt-4'):
                    new_bm_input = ui.input(placeholder='Biomarker name').classes('flex-grow')
                    ui.button('Add', icon='add', on_click=lambda: (
                        biomarker_controls.update({
                            new_bm_input.value: {
                                'status': create_biomarker_row(new_bm_input.value)[0],
                                'value': create_biomarker_row(new_bm_input.value)[1]
                            }
                        }),
                        new_bm_input.set_value('')
                    ))
        
            # Clinical Data Section
        # Genomic Variants Section - Refactored
        with ui.expansion('GENOMIC VARIANTS', icon='dna', value=False).classes('w-full'):
            with ui.card().classes('w-full p-6 bg-gray-50 shadow-sm rounded-lg gap-4 hover:bg-white transition-colors'):
                def create_variant_row(gene='', variant='', significance=''):
                    """Create standardized variant input row"""
                    with ui.row().classes('items-center gap-2'):
                        gene_input = ui.input(value=gene, placeholder='Gene').classes('w-24')
                        variant_input = ui.input(value=variant, placeholder='Variant').classes('w-32')
                        significance_input = ui.input(value=significance, placeholder='Significance').classes('w-32')
                    return gene_input, variant_input, significance_input

                # Existing variants from patient profile
                variant_controls = []
                for v in patient_profile['genomic_variants']:
                    gene, variant, sig = v.get('gene', ''), v.get('variant', ''), v.get('significance', '')
                    variant_controls.append(create_variant_row(gene, variant, sig))

                # Dynamic variant adder
                with ui.row().classes('items-center gap-2 mt-4'):
                    ui.button('Add Variant', icon='add', on_click=lambda: (
                        variant_controls.append(create_variant_row())
                    ))
                
                # Treatment History
                ui.label('Treatment').classes('text-md font-bold mt-4')
                treatment_type_select = ui.select(
                    ['Chemotherapy', 'Radiation', 'Surgery', 'Hormonal', 'Immunotherapy'],
                    value=patient_profile['treatments'][0]['type'] if patient_profile['treatments'] else '',
                    label='Type'
                )
                treatment_name_input = ui.input(
                    label='Regimen/Name',
                    value=patient_profile['treatments'][0]['name'] if patient_profile['treatments'] else ''
                )
                with ui.row():
                    ui.label('Start')
                    treatment_start_input = ui.input(
                        value=patient_profile['treatments'][0]['start_date'] if patient_profile['treatments'] else ''
                    ).props('type=date')
                    ui.label('End')
                    treatment_end_input = ui.input(
                        value=patient_profile['treatments'][0]['end_date'] if patient_profile['treatments'] else ''
                    ).props('type=date')
                
                # Performance Status
                ui.label('Performance Status').classes('text-md font-bold mt-4')
                performance_system_select = ui.select(
                    ['ECOG', 'Karnofsky'],
                    value=patient_profile['performance_status']['system'],
                    label='System'
                )
                performance_score_select = ui.select(
                    ['0', '1', '2', '3', '4'],
                    value=patient_profile['performance_status']['score'],
                    label='Score'
                )
                ui.label('Date')
                performance_date_input = ui.input(
                    value=patient_profile['performance_status']['date']
                ).props('type=date')
                
                # Comorbidities
                ui.label('Comorbidities').classes('text-md font-bold mt-4')
                comorbidity_input = ui.input(
                    label='Condition',
                    value=patient_profile['comorbidities'][0]['condition'] if patient_profile['comorbidities'] else ''
                )
                comorbidity_severity_select = ui.select(
                    ['Mild', 'Moderate', 'Severe'],
                    value=patient_profile['comorbidities'][0]['severity'] if patient_profile['comorbidities'] else '',
                    label='Severity'
                )
            
# Update patient profile function
def validate_form():
    """Validate all form fields before submission"""
    valid = True
    try:
        if not age_input.validate():
            raise ValueError('Please enter a valid age (18-100)')
        if not cancer_type_select.validate():
            raise ValueError('Please select a cancer type')
        if not histology_input.validate():
            raise ValueError('Histology is required')
        if not grade_select.validate():
            raise ValueError('Please select a grade')
    except ValueError as e:
        handle_error(e, "Form validation error")
        return False
    return valid

def update_patient_profile():
    """Update patient profile after validation"""
    if not validate_form():
        return
        
    try:
        # Convert UI inputs to structured format
        new_biomarkers = [
            {'name': 'ER', 'status': biomarker_controls['ER']['status'].value, 'value': biomarker_controls['ER']['value'].value},
            {'name': 'PR', 'status': biomarker_controls['PR']['status'].value, 'value': biomarker_controls['PR']['value'].value},
            {'name': 'HER2', 'status': biomarker_controls['HER2']['status'].value, 'value': biomarker_controls['HER2']['value'].value},
            {'name': 'Ki67', 'status': biomarker_controls['Ki67']['status'].value, 'value': biomarker_controls['Ki67']['value'].value}
        ]
        
        new_variants = []
        for gene_input, variant_input, significance_input in variant_controls:
            new_variants.append({
                'gene': gene_input.value.strip(),
                'variant': variant_input.value.strip(),
                'significance': significance_input.value.strip()
            })

        patient_profile.update({
            # Demographics
            'cancer_type': cancer_type_select.value,
            'age': int(age_input.value),
            'gender': gender_select.value,
            'birth_date': birth_date_input.value,
            
            # Cancer Condition
            'stage': stage_select.value,
            'histology': histology_input.value,
            'grade': grade_select.value,
            'tnm_staging': {
                't': t_stage_select.value,
                'n': n_stage_select.value,
                'm': m_stage_select.value
            },
            'primary_site': primary_site_input.value,
            'laterality': laterality_select.value,
            
            # Biomarkers
            'biomarkers': new_biomarkers,
            
            # Genomic Variants
            'genomic_variants': new_variants,
            
            # Treatment History
            'treatments': [{
                'type': treatment_type_select.value,
                'name': treatment_name_input.value,
                'start_date': treatment_start_input.value,
                'end_date': treatment_end_input.value
            }],
            
            # Performance Status
            'performance_status': {
                'system': performance_system_select.value,
                'score': performance_score_select.value,
                'date': performance_date_input.value
            },
            
            # Comorbidities
            'comorbidities': [{
                'condition': comorbidity_input.value,
                'severity': comorbidity_severity_select.value
            }]
        })
        ui.notify('Patient profile updated!')
    except Exception as e:
        handle_error(e, "Profile update error")

    # Update Button (outside accordion)
    ui.button('UPDATE PROFILE', icon='save', on_click=update_patient_profile) \
        .classes('mt-6 w-full py-3 sm:py-2 bg-primary text-white hover:bg-primary-dark text-base sm:text-sm')

# Basic search interface
with ui.column().classes('w-full items-center'):
    ui.label('CLINICAL TRIALS SEARCH').classes('text-2xl font-bold mb-6')
    
    # Search controls
    with ui.card().classes('w-full p-4'):
        with ui.row().classes('w-full flex-col sm:flex-row items-center gap-4'):
            # Search input
            search_input = ui.input(
                placeholder='Enter cancer type or keywords',
                value=patient_profile['cancer_type']
            ).props('''
                clearable
                tooltip="Search clinical trials by cancer type, biomarkers, or other criteria"
            ''').classes('flex-grow')
            
            # Search actions
            with ui.row().classes('items-center gap-2 mt-4 sm:mt-0'):
                search_button = ui.button('Search', icon='search') \
                    .props('''
                        color=primary
                        tooltip="Find matching clinical trials"
                    ''').classes('min-w-[120px]')
                reset_button = ui.button('Reset', icon='refresh') \
                    .props('''
                        outline
                        tooltip="Clear search results"
                    ''').classes('min-w-[120px]')
        
        # Advanced options in expansion
        with ui.expansion('ADVANCED OPTIONS', icon='tune').classes('w-full mt-4'):
            with ui.grid(columns=1).classes('w-full gap-4').props('sm:columns=2 lg:columns=3'):
                # Results limit
                with ui.card().classes('p-4'):
                    ui.label('Results Limit').classes('text-lg font-bold')
                    limit_slider = ui.slider(min=1, max=20, value=5) \
                        .props('label-always tooltip="Maximum number of clinical trials to return"') \
                        .classes('w-full')
                
                # NLP Engine selection
                with ui.card().classes('p-4'):
                    ui.label('NLP Engine').classes('text-lg font-bold')
                    engine_select = ui.radio(
                        ['Regex', 'SpaCy', 'LLM'],
                        value='LLM'
                    ).props('''
                        inline
                        tooltip="Regex: Fast pattern matching|SpaCy: Balanced accuracy|LLM: Most accurate (OpenAI)"
                    ''')
                
                # Pagination controls
                with ui.card().classes('p-4'):
                    ui.label('Pagination').classes('text-lg font-bold')
                    with ui.row().classes('items-center gap-2'):
                        prev_button = ui.button('Previous', icon='chevron_left') \
                            .props('outline') \
                            .classes('min-w-[100px]')
                        next_button = ui.button('Next', icon='chevron_right') \
                            .props('outline') \
                            .classes('min-w-[100px]')
                        page_status = ui.label('Page 1').classes('text-sm')
                    
                
    # Results display - split panel layout
    with ui.row().classes('w-full h-[calc(100vh-300px)] gap-4'):
        # Left panel (40%) for condensed trial results
        trials_container = ui.column().classes('w-2/5 p-4 gap-4 overflow-y-auto')
        
        # Right panel (60%) for detailed view
        details_panel = ui.column().classes('w-3/5 p-4 overflow-y-auto')

def handle_error(error: Exception, context: str = "", ui_cleanup: Optional[Callable] = None):
    """Centralized error handler with consistent logging and user feedback"""
    error_msg = f"{context}: {str(error)}" if context else str(error)
    
    # Log the error with full traceback
    logger.error(f"{error_msg}\n{traceback.format_exc()}")
    
    # User notification
    ui.notify(f"Error: {error_msg}", type='negative')
    
    # Cleanup UI if needed
    if ui_cleanup:
        ui_cleanup()
    
    return error_msg

# Track currently selected trial card
current_selected_card = None

def create_trial_card(protocol_section):
    """Create a condensed trial card showing only mandatory fields"""
    global current_selected_card
    
    identification_module = protocol_section.get('identificationModule', {})
    status_module = protocol_section.get('statusModule', {})
    conditions_module = protocol_section.get('conditionsModule', {})
    
    # Extract basic information
    title = identification_module.get('briefTitle', 'No title available')
    nct_id = identification_module.get('nctId', 'N/A')
    status = status_module.get('overallStatus', 'Unknown')
    conditions = conditions_module.get('conditions', [])
    
    # Create the condensed trial card with selection state
    card = ui.card().classes('w-full p-3 mb-2 bg-white shadow-sm rounded-lg hover:shadow-md transition-shadow cursor-pointer')
    selected_style = 'border-2 border-blue-500 bg-blue-50'
    with card:
        # Header with title and status
        with ui.row().classes('items-center justify-between w-full'):
            # Truncate title if too long
            short_title = (title[:40] + '...') if len(title) > 40 else title
            ui.label(short_title).classes('text-md font-bold text-gray-800 flex-grow')
            
            # Compact status badge
            status_color = 'green' if status.lower() in ['active', 'recruiting', 'completed'] else 'orange' if status.lower() in ['not yet recruiting', 'suspended'] else 'red'
            ui.badge(status[0].upper()).props(f'color={status_color} size=sm')
        
        # Basic information in compact form
        with ui.column().classes('w-full gap-1 mt-1'):
            ui.label(f"{nct_id}").classes('text-xs text-gray-600 font-mono')
            
            # Conditions (first 2 only)
            if conditions:
                display_conditions = conditions[:2]  # Show up to 2 conditions
                if len(conditions) > 2:
                    display_text = f"{', '.join(display_conditions)} +{len(conditions) - 2}"
                else:
                    display_text = ', '.join(display_conditions)
                
                with ui.tooltip(f"All conditions: {', '.join(conditions)}"):
                    ui.label(display_text).classes('text-xs text-blue-600')
    
    # Add click handler to show details in right panel
    def on_click():
        global current_selected_card
        
        # Clear previous selection
        if current_selected_card:
            current_selected_card.classes(remove=selected_style)
        
        # Set new selection
        card.classes(add=selected_style)
        current_selected_card = card
        
        details_panel.clear()
        with details_panel:
            loading = ui.spinner('dots', size='lg', color='primary')
            status_label = ui.label('Loading trial details...')
        
        async def load_details():
            try:
                full_study = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: get_full_study(nct_id)
                )
                if full_study:
                    mcode_data = await extract_mcode_data(full_study)
                    selected_engine = engine_select.value
                    engine = engines[selected_engine]
                    criteria = protocol_section.get('eligibilityModule', {}).get('eligibilityCriteria', '')
                    display_data = await run.io_bound(engine.process_text, criteria)
                    
                    details_panel.clear()
                    with details_panel:
                        display_trial_results(protocol_section, display_data, mcode_data)
                else:
                    status_label.set_text('Failed to load trial details')
                    loading.set_visibility(False)
            except Exception as e:
                status_label.set_text('Error loading details')
                loading.set_visibility(False)
                ui.notify(f"Error loading trial details: {str(e)}", type='negative')
        
        asyncio.create_task(load_details())
    
    card.on('click', on_click)
    
    # Return None since we won't track processing status in the card anymore
    return None, None

async def process_trial_async(protocol_section, criteria, trial_card_element, trial_spinner):
    """Process a trial asynchronously and display results in right panel"""
    try:
        # Extract basic trial info
        identification_module = protocol_section.get('identificationModule', {})
        nct_id = identification_module.get('nctId', 'Unknown')
        
        logger.info(f"Starting async processing for trial {nct_id}")
        
        # Clear and update right panel with loading state
        details_panel.clear()
        with details_panel:
            loading = ui.spinner('dots', size='lg', color='primary')
            status_label = ui.label('Loading trial details...')
        
        # Get full study details for eligibility criteria with timeout
        try:
            full_study = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, lambda: get_full_study(nct_id)),
                timeout=30.0
            )
            if not full_study:
                logger.warning(f"Failed to get full study details for {nct_id}")
                status_label.set_text('Failed to fetch trial details')
                loading.set_visibility(False)
                return
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching study details for {nct_id}")
            status_label.set_text('Timeout fetching details')
            loading.set_visibility(False)
            return
        
        # Update status for mCODE extraction
        status_label.set_text('Extracting mCODE data...')
        
        # Extract mCODE data with timeout
        try:
            mcode_data = await asyncio.wait_for(extract_mcode_data(full_study), timeout=60.0)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout extracting mCODE data for {nct_id}")
            status_label.set_text('Timeout extracting mCODE')
            loading.set_visibility(False)
            return
        
        # Update status for NLP processing
        status_label.set_text('Processing eligibility criteria...')
        
        # Process with selected NLP engine
        selected_engine = engine_select.value
        engine = engines[selected_engine]
        
        # Process eligibility criteria with timeout
        try:
            # Use run.io_bound to handle synchronous NLP processing asynchronously
            display_data = await asyncio.wait_for(
                run.io_bound(engine.process_text, criteria),
                timeout=45.0
            )
            
            # Update the right panel with results
            details_panel.clear()
            with details_panel:
                display_trial_results(protocol_section, display_data, mcode_data)
            
            logger.info(f"Successfully processed trial {nct_id}")
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout processing criteria for trial {nct_id}")
            status_label.set_text('Timeout processing criteria')
            loading.set_visibility(False)
            ui.notify(f"Timeout processing trial {nct_id}", type='warning')
        except Exception as e:
            logger.error(f"Error processing criteria for trial {nct_id}: {str(e)}")
            status_label.set_text('Processing failed')
            loading.set_visibility(False)
            ui.notify(f"Failed to process trial {nct_id}: {str(e)}", type='warning')
            
    except Exception as e:
        logger.error(f"Error in async trial processing for {nct_id}: {str(e)}")
        status_label.set_text('Processing error')
        loading.set_visibility(False)
        ui.notify(f"Error processing trial {nct_id}", type='negative')
    finally:
        # Clean up any resources if needed
        pass

# Define search handler
current_page_token = None

def cancel_active_tasks():
    """Cancel all active processing tasks"""
    global active_tasks
    if active_tasks:
        logger.info(f"Cancelling {len(active_tasks)} active tasks")
        for task in active_tasks:
            if not task.done():
                task.cancel()
        active_tasks.clear()
        ui.notify("Cancelled all active processing tasks", type='info')

async def on_search(page_token=None):
    global current_page_token
    current_page_token = page_token
    
    # Cancel any existing tasks before starting new search
    cancel_active_tasks()
    
    trials_container.clear()
    details_panel.clear()
    
    with details_panel:
        # Enhanced loading state with comprehensive progress tracking
        loading = ui.spinner('dots', size='lg', color='primary')
        status_label = ui.label('Searching clinical trials...')
        progress_bar = ui.linear_progress(0).classes('w-full h-2')
        progress_bar.set_visibility(False)
        
        # Add cancel button for long-running operations
        with ui.row().classes('items-center gap-2'):
            cancel_button = ui.button('Cancel Processing', icon='cancel', on_click=cancel_active_tasks) \
                .props('outline color=negative') \
                .classes('ml-auto')
        
        def cleanup():
            loading.set_visibility(False)
            status_label.set_text("Search failed")
            progress_bar.set_visibility(False)
            cancel_button.set_visibility(False)
        
        try:
            # Validate inputs
            if not search_input.value.strip():
                raise ValueError("Please enter a search term")
            if not 1 <= limit_slider.value <= 20:
                raise ValueError("Results limit must be between 1-20")
            
        except ValueError as e:
            handle_error(e, "Validation error", cleanup)
            return
        
        try:
            ui.notify(f"Searching for: {search_input.value}")
            logger.info(f"Starting search for: {search_input.value}")
            logger.info(f"Search parameters - max_results: {limit_slider.value}, page_token: {page_token}")
            
            def search_cleanup():
                loading.set_visibility(False)
                status_label.set_text("Search operation failed")
            
            # Update status to show search in progress
            status_label.set_text('Connecting to ClinicalTrials.gov API...')
            
            # Perform async search with timeout
            try:
                def search_task():
                    logger.info(f"Calling API with search term: '{search_input.value}'")
                    logger.debug(f"Search parameters: max_results={limit_slider.value}, page_token={page_token}")
                    results = search_trials(
                        search_input.value,
                        fields=["NCTId", "BriefTitle", "Condition", "OverallStatus", "BriefSummary"],
                        max_results=int(limit_slider.value),
                        page_token=page_token,
                        use_cache=True
                    )
                    logger.info(f"API returned {len(results.get('studies', []))} results")
                    logger.debug(f"API response keys: {list(results.keys()) if results else 'None'}")
                    return results
                
                results = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, search_task),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                logger.error("Timeout during API search operation")
                handle_error(Exception("Search operation timed out"), "API timeout", search_cleanup)
                return
            except Exception as e:
                logger.error(f"Search operation failed: {str(e)}")
                handle_error(e, "Search operation failed", search_cleanup)
                return
            
            # Update status after search completes
            status_label.set_text('Processing trial data...')
            progress_bar.set_visibility(True)
            
            if not results:
                status_label.set_text('No results found')
                ui.notify('No trials found', type='info')
                logger.info("No trials found")
                return
            
            # Update status with pagination info
            status_text = f"Found {len(results.get('studies', []))} trials"
            logger.info(f"Search results: {len(results.get('studies', []))} trials found")
            if 'pagination' in results:
                page_number = results['pagination'].get('page_number', 1)
                status_text += f" (Page {page_number})"
                logger.info(f"Pagination info: page {page_number}")
            status_label.set_text(status_text)
            loading.set_visibility(False)
            
            # Update pagination controls
            if 'nextPageToken' in results:
                next_button.set_visibility(True)
                logger.debug(f"Next page token available: {results['nextPageToken'][:10]}...")
                next_button.on('click', lambda: on_search(results['nextPageToken']))
            else:
                next_button.set_visibility(False)
                logger.debug("No next page token available")
                
            if page_token:
                prev_button.set_visibility(True)
                logger.debug("Previous button enabled")
                prev_button.on('click', lambda: on_search(None))  # Go back to first page
            else:
                prev_button.set_visibility(False)
                logger.debug("Previous button disabled (on first page)")
            
            studies = results.get('studies', [])
            if not studies:
                status_label.set_text('No results found')
                ui.notify('No trials found', type='info')
                logger.info("No trials found")
                return

            # Initialize ClinicalTrials client
            ct = ClinicalTrials()
            
            # Store trial processing tasks
            processing_tasks = []
            
            # Process all studies (whether they're NCT IDs or full objects)
            for study in studies:
                try:
                    nct_id = None
                    if isinstance(study, str):
                        # For NCT IDs, fetch full details
                        nct_id = study
                        full_study = ct.get_study_fields(
                            search_expr=study,
                            fields=ct.study_fields['json'],
                            max_studies=1,
                            fmt='json'
                        )
                        if not full_study:
                            logger.warning(f"No data returned for study {study}")
                            continue
                        study = full_study[0]
                    else:
                        # For study objects, extract NCT ID
                        nct_id = study.get('protocolSection', {}).get('identificationModule', {}).get('nctId')
                    
                    # Always fetch full study details for eligibility criteria extraction
                    if nct_id:
                        logger.info(f"Fetching full study details for {nct_id}")
                        full_study_details = get_full_study(nct_id)
                        if full_study_details:
                            study = full_study_details
                        else:
                            logger.warning(f"Failed to fetch full study details for {nct_id}")
                    
                    # Process study object
                    protocol_section = study.get('protocolSection', {})
                    eligibility_module = protocol_section.get('eligibilityModule', {})
                    criteria = eligibility_module.get('eligibilityCriteria', '')
                    
                    # Create trial card immediately with basic information
                    with trials_container:
                        create_trial_card(protocol_section)
                    
                    # Start async processing for this trial
                    task = asyncio.create_task(
                        process_trial_async(protocol_section, criteria, None, None)
                    )
                    processing_tasks.append(task)
                    active_tasks.add(task)
                    # Add cleanup callback to remove task from active_tasks when done
                    task.add_done_callback(lambda t: active_tasks.discard(t))
                    
                except Exception as e:
                    logger.error(f"Failed to process study: {str(e)}")
                    continue
            
            # Track progress with proper variable scope
            completed = 0
            total = len(processing_tasks)
            
            # Create a wrapper to track completion with proper variable access
            async def track_progress(task, completed_ref, total_ref, progress_bar, status_label):
                try:
                    result = await task
                    completed_ref[0] += 1
                    progress = completed_ref[0] / total_ref if total_ref > 0 else 1
                    progress_bar.value = progress
                    status_label.set_text(f'Processed {completed_ref[0]} of {total_ref} trials...')
                    return result
                except Exception as e:
                    completed_ref[0] += 1
                    progress = completed_ref[0] / total_ref if total_ref > 0 else 1
                    progress_bar.value = progress
                    status_label.set_text(f'Processed {completed_ref[0]} of {total_ref} trials (some failed)...')
                    raise e
            
            # Process all tasks with progress tracking using mutable reference
            completed_ref = [0]  # Use list to allow modification in nested function
            processing_tasks_with_progress = [
                track_progress(task, completed_ref, total, progress_bar, status_label)
                for task in processing_tasks
            ]
            
            await asyncio.gather(*processing_tasks_with_progress)
            
            # Final status update
            status_label.set_text(f'Completed processing {total} trials')
            progress_bar.value = 1
                    
        except Exception as e:
            handle_error(e, "Search operation failed", cleanup)
        finally:
            loading.set_visibility(False)
            cancel_button.set_visibility(False)
            # Keep progress bar visible to show completion state


def display_trial_results(protocol_section, display_data, mcode_data=None):
    """Display detailed trial results in right panel"""
    logger.debug(f"Entering display_trial_results with display_data type: {type(display_data)}")
    
    # Validate input data - handle both dict and ProcessingResult objects
    if not display_data:
        logger.error(f"Invalid display_data: {display_data}")
        ui.label('Invalid trial data format').classes('text-red-500')
        return
    
    # Extract features from different data formats
    features = {}
    if isinstance(display_data, dict):
        # Handle dict format
        if 'single' in display_data:
            # New format: extraction result is under 'single' key
            features = display_data['single'].get('features', {})
        else:
            # Old format: features are directly in display_data
            features = display_data.get('features', {})
    elif hasattr(display_data, 'features'):
        # Handle ProcessingResult object
        features = display_data.features
    else:
        logger.error(f"Unsupported display_data format: {type(display_data)}")
        ui.label('Unsupported data format').classes('text-red-500')
        return
    
    # Create safe copy of features data
    try:
        features_copy = {
            'genomic_variants': features.get('genomic_variants', []),
            'biomarkers': features.get('biomarkers', []),
            'cancer_characteristics': features.get('cancer_characteristics', {}),
            'treatment_history': features.get('treatment_history', {}),
            'performance_status': features.get('performance_status', {}),
            'demographics': features.get('demographics', {})
        }
        logger.debug(f"Features structure validated: {features_copy.keys()}")
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        logger.error(traceback.format_exc())
        return
    
    # Detailed Trial View in Right Panel
    with details_panel.classes('w-full p-4'):
        with ui.card().classes('w-full'):
            # Consolidated expansion panel for all trial analysis
            with ui.expansion('TRIAL ANALYSIS', icon='analytics', value=True).classes('w-full bg-blue-50'):
                # Enhanced Trial Information Header with all mandatory fields
                with ui.card().classes('w-full p-6 bg-white shadow-md rounded-lg mb-4'):
                    # Header row with title, NCT ID, and status
                    with ui.row().classes('items-center justify-between w-full mb-4'):
                        with ui.column().classes('flex-grow'):
                            # Title - mandatory field
                            title = protocol_section.get('identificationModule', {}).get('officialTitle') or \
                                protocol_section.get('identificationModule', {}).get('briefTitle', 'No title available')
                            ui.label(title).classes('text-2xl font-bold text-gray-800')
                            
                            # NCT ID
                            nct_id = protocol_section.get('identificationModule', {}).get('nctId', 'N/A')
                            ui.label(f"NCT ID: {nct_id}").classes('text-sm text-gray-600 font-mono')
                        
                        # Trial status badge with enhanced styling
                        status_module = protocol_section.get('statusModule', {})
                        status = status_module.get('overallStatus', 'Unknown')
                        status_color = 'green' if status.lower() in ['active', 'recruiting', 'completed'] else 'orange' if status.lower() in ['not yet recruiting', 'suspended'] else 'red'
                        ui.badge(status.upper()).props(f'color={status_color} size=lg').classes('ml-4 px-3 py-1 font-bold')
                    
                    # Description - mandatory field (always visible, not in expansion)
                    description_module = protocol_section.get('descriptionModule', {})
                    brief_summary = description_module.get('briefSummary', 'No description available')
                    if brief_summary:
                        with ui.card().classes('w-full p-4 bg-white rounded-lg mb-4'):
                            ui.label('DESCRIPTION').classes('text-lg font-bold text-primary mb-2')
                            ui.markdown(brief_summary).classes('text-sm text-gray-700 leading-relaxed')
                    
                    # Trial metadata grid with dates and tags - mandatory fields
                    with ui.grid(columns=3).classes('w-full gap-4'):
                        # Start Date
                        start_date = status_module.get('startDate', 'Unknown')
                        with ui.card().classes('p-3 bg-white rounded-lg text-center'):
                            ui.label('START DATE').classes('text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1')
                            ui.label(start_date).classes('text-sm font-medium text-gray-800')
                        
                        # Completion Date
                        completion_date = status_module.get('completionDate', 'Unknown')
                        with ui.card().classes('p-3 bg-white rounded-lg text-center'):
                            ui.label('COMPLETION DATE').classes('text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1')
                            ui.label(completion_date).classes('text-sm font-medium text-gray-800')
                        
                        # Conditions/Tags - mandatory field
                        conditions_module = protocol_section.get('conditionsModule', {})
                        conditions = conditions_module.get('conditions', [])
                        
                        # Debug logging for conditions
                        logger.debug(f"display_trial_results - conditions_module: {conditions_module}")
                        logger.debug(f"display_trial_results - conditions: {conditions}")
                        logger.debug(f"display_trial_results - conditions type: {type(conditions)}")
                        with ui.card().classes('p-3 bg-blue-50 rounded-lg border border-blue-200'):
                            ui.label('CONDITIONS').classes('text-xs font-semibold text-blue-800 uppercase tracking-wide mb-2')
                            if conditions:
                                # Show first 2-3 conditions with tooltip for all
                                display_conditions = conditions[:3]  # Show up to 3 conditions
                                if len(conditions) > 3:
                                    display_text = f"{', '.join(display_conditions)} +{len(conditions) - 3} more"
                                else:
                                    display_text = ', '.join(display_conditions)
                                
                                with ui.tooltip(f"All conditions: {', '.join(conditions)}"):
                                    ui.label(display_text).classes('text-sm font-medium text-blue-900 cursor-help')
                            else:
                                ui.label('No conditions specified').classes('text-sm text-gray-500')
                    
                    # Additional trial information
                    with ui.row().classes('w-full justify-between mt-3'):
                        # Study Type
                        design_module = protocol_section.get('designModule', {})
                        study_type = design_module.get('studyType', 'Unknown')
                        with ui.column().classes('items-center'):
                            ui.label('Study Type').classes('text-xs text-gray-500')
                            ui.label(study_type).classes('text-sm font-medium')
                        
                        # Phase
                        phase = design_module.get('phases', ['Unknown'])[0] if design_module.get('phases') else 'Unknown'
                        with ui.column().classes('items-center'):
                            ui.label('Phase').classes('text-xs text-gray-500')
                            ui.label(phase).classes('text-sm font-medium')
                        
                        # Enrollment
                        enrollment_info = status_module.get('enrollmentInfo', {})
                        enrollment = enrollment_info.get('count', 'Unknown')
                        with ui.column().classes('items-center'):
                            ui.label('Enrollment').classes('text-xs text-gray-500')
                            ui.label(str(enrollment)).classes('text-sm font-medium')
            
            # Enhanced Patient Match Analysis
            with ui.expansion('PATIENT MATCH ANALYSIS', icon='person', value=True).classes('w-full mb-4'):
                # Match strength indicator - use hasattr for ProcessingResult objects
                if display_data and hasattr(display_data, 'features') and matching_toggle.value:
                    from src.matching_engine.matcher import PatientMatcher
                    matcher = PatientMatcher()
                    
                    # Prepare patient biomarkers as dict for matching, ignoring NOT_FOUND
                    patient_biomarkers = {b['name']: b['status'] for b in patient_profile['biomarkers'] if b['name'] != 'NOT_FOUND'}
                    patient_variants = [v.get('gene', '') for v in patient_profile.get('genomic_variants', [])
                                            if v.get('gene') != 'NOT_FOUND']
                    
                    patient_data = {
                        'cancer_type': patient_profile.get('cancer_type', ''),
                        'stage': patient_profile.get('stage', ''),
                        'biomarkers': patient_biomarkers,
                        'genomic_variants': patient_variants,
                        'cancer_characteristics': {
                            'stage': patient_profile.get('stage', ''),
                            'histology': patient_profile.get('histology', ''),
                            'grade': patient_profile.get('grade', ''),
                            'tnm_staging': patient_profile.get('tnm_staging', {'t': '', 'n': '', 'm': ''})
                        }
                    }
                    
                    # Use the already extracted features from the proper format handling
                    # features variable is already properly extracted above
                    
                    # Extract biomarkers and variants, filtering out NOT_FOUND
                    biomarkers = [b for b in features.get('biomarkers', []) if b.get('name') != 'NOT_FOUND']
                    variants = [v for v in features.get('genomic_variants', []) if v.get('gene') != 'NOT_FOUND']
                    
                    trial_features = {
                        'cancer_type': features.get('cancer_type', ''),
                        'biomarkers': {b['name']: b['status'] for b in biomarkers},
                        'genomic_variants': [v['gene'] for v in variants],
                        'cancer_characteristics': {
                            'stage': features.get('stage', ''),
                            'histology': features.get('histology', ''),
                            'grade': features.get('grade', ''),
                            'tnm_staging': features.get('tnm_staging', {'t': '', 'n': '', 'm': ''})
                        }
                    }
                    
                    # Calculate match score with detailed breakdown
                    logger.debug(f"Calculating match score for trial {protocol_section.get('identificationModule', {}).get('nctId')}")
                    logger.debug(f"Patient data: {patient_data}")
                    logger.debug(f"Trial features: {trial_features}")
                    match_score, match_details = matcher.calculate_match_score(patient_data, trial_features, return_details=True)
                    match_desc = matcher.get_match_description(match_score)
                    logger.info(f"Match score for trial {protocol_section.get('identificationModule', {}).get('nctId')}: {match_score}% - {match_desc}")
                    logger.debug(f"Match details: {match_details}")
                    
                    # Enhanced match visualization with comprehensive UI
                    with ui.column().classes('w-full gap-4'):
                        # Header with quick stats
                        with ui.row().classes('items-center justify-between w-full p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg'):
                            with ui.column().classes('items-center'):
                                ui.label('OVERALL MATCH').classes('text-xs font-semibold text-gray-600 uppercase tracking-wide')
                                ui.label(f"{match_score}%").classes('text-3xl font-bold text-primary')
                                ui.label(match_desc).classes('text-sm font-medium')
                            
                            # Quick stats grid
                            with ui.grid(columns=4).classes('gap-4'):
                                # Cancer Type
                                with ui.column().classes('items-center'):
                                    ui.icon('local_hospital').classes(f'text-{"green" if match_details["cancer_type"] else "red"}-500 text-xl')
                                    ui.label('Cancer Type').classes('text-xs font-medium')
                                    ui.label('30%').classes('text-xs text-gray-500')
                                
                                # Biomarkers
                                with ui.column().classes('items-center'):
                                    ui.icon('science').classes(f'text-{"green" if match_details["biomarkers"] else "red"}-500 text-xl')
                                    ui.label('Biomarkers').classes('text-xs font-medium')
                                    ui.label('25%').classes('text-xs text-gray-500')
                                
                                # Variants
                                with ui.column().classes('items-center'):
                                    ui.icon('dna').classes(f'text-{"green" if match_details["genomic_variants"] else "red"}-500 text-xl')
                                    ui.label('Variants').classes('text-xs font-medium')
                                    ui.label('25%').classes('text-xs text-gray-500')
                                
                                # Stage/Grade
                                with ui.column().classes('items-center'):
                                    ui.icon('trending_up').classes(f'text-{"green" if match_details["stage_grade"] else "red"}-500 text-xl')
                                    ui.label('Stage/Grade').classes('text-xs font-medium')
                                    ui.label('20%').classes('text-xs text-gray-500')
                        
                        # Main match score visualization
                        with ui.card().classes('w-full p-6 bg-white shadow-md rounded-lg'):
                            with ui.column().classes('items-center w-full gap-4'):
                                # Match score progress with enhanced styling
                                color = 'green' if match_score >= 75 else 'orange' if match_score >= 50 else 'red'
                                with ui.row().classes('items-center w-full gap-4'):
                                    ui.label('MATCH SCORE').classes('text-lg font-bold text-gray-700')
                                    ui.linear_progress(match_score/100).classes('flex-grow h-6') \
                                        .props(f'color={color} rounded stripe')
                                    ui.label(f"{match_score}%").classes('text-2xl font-bold min-w-[60px] text-center')
                                
                                # Match description with enhanced tooltip
                                with ui.tooltip('''Score Breakdown:
• Cancer Type: 30% weight
• Biomarkers: 25% weight
• Genomic Variants: 25% weight
• Stage/Grade: 20% weight'''):
                                    ui.label(match_desc.upper()).classes('text-lg font-semibold text-gray-600') \
                                        .props('tooltip-color=primary')
# Detailed matching breakdown
                        with ui.expansion('Detailed Matching Analysis').classes('w-full mt-2'):
                            with ui.card().classes('w-full p-4 bg-gray-50'):
                                ui.label('mCODE ELEMENT MATCHING').classes('text-lg font-bold mb-4 text-primary')
                                
                                # Cancer Type match
                                with ui.card().classes('w-full p-4'):
                                    with ui.row().classes('items-center justify-between'):
                                        with ui.row().classes('items-center gap-2'):
                                            ui.icon('check_circle' if match_details['cancer_type'] else 'cancel') \
                                                .classes(f'text-{"green" if match_details["cancer_type"] else "red"}-500')
                                            ui.label('CANCER TYPE').classes('font-bold')
                                        ui.label('30% weight').classes('text-xs text-gray-500')
                                    
                                    with ui.column().classes('ml-8 gap-1'):
                                        ui.label(f"Patient: {patient_data['cancer_type']}").classes('text-sm')
                                        ui.label(f"Trial: {trial_features['cancer_type']}").classes('text-sm')
                                
                                # Biomarkers match
                                with ui.card().classes('w-full p-4'):
                                    with ui.row().classes('items-center justify-between'):
                                        with ui.row().classes('items-center gap-2'):
                                            ui.icon('check_circle' if match_details['biomarkers'] else 'cancel') \
                                                .classes(f'text-{"green" if match_details["biomarkers"] else "red"}-500')
                                            ui.label('BIOMARKERS').classes('font-bold')
                                        ui.label('25% weight').classes('text-xs text-gray-500')
                                    
                                    with ui.column().classes('ml-8 gap-1'):
                                        if match_details['biomarkers']:
                                            matched = [b for b in patient_data['biomarkers'] if b in trial_features['biomarkers']]
                                            for b in matched:
                                                with ui.row().classes('items-center gap-2'):
                                                    ui.icon('check').classes('text-green-500 text-sm')
                                                    ui.label(f"{b}: {patient_data['biomarkers'][b]}").classes('text-sm')
                                        else:
                                            ui.label('No biomarker matches').classes('text-sm text-gray-500')
                                
                                # Genomic Variants match
                                with ui.card().classes('w-full p-4'):
                                    with ui.row().classes('items-center justify-between'):
                                        with ui.row().classes('items-center gap-2'):
                                            ui.icon('check_circle' if match_details['genomic_variants'] else 'cancel') \
                                                .classes(f'text-{"green" if match_details["genomic_variants"] else "red"}-500')
                                            ui.label('GENOMIC VARIANTS').classes('font-bold')
                                        ui.label('25% weight').classes('text-xs text-gray-500')
                                    
                                    with ui.column().classes('ml-8 gap-1'):
                                        if match_details['genomic_variants']:
                                            matched = [v for v in patient_data['genomic_variants'] if v in trial_features['genomic_variants']]
                                            for v in matched:
                                                with ui.row().classes('items-center gap-2'):
                                                    ui.icon('check').classes('text-green-500 text-sm')
                                                    ui.label(v).classes('text-sm')
                                        else:
                                            ui.label('No variant matches').classes('text-sm text-gray-500')
                                
                                # Stage/Grade match
                                with ui.card().classes('w-full p-4'):
                                    with ui.row().classes('items-center justify-between'):
                                        with ui.row().classes('items-center gap-2'):
                                            ui.icon('check_circle' if match_details['stage_grade'] else 'cancel') \
                                                .classes(f'text-{"green" if match_details["stage_grade"] else "red"}-500')
                                            ui.label('STAGE/GRADE').classes('font-bold')
                                        ui.label('20% weight').classes('text-xs text-gray-500')
                                    
                                    with ui.column().classes('ml-8 gap-1'):
                                        if match_details['stage_grade']:
                                            ui.label(f"Patient Stage: {patient_data['cancer_characteristics']['stage']}").classes('text-sm')
                                            ui.label(f"Trial Minimum: {trial_features['cancer_characteristics']['stage']}").classes('text-sm')
                                        else:
                                            ui.label('Stage/grade mismatch').classes('text-sm text-gray-500')
            
        
        # Eligibility criteria
        with ui.expansion('View Eligibility Criteria').classes('w-full'):
            ui.markdown(f"```\n{protocol_section.get('eligibilityModule', {}).get('eligibilityCriteria', 'No criteria available')}\n```")
        
        with ui.expansion('mCODE Features').classes('w-full'):
            with ui.tabs().classes('w-full') as tabs:
                demographics_tab = ui.tab('Demographics')
                cancer_tab = ui.tab('Cancer Condition')
                genomics_tab = ui.tab('Genomic Variants')
                biomarkers_tab = ui.tab('Biomarkers')
                treatment_tab = ui.tab('Treatment History')
                performance_tab = ui.tab('Performance Status')
            
            with ui.tab_panels(tabs, value=demographics_tab).classes('w-full'):
                # Demographics
                with ui.tab_panel(demographics_tab):
                    demographics = features.get('demographics', {})
                    if demographics:
                        with ui.card().classes('w-full p-6 bg-white shadow-md rounded-lg hover:shadow-lg transition-shadow'):
                            ui.label('DEMOGRAPHIC DETAILS').classes('text-xl font-bold mb-4 text-primary')
                            with ui.grid(columns=2).classes('w-full gap-4'):
                                for key, value in demographics.items():
                                    extracted = bool(value) and value != 'Not specified'
                                    with ui.card().classes(f'p-4 rounded-lg hover:bg-white transition-colors '
                                                            f'{"bg-gray-50" if extracted else "bg-gray-50 opacity-50"}'):
                                        ui.label(key.replace('_', ' ').title()).classes('font-medium text-sm')
                                        ui.label(str(value) if value else 'Not specified').classes('' if extracted else 'opacity-70')
                    else:
                        ui.label('No demographic data available').classes('text-gray-500')
                
                # Cancer Condition
                with ui.tab_panel(cancer_tab):
                    cancer_data = features.get('cancer_characteristics', {})
                    if cancer_data:
                        with ui.card().classes('w-full p-4'):
                            ui.label('CANCER CHARACTERISTICS').classes('text-xl font-bold mb-4 text-primary')
                            with ui.grid(columns=2).classes('w-full gap-4'):
                                for key, value in cancer_data.items():
                                    # Dim card if value is empty
                                    extracted = bool(value) and value != 'Not specified'
                                    card_classes = f'p-4 rounded-lg hover:bg-white transition-colors '
                                    card_classes += 'bg-gray-50 opacity-50' if not extracted else 'bg-gray-50'
                                    
                                    with ui.card().classes(card_classes):
                                        ui.label(key.replace('_', ' ').title()).classes('font-medium text-sm')
                                        
                                        # Show "not mentioned" for empty values
                                        if isinstance(value, dict):
                                            for k, v in value.items():
                                                if v:
                                                    ui.label(f"{k}: {v}").classes('text-xs')
                                                else:
                                                    ui.label(f"{k}: not mentioned").classes('text-xs opacity-50')
                                        else:
                                            if value:
                                                ui.label(str(value)).classes('')
                                            else:
                                                ui.label('not mentioned').classes('opacity-50')
                    else:
                        ui.label('No cancer characteristics data').classes('text-gray-500')
                
                # Genomic Variants
                with ui.tab_panel(genomics_tab):
                    variants = features_copy.get('genomic_variants', [])
                    with ui.card().classes('w-full p-4'):
                        ui.label('GENOMIC VARIANTS').classes('text-xl font-bold mb-4 text-primary')
                        
                        if not variants:
                            ui.label('No genomic variants found').classes('text-gray-500')
                        else:
                            with ui.grid(columns=2).classes('w-full gap-4'):
                                for variant in variants:
                                    if not isinstance(variant, dict):
                                        continue
                                    
                                    # Dim card if variant is NOT_FOUND
                                    if variant.get('gene') == 'NOT_FOUND':
                                        with ui.card().classes('p-4 bg-gray-50 rounded-lg hover:bg-white transition-colors opacity-50'):
                                            ui.label("No genomic variants mentioned").classes('font-bold text-gray-500')
                                    else:
                                        with ui.card().classes('p-4 bg-gray-50 rounded-lg hover:bg-white transition-colors'):
                                            with ui.column().classes('gap-1'):
                                                ui.label(f"Gene: {variant.get('gene', 'Unknown')}").classes('font-bold')
                                                ui.label(f"Variant: {variant.get('variant', 'N/A')}").classes('text-sm')
                                                significance = variant.get('significance', 'N/A')
                                                if significance and significance != 'N/A':
                                                    ui.label(f"Significance: {significance}").classes('text-sm text-blue-600')
                
                # Biomarkers
                with ui.tab_panel(biomarkers_tab):
                    biomarkers = features_copy.get('biomarkers', [])
                    with ui.card().classes('w-full p-4'):
                        ui.label('BIOMARKERS').classes('text-xl font-bold mb-4 text-primary')
                        
                        if not biomarkers:
                            ui.label('No biomarkers found').classes('text-gray-500')
                        else:
                            with ui.grid(columns=2).classes('w-full gap-4'):
                                for biomarker in biomarkers:
                                    if not isinstance(biomarker, dict):
                                        continue
                                    
                                    # Dim card if biomarker is NOT_FOUND
                                    if biomarker.get('name') == 'NOT_FOUND':
                                        with ui.card().classes('p-4 bg-gray-50 rounded-lg hover:bg-white transition-colors opacity-50'):
                                            ui.label("No biomarkers mentioned").classes('font-bold text-gray-500')
                                    else:
                                        with ui.card().classes('p-4 bg-gray-50 rounded-lg hover:bg-white transition-colors'):
                                            with ui.column().classes('gap-1'):
                                                ui.label(biomarker.get('name', 'Unknown')).classes('font-bold')
                                                
                                                # Show status with "not mentioned" if missing
                                                status = biomarker.get('status', 'N/A')
                                                if status and status != 'N/A':
                                                    with ui.row().classes('items-center gap-2'):
                                                        ui.label('Status:').classes('text-sm font-medium')
                                                        ui.label(status).classes('text-sm')
                                                else:
                                                    with ui.row().classes('items-center gap-2 opacity-50'):
                                                        ui.label('Status:').classes('text-sm font-medium')
                                                        ui.label('not mentioned').classes('text-sm')
                                                
                                                # Show value with "not mentioned" if missing
                                                value = biomarker.get('value', 'N/A')
                                                if value and value != 'N/A':
                                                    with ui.row().classes('items-center gap-2'):
                                                        ui.label('Value:').classes('text-sm font-medium')
                                                        ui.label(value).classes('text-sm')
                                                else:
                                                    with ui.row().classes('items-center gap-2 opacity-50'):
                                                        ui.label('Value:').classes('text-sm font-medium')
                                                        ui.label('not mentioned').classes('text-sm')
                
                # Treatment History
                with ui.tab_panel(treatment_tab):
                    treatment = features.get('treatment_history', {})
                    if treatment:
                        with ui.card().classes('w-full p-4'):
                            ui.label('TREATMENT HISTORY').classes('text-xl font-bold mb-4 text-primary')
                            with ui.grid(columns=2).classes('w-full gap-4'):
                                for key, value in treatment.items():
                                    # Dim card if value is empty
                                    extracted = bool(value) and value != 'Not specified'
                                    card_classes = f'p-4 rounded-lg hover:bg-white transition-colors '
                                    card_classes += 'bg-gray-50 opacity-50' if not extracted else 'bg-gray-50'
                                    
                                    with ui.card().classes(card_classes):
                                        ui.label(key.replace('_', ' ').title()).classes('font-medium text-sm')
                                        
                                        # Show "not mentioned" for empty values
                                        if value:
                                            ui.label(str(value)).classes('')
                                        else:
                                            ui.label('not mentioned').classes('opacity-50')
                    else:
                        ui.label('No treatment history data').classes('text-gray-500')
                
                # Performance Status
                with ui.tab_panel(performance_tab):
                    performance = features.get('performance_status', {})
                    if performance:
                        with ui.card().classes('w-full p-4'):
                            ui.label('PERFORMANCE STATUS').classes('text-xl font-bold mb-4 text-primary')
                            with ui.grid(columns=2).classes('w-full gap-4'):
                                for key, value in performance.items():
                                    # Dim card if value is empty
                                    extracted = bool(value) and value != 'Not specified'
                                    card_classes = f'p-4 rounded-lg hover:bg-white transition-colors '
                                    card_classes += 'bg-gray-50 opacity-50' if not extracted else 'bg-gray-50'
                                    
                                    with ui.card().classes(card_classes):
                                        ui.label(key.replace('_', ' ').title()).classes('font-medium text-sm')
                                        
                                        # Show "not mentioned" for empty values
                                        if value:
                                            ui.label(str(value)).classes('')
                                        else:
                                            ui.label('not mentioned').classes('opacity-50')
                    else:
                        ui.label('No performance status data').classes('text-gray-500')
                

        # Raw mCODE Data Panel
        with ui.expansion('RAW mCODE DATA', icon='code', value=False).classes('w-full'):
            with ui.card().classes('w-full p-4 bg-gray-50'):
                ui.label('RAW mCODE EXTRACTION DATA').classes('text-xl font-bold mb-4 text-primary')
                
                if not mcode_data:
                    ui.label('No mCODE data available for raw display').classes('text-gray-500')
                else:
                    # Display raw mCODE data in a structured format
                    with ui.tabs().classes('w-full') as raw_tabs:
                        raw_overview_tab = ui.tab('Overview')
                        raw_nlp_tab = ui.tab('NLP Results')
                        raw_codes_tab = ui.tab('Extracted Codes')
                        raw_mapped_tab = ui.tab('Mapped mCODE')
                        raw_structured_tab = ui.tab('Structured Data')
                        raw_validation_tab = ui.tab('Validation')
                    
                    with ui.tab_panels(raw_tabs, value=raw_overview_tab).classes('w-full'):
                        # Overview tab
                        with ui.tab_panel(raw_overview_tab):
                            with ui.card().classes('w-full p-4'):
                                ui.label('mCODE EXTRACTION OVERVIEW').classes('text-lg font-bold mb-4')
                                with ui.grid(columns=2).classes('w-full gap-4'):
                                    # Basic trial info
                                    with ui.card().classes('p-3 bg-blue-50'):
                                        ui.label('TRIAL INFO').classes('font-bold text-blue-700 text-sm mb-2')
                                        ui.label(f"NCT ID: {mcode_data.get('nct_id', 'N/A')}").classes('text-sm')
                                        ui.label(f"Title: {mcode_data.get('title', 'N/A')}").classes('text-sm')
                                        ui.label(f"Status: {mcode_data.get('status', 'N/A')}").classes('text-sm')
                                    
                                    # Extraction metrics
                                    with ui.card().classes('p-3 bg-green-50'):
                                        ui.label('EXTRACTION METRICS').classes('font-bold text-green-700 text-sm mb-2')
                                        nlp_entities = len(mcode_data.get('nlp_result', {}).get('entities', []))
                                        ui.label(f"NLP Entities: {nlp_entities}").classes('text-sm')
                                        mapped_count = len(mcode_data.get('mapped_mcode', []))
                                        ui.label(f"Mapped Elements: {mapped_count}").classes('text-sm')
                                        validation = mcode_data.get('validation', {})
                                        ui.label(f"Validation: {'VALID' if validation.get('is_valid') else 'INVALID'}").classes('text-sm')
                        
                        # NLP Results tab
                        with ui.tab_panel(raw_nlp_tab):
                            nlp_result = mcode_data.get('nlp_result', {})
                            if nlp_result and nlp_result.get('entities'):
                                with ui.card().classes('w-full p-4'):
                                    ui.label('NLP EXTRACTED ENTITIES').classes('text-lg font-bold mb-4')
                                    with ui.column().classes('w-full gap-2'):
                                        for entity in nlp_result.get('entities', []):
                                            with ui.card().classes('p-3 bg-gray-50 rounded-lg'):
                                                with ui.row().classes('items-center justify-between'):
                                                    ui.label(entity.get('text', 'Unknown')).classes('font-medium')
                                                    if entity.get('confidence'):
                                                        ui.label(f"Confidence: {entity.get('confidence'):.2f}").classes('text-xs text-gray-600')
                                                if entity.get('codes'):
                                                    with ui.row().classes('items-center gap-2 mt-1'):
                                                        ui.label('Codes:').classes('text-xs font-medium')
                                                        for system, code in entity.get('codes', {}).items():
                                                            ui.label(f"{system}: {code}").classes('text-xs text-blue-600')
                            else:
                                ui.label('No NLP entities extracted').classes('text-gray-500')
                        
                        # Extracted Codes tab
                        with ui.tab_panel(raw_codes_tab):
                            extracted_codes = mcode_data.get('extracted_codes', {})
                            if extracted_codes and extracted_codes.get('extracted_codes'):
                                with ui.card().classes('w-full p-4'):
                                    ui.label('CODE SYSTEM EXTRACTION').classes('text-lg font-bold mb-4')
                                    for system, codes in extracted_codes.get('extracted_codes', {}).items():
                                        with ui.expansion(f"{system.upper()} CODES ({len(codes)})").classes('w-full mb-2'):
                                            with ui.column().classes('w-full gap-2'):
                                                for code_info in codes:
                                                    with ui.card().classes('p-3 bg-gray-50 rounded-lg'):
                                                        with ui.row().classes('items-center justify-between'):
                                                            ui.label(f"{code_info.get('text', 'Unknown')}").classes('font-medium')
                                                            ui.label(f"Confidence: {code_info.get('confidence', 0):.2f}").classes('text-xs text-gray-600')
                                                        ui.label(f"Code: {code_info.get('code', 'N/A')}").classes('text-sm')
                                                        if code_info.get('system'):
                                                            ui.label(f"System: {code_info.get('system')}").classes('text-xs text-gray-500')
                            else:
                                ui.label('No codes extracted from eligibility criteria').classes('text-gray-500')
                        
                        # Mapped mCODE tab
                        with ui.tab_panel(raw_mapped_tab):
                            mapped_mcode = mcode_data.get('mapped_mcode', [])
                            if mapped_mcode:
                                with ui.card().classes('w-full p-4'):
                                    ui.label('MAPPED mCODE ELEMENTS').classes('text-lg font-bold mb-4')
                                    with ui.column().classes('w-full gap-3'):
                                        for element in mapped_mcode:
                                            with ui.card().classes('p-4 bg-white shadow-sm rounded-lg'):
                                                with ui.row().classes('items-center justify-between mb-2'):
                                                    ui.label(element.get('element_name', 'Unknown')).classes('font-bold text-lg')
                                                    if element.get('confidence'):
                                                        ui.badge(f"{element.get('confidence'):.2f}").props(f'color={"green" if element.get("confidence", 0) > 0.7 else "orange" if element.get("confidence", 0) > 0.5 else "red"}')
                                                ui.label(f"Type: {element.get('element_type', 'Unknown')}").classes('text-sm')
                                                if element.get('value'):
                                                    ui.label(f"Value: {element.get('value')}").classes('text-sm text-blue-600')
                                                if element.get('source_text'):
                                                    with ui.expansion('Source Text').classes('w-full mt-2'):
                                                        ui.markdown(f"`{element.get('source_text')}`").classes('text-xs')
                            else:
                                ui.label('No mCODE elements mapped').classes('text-gray-500')
                        
                        # Structured Data tab
                        with ui.tab_panel(raw_structured_tab):
                            structured_data = mcode_data.get('structured_data', {})
                            if structured_data:
                                with ui.card().classes('w-full p-4'):
                                    ui.label('STRUCTURED mCODE DATA').classes('text-lg font-bold mb-4')
                                    ui.markdown(f"```json\n{json.dumps(structured_data, indent=2)}\n```").classes('text-xs font-mono')
                            else:
                                ui.label('No structured mCODE data available').classes('text-gray-500')
                        
                        # Validation tab
                        with ui.tab_panel(raw_validation_tab):
                            validation = mcode_data.get('validation', {})
                            if validation:
                                with ui.card().classes('w-full p-4'):
                                    ui.label('mCODE VALIDATION RESULTS').classes('text-lg font-bold mb-4')
                                    with ui.grid(columns=2).classes('w-full gap-4'):
                                        # Validation status
                                        with ui.card().classes(f'p-3 {"bg-green-50" if validation.get("is_valid") else "bg-red-50"}'):
                                            ui.label('VALIDATION STATUS').classes('font-bold text-sm mb-2')
                                            ui.label('VALID' if validation.get('is_valid') else 'INVALID').classes(f'font-bold {"text-green-600" if validation.get("is_valid") else "text-red-600"}')
                                        
                                        # Required elements
                                        with ui.card().classes('p-3 bg-blue-50'):
                                            ui.label('REQUIRED ELEMENTS').classes('font-bold text-sm mb-2')
                                            required = validation.get('required_elements_present', [])
                                            ui.label(f"{len(required)} present").classes('text-sm')
                                        
                                        # Optional elements
                                        with ui.card().classes('p-3 bg-purple-50'):
                                            ui.label('OPTIONAL ELEMENTS').classes('font-bold text-sm mb-2')
                                            optional = validation.get('optional_elements_present', [])
                                            ui.label(f"{len(optional)} present").classes('text-sm')
                                        
                                        # Missing elements
                                        with ui.card().classes('p-3 bg-orange-50'):
                                            ui.label('MISSING ELEMENTS').classes('font-bold text-sm mb-2')
                                            missing = validation.get('missing_elements', [])
                                            ui.label(f"{len(missing)} missing").classes('text-sm')
                                    
                                    # Detailed validation results
                                    if validation.get('validation_details'):
                                        with ui.expansion('Detailed Validation').classes('w-full mt-4'):
                                            ui.markdown(f"```json\n{json.dumps(validation.get('validation_details', {}), indent=2)}\n```").classes('text-xs font-mono')
                            else:
                                ui.label('No validation data available').classes('text-gray-500')

# Reset function
def on_reset():
    # Cancel any active tasks before clearing
    cancel_active_tasks()
    trials_container.clear()
    search_input.value = patient_profile['cancer_type']
    ui.notify('Search results cleared', type='info')

# Bind handlers to buttons
search_button.on('click', on_search)
reset_button.on('click', on_reset)

ui.run(title='mCODE Clinical Trials Search')