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
from typing import Dict, List, Optional, Union, Callable
from pytrials.client import ClinicalTrials
from src.utils.config import Config
from src.utils.cache import CacheManager

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
from nicegui import ui

from src.pipeline.extraction_pipeline import ExtractionPipeline
from src.nlp_engine.regex_nlp_engine import RegexNLPEngine
from src.nlp_engine.spacy_nlp_engine import SpacyNLPEngine
from src.nlp_engine.llm_nlp_engine import LLMNLPEngine

class ClinicalTrialsAPIError(Exception):
    """Base exception for ClinicalTrialsAPI errors"""
    pass

def search_trials(search_expr: str, fields=None, max_results: int = 100):
    """
    Search for clinical trials matching the expression
    
    Args:
        search_expr: Search expression (e.g., "breast cancer")
        fields: List of fields to retrieve (default: ['NCTId', 'BriefTitle', 'Conditions'])
        max_results: Maximum number of results to return (default: 100)
        
    Returns:
        Dictionary containing search results
        
    Raises:
        ClinicalTrialsAPIError: If there's an error with the API request
    """
    # Initialize config and cache manager
    config = Config()
    cache_manager = CacheManager(config)
    
    # Set default fields if none provided
    if fields is None:
        fields = ['NCTId', 'BriefTitle', 'Condition']  # 'Condition' is singular in JSON format
        
    # Initialize client and log available fields
    ct = ClinicalTrials()
    logger.debug(f"Available fields: {ct.study_fields}")
    
    # Create cache key
    cache_key_data = f"search:{search_expr}:{','.join(fields) if fields else 'all'}:{max_results}"
    cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()
    
    # Try to get from cache first
    cached_result = cache_manager.get(cache_key)
    if cached_result:
        return {'studies': cached_result}
    
    try:
        # Rate limiting
        time.sleep(config.rate_limit_delay)
        
        # Initialize pytrials client
        ct = ClinicalTrials()
        
        # Get study fields with specified fields
        result = ct.get_study_fields(
            search_expr=search_expr,
            fields=fields,
            max_studies=max_results,
            fmt='json'
        )
        logger.debug(f"API response received with {len(result)} studies")
        
        # Convert to consistent format
        if not result:
            return {'studies': []}
            
        # Cache the result
        cache_manager.set(cache_key, result)
        
        return {'studies': result}
    except Exception as e:
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
        
        # Use pytrials get_study method
        result = ct.get_study(nct_id)
        
        # Cache the result
        cache_manager.set(cache_key, result)
        
        return result
    except Exception as e:
        raise ClinicalTrialsAPIError(f"API request failed: {str(e)}")

# Initialize extraction pipelines with caching
extraction_cache = {}
engines = {
    'Regex': ExtractionPipeline(engine_type='Regex'),
    'SpaCy': ExtractionPipeline(engine_type='SpaCy'),
    'LLM': ExtractionPipeline(engine_type='LLM')
}

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
                    
                
    # Results display
    trials_container = ui.column().classes('w-full p-4')

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

# Define search handler
async def on_search():
    trials_container.clear()
    with trials_container:
        # Show loading state
        loading = ui.spinner('dots', size='lg', color='primary')
        status_label = ui.label('Searching clinical trials...')
        
        def cleanup():
            loading.set_visibility(False)
            status_label.set_text("Search failed")
        
        try:
            # Validate inputs
            if not search_input.value.strip():
                raise ValueError("Please enter a search term")
            if not 1 <= limit_slider.value <= 20:
                raise ValueError("Results limit must be between 1-20")
            
            # Rest of search logic...
            
        except ValueError as e:
            handle_error(e, "Validation error", cleanup)
        except Exception as e:
            handle_error(e, "Unexpected search error", cleanup)
        
        try:
            ui.notify(f"Searching for: {search_input.value}")
            logger.info(f"Starting search for: {search_input.value}")
            
            def search_cleanup():
                loading.set_visibility(False)
                status_label.set_text("Search operation failed")
            
            # Perform async search
            def search_task():
                logger.info(f"Calling API with search term: '{search_input.value}'")
                results = search_trials(search_input.value, max_results=int(limit_slider.value))
                logger.info(f"API returned {len(results.get('studies', []))} results")
                return results
            
            results = await asyncio.get_event_loop().run_in_executor(None, search_task)
            
            if not results:
                status_label.set_text('No results found')
                ui.notify('No trials found', type='info')
                logger.info("No trials found")
                return
            
            # Update status
            status_label.set_text(f"Found {len(results.get('studies', []))} trials")
            loading.set_visibility(False)
            
            studies = results.get('studies', [])
            if not studies:
                status_label.set_text('No results found')
                ui.notify('No trials found', type='info')
                logger.info("No trials found")
                return

            # Initialize ClinicalTrials client
            ct = ClinicalTrials()
            
            # Process all studies (whether they're NCT IDs or full objects)
            for study in studies:
                try:
                    if isinstance(study, str):
                        # For NCT IDs, fetch full details
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
                    
                    # Process study object
                    protocol_section = study.get('protocolSection', {})
                    eligibility_module = protocol_section.get('eligibilityModule', {})
                    criteria = eligibility_module.get('eligibilityCriteria', '')
                    # Initialize extraction_result to None
                    extraction_result = None
                    
                    # Show processing status
                    processing_label = ui.label(f"Processing trial {protocol_section.get('identificationModule', {}).get('nctId')}...")
                
                except Exception as e:
                    logger.error(f"Failed to process study: {str(e)}")
                    continue
                
                try:
                    # Check cache before processing
                    extraction_result = None
                    if criteria in extraction_cache:
                        extraction_result = extraction_cache[criteria]
                        logger.info(f"Using cached extraction for trial {protocol_section.get('identificationModule', {}).get('nctId')}")
                    elif criteria:
                        extraction_status = ui.label('Extracting mCODE features...')
                        ui.notify('Starting mCODE feature extraction...', type='info')
                        selected_engine = engine_select.value
                        logger.info(f"Starting feature extraction for trial {protocol_section.get('identificationModule', {}).get('nctId')} using {selected_engine} engine")
                        logger.debug(f"Engine selection state: {engine_select.value}")
                        
                        # Run extraction in executor to prevent blocking
                        def extraction_task():
                            try:
                                logger.info(f"Starting extraction for criteria: {criteria[:100]}...")
                                selected_engine = engine_select.value
                                logger.debug(f"Executing extraction with engine: {selected_engine}")
                                logger.debug(f"Engines available: {list(engines.keys())}")
                                
                                def process_with_engine(engine_name, criteria_text):
                                    """Standardized engine processing with consistent output format"""
                                    try:
                                        start = time.time()
                                        logger.info(f"Processing with {engine_name} engine...")
                                        raw_result = engines[engine_name].process_criteria(criteria_text)
                                        elapsed = time.time() - start
                                        
                                        # Standardize output format
                                        standardized = {
                                            'features': raw_result.get('features', {}),
                                            'entities': raw_result.get('entities', []),
                                            'metadata': {
                                                'processing_time': elapsed,
                                                'engine': engine_name,
                                                'biomarkers_count': len(raw_result.get('features', {}).get('biomarkers', [])),
                                                'variants_count': len(raw_result.get('features', {}).get('genomic_variants', []))
                                            }
                                        }
                                        logger.debug(f"{engine_name} result: {json.dumps(standardized, indent=2)}")
                                        return standardized, None
                                    except Exception as e:
                                        logger.error(f"{engine_name} engine error: {str(e)}")
                                        return None, str(e)
            
                                # Process with selected engine only
                                result, error = process_with_engine(selected_engine, criteria)
                                if error:
                                    return {'error': error}
                                return {'single': result}
                            except Exception as e:
                                logger.error(f"Extraction error: {str(e)}\n{traceback.format_exc()}")
                                raise
                            
                        extraction_result = await asyncio.get_event_loop().run_in_executor(None, extraction_task)
                        extraction_cache[criteria] = extraction_result
                        
                        extraction_status.set_text('Extraction complete')
                        ui.notify('mCODE features extracted successfully', type='positive')
                        logger.info(f"Extraction complete for trial {protocol_section.get('identificationModule', {}).get('nctId')}")
                    
                    # Directly display extraction results
                    with trials_container:
                        display_trial_results(protocol_section, extraction_result)
                except Exception as e:
                    def trial_cleanup():
                        processing_label.set_text("Trial processing failed")
                    
                    handle_error(e, "Trial processing error", trial_cleanup)
                    continue
                    
        except Exception as e:
            handle_error(e, "Search operation failed", cleanup)
        finally:
            loading.set_visibility(False)


def display_trial_results(protocol_section, display_data):
    """Display trial results with mCODE matching information"""
    logger.debug(f"Entering display_trial_results with display_data type: {type(display_data)}")
    
    # Validate input data
    if not display_data or not isinstance(display_data, dict):
        logger.error(f"Invalid display_data: {display_data}")
        ui.label('Invalid trial data format').classes('text-red-500')
        return
        
    # Extract features from display_data
    if 'single' in display_data:
        # New format: extraction result is under 'single' key
        features = display_data['single'].get('features', {})
    else:
        # Old format: features are directly in display_data
        features = display_data.get('features', {})
    
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
                
    
    with ui.card().classes('w-full'):
        # Trial header with match strength
        with ui.expansion('Patient Match').classes('w-full'):
            # Match strength indicator
            if display_data and 'features' in display_data and matching_toggle.value:
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
                
                # Get features from display_data
                features = display_data['features']
                
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
                
                # Ensure trial_features has required structure
                if not isinstance(trial_features, dict):
                    trial_features = {}
                if 'cancer_characteristics' not in trial_features:
                    trial_features['cancer_characteristics'] = {}
                
                # Safely handle missing metadata
                genomic_count = features.get('metadata', {}).get('genomic_variants_count', 0)
                biomarkers_count = features.get('metadata', {}).get('biomarkers_count', 0)
                
                match_score, match_details = matcher.calculate_match_score(patient_data, trial_features, return_details=True)
                match_desc = matcher.get_match_description(match_score)
                
                with ui.column().classes('items-center'):
                    ui.linear_progress(match_score/100).classes('w-16 h-2')
                    ui.label(f"{match_score}%").classes('text-xs')
                    ui.label(match_desc).classes('text-xs font-bold')
                    
                    # Show matching details expansion
                    with ui.expansion('Matching Details').classes('w-full mt-2'):
                        with ui.card().classes('w-full p-4 bg-gray-50'):
                            ui.label('mCODE Elements Matched').classes('font-bold mb-2')
                            
                            # Cancer Type
                            with ui.row().classes('items-center'):
                                ui.icon('check_circle' if match_details['cancer_type'] else 'cancel').classes('text-green-500' if match_details['cancer_type'] else 'text-red-500')
                                with ui.column().classes('ml-2'):
                                    ui.label('Cancer Type').classes('font-medium')
                                    ui.label(f"Patient: {patient_data['cancer_type']}").classes('text-xs text-gray-600')
                                    ui.label(f"Trial: {trial_features['cancer_type']}").classes('text-xs text-gray-600')
                            
                            # Biomarkers
                            with ui.row().classes('items-center'):
                                ui.icon('check_circle' if match_details['biomarkers'] else 'cancel').classes('text-green-500' if match_details['biomarkers'] else 'text-red-500')
                                with ui.column().classes('ml-2'):
                                    ui.label('Biomarkers').classes('font-medium')
                                    if match_details['biomarkers']:
                                        matched = [b for b in patient_data['biomarkers'] if b in trial_features['biomarkers']]
                                        ui.label(f"Matched: {', '.join(matched)}").classes('text-xs text-gray-600')
                            
                            # Genomic Variants
                            with ui.row().classes('items-center'):
                                ui.icon('check_circle' if match_details['genomic_variants'] else 'cancel').classes('text-green-500' if match_details['genomic_variants'] else 'text-red-500')
                                with ui.column().classes('ml-2'):
                                    ui.label('Genomic Variants').classes('font-medium')
                                    if match_details['genomic_variants']:
                                        matched = [v for v in patient_data['genomic_variants'] if v in trial_features['genomic_variants']]
                                        ui.label(f"Matched: {', '.join(matched)}").classes('text-xs text-gray-600')
                            
                            # Stage/Grade
                            with ui.row().classes('items-center'):
                                ui.icon('check_circle' if match_details['stage_grade'] else 'cancel').classes('text-green-500' if match_details['stage_grade'] else 'text-red-500')
                                with ui.column().classes('ml-2'):
                                    ui.label('Stage/Grade').classes('font-medium')
                                    if match_details['stage_grade']:
                                        ui.label(f"Patient Stage: {patient_data['cancer_characteristics']['stage']}").classes('text-xs text-gray-600')
                                        ui.label(f"Trial Minimum: {trial_features['cancer_characteristics']['stage']}").classes('text-xs text-gray-600')
            
            # Trial info
            with ui.column().classes('flex-grow'):
                ui.label(protocol_section.get('identificationModule', {}).get('officialTitle')).classes('text-lg')
                ui.label(f"NCT ID: {protocol_section.get('identificationModule', {}).get('nctId')}")
        
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
                

# Reset function
def on_reset():
    trials_container.clear()
    search_input.value = patient_profile['cancer_type']
    ui.notify('Search results cleared', type='info')

# Bind handlers to buttons
search_button.on('click', on_search)
reset_button.on('click', on_reset)

ui.run(title='mCODE Clinical Trials Search', port=8081)