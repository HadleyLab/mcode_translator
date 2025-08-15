import json
import sys
import os
import asyncio
import logging
import traceback
import colorlog

# Configure colored logging FIRST
logger = colorlog.getLogger()
logger.setLevel(logging.INFO)
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

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clinical_trials_api import ClinicalTrialsAPI
from src.extraction_pipeline import ExtractionPipeline
from src.regex_nlp_engine import RegexNLPEngine
from src.spacy_nlp_engine import SpacyNLPEngine
from src.llm_nlp_engine import LLMNLPEngine
import time

# Initialize API client
api_client = ClinicalTrialsAPI()

# Configure colored logging
print("Initializing colored logging...")  # Debug
logger = colorlog.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove any existing handlers
logger.handlers = []

# Create new handler with color formatting
handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s:%(lineno)d%(reset)s %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
print(f"Formatter created: {formatter}")  # Debug
handler.setFormatter(formatter)
logger.addHandler(handler)

# Prevent propagation to root logger
logger.propagate = False

# Test logging
logger.debug("Debug test message - should be cyan")
logger.info("Info test message - should be green")
logger.warning("Warning test message - should be yellow")
logger.error("Error test message - should be red")
logger.critical("Critical test message - should be bold red")

# Initialize extraction pipelines with caching
extraction_cache = {}
engines = {
    'Regex': ExtractionPipeline(RegexNLPEngine()),
    'SpaCy': ExtractionPipeline(SpacyNLPEngine()),
    'LLM': ExtractionPipeline(LLMNLPEngine())
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

# Update patient profile function
def validate_form():
    """Validate all form fields before submission"""
    valid = True
    if not age_input.validate():
        ui.notify('Please enter a valid age (18-100)', type='negative')
        valid = False
    if not cancer_type_select.validate():
        ui.notify('Please select a cancer type', type='negative')
        valid = False
    if not histology_input.validate():
        ui.notify('Histology is required', type='negative')
        valid = False
    if not grade_select.validate():
        ui.notify('Please select a grade', type='negative')
        valid = False
    return valid

def update_patient_profile():
    """Update patient profile after validation"""
    if not validate_form():
        return
    # Convert UI inputs to structured format
    new_biomarkers = [
        {'name': 'ER', 'status': er_select.value, 'value': er_value_input.value},
        {'name': 'PR', 'status': pr_select.value, 'value': pr_value_input.value},
        {'name': 'HER2', 'status': her2_select.value, 'value': her2_value_input.value},
        {'name': 'Ki67', 'status': ki67_select.value, 'value': ki67_value_input.value}
    ]
    
    new_variants = []
    for variant in variants_input.value.split(';'):
        if variant.strip():
            parts = variant.strip().split(',')
            new_variants.append({
                'gene': parts[0].strip(),
                'variant': parts[1].strip() if len(parts) > 1 else '',
                'significance': parts[2].strip() if len(parts) > 2 else ''
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

# Patient profile UI with expansion sections
with ui.expansion('PATIENT PROFILE', icon='person', value=False).classes('w-full'):
    with ui.card().classes('w-full p-4 sm:p-6 mb-6 bg-white shadow-md rounded-lg hover:shadow-lg transition-shadow'):
        with ui.column().classes('w-full gap-6'):
            # Patient Information Section
            with ui.expansion('PATIENT INFORMATION', icon='person').classes('w-full'):
                with ui.card().classes('w-full p-6 bg-white shadow-sm gap-4'):
                    with ui.grid(columns=1).classes('w-full gap-4').props('sm:columns=2'):
                    # Basic Demographics
                        with ui.column().classes('gap-4'):
                            ui.label('Basic Information').classes('text-lg font-medium')
                            age_input = ui.number(
                                label='Age*',
                                value=patient_profile['age'],
                                min=18,
                                max=100,
                                validation={'Please enter age between 18-100': lambda value: 18 <= value <= 100}
                            ).props('tooltip="Patient age in years (18-100)" required')
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
        
        # Biomarkers Section
        with ui.expansion('BIOMARKERS', icon='science', value=False).classes('w-full'):
            with ui.card().classes('w-full p-6 bg-gray-50 shadow-sm rounded-lg gap-4 hover:bg-white transition-colors'):
                with ui.column().classes('gap-2'):
                    # ER Biomarker
                    with ui.row().classes('items-center gap-2'):
                        ui.label('ER:').classes('font-medium w-16')
                        er_select = ui.select(
                            ['Positive', 'Negative', 'Unknown'],
                            value=next((b['status'] for b in patient_profile['biomarkers'] if b['name'] == 'ER'), 'Unknown'),
                        ).classes('flex-grow')
                        er_value_input = ui.input(
                            value=next((b['value'] for b in patient_profile['biomarkers'] if b['name'] == 'ER'), '')
                        ).classes('w-24')

                    # PR Biomarker
                    with ui.row().classes('items-center gap-2'):
                        ui.label('PR:').classes('font-medium w-16')
                        pr_select = ui.select(
                            ['Positive', 'Negative', 'Unknown'],
                            value=next((b['status'] for b in patient_profile['biomarkers'] if b['name'] == 'PR'), 'Unknown'),
                        ).classes('flex-grow')
                        pr_value_input = ui.input(
                            value=next((b['value'] for b in patient_profile['biomarkers'] if b['name'] == 'PR'), '')
                        ).classes('w-24')

                    # HER2 Biomarker
                    with ui.row().classes('items-center gap-2'):
                        ui.label('HER2:').classes('font-medium w-16')
                        her2_select = ui.select(
                            ['Positive', 'Negative', 'Unknown'],
                            value=next((b['status'] for b in patient_profile['biomarkers'] if b['name'] == 'HER2'), 'Unknown'),
                        ).classes('flex-grow')
                        her2_value_input = ui.input(
                            value=next((b['value'] for b in patient_profile['biomarkers'] if b['name'] == 'HER2'), '')
                        ).classes('w-24')

                    # Ki67 Biomarker
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Ki67:').classes('font-medium w-16')
                        ki67_select = ui.select(
                            ['Positive', 'Negative', 'Unknown'],
                            value=next((b['status'] for b in patient_profile['biomarkers'] if b['name'] == 'Ki67'), 'Unknown'),
                        ).classes('flex-grow')
                        ki67_value_input = ui.input(
                            value=next((b['value'] for b in patient_profile['biomarkers'] if b['name'] == 'Ki67'), '')
                        ).classes('w-24')
        
            # Clinical Data Section
        # Clinical Data Section
        with ui.expansion('CLINICAL DATA', icon='medical_services', value=False).classes('w-full z-10'):
                with ui.card().classes('w-full p-4 sm:p-6 bg-gray-50 shadow-sm rounded-lg gap-4 hover:bg-white transition-colors relative'):
                    # Genomic Variants
                    with ui.column().classes('gap-2 mb-4'):
                        ui.label('Genomic Variants').classes('font-medium')
                        variant_text = []
                        for v in patient_profile['genomic_variants']:
                            variant_text.append(f"{v['gene']}, {v.get('variant', '')}, {v.get('significance', '')}")
                        variants_input = ui.textarea(
                            value='; '.join(variant_text),
                            placeholder='gene,variant,significance (one per line)'
                        ).classes('w-full text-xs')
                
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
            
    # Update Button (outside accordion)
    ui.button('UPDATE PROFILE', icon='save', on_click=update_patient_profile) \
        .classes('mt-6 w-full py-3 sm:py-2 bg-primary text-white hover:bg-primary-dark text-base sm:text-sm')
            
# Basic search interface
with ui.column().classes('w-full items-center'):
    ui.label('mCODE CLINICAL TRIALS SEARCH').classes('text-2xl font-bold mb-6 text-primary')
    
    # Search controls in a card with better organization
    with ui.card().classes('w-full p-4 sm:p-6 bg-white shadow-md rounded-lg hover:shadow-lg transition-shadow'):
        with ui.row().classes('w-full flex-col sm:flex-row items-center gap-4'):
            # Search input with clear label
            search_input = ui.input(
                label='Search Clinical Trials',
                placeholder='Enter cancer type or keywords',
                value=patient_profile['cancer_type']
            ).props('clearable').classes('flex-grow')
            
            # Search actions
            with ui.row().classes('items-center gap-2 mt-4 sm:mt-0'):
                search_button = ui.button('Search', icon='search') \
                    .props('color=primary') \
                    .classes('min-w-[120px]')
                reset_button = ui.button('Reset', icon='refresh') \
                    .props('outline') \
                    .classes('min-w-[120px]')
        
        # Advanced options in expansion
        with ui.expansion('Advanced Options', icon='tune').classes('w-full mt-2'):
            with ui.grid(columns=1).classes('w-full gap-4').props('sm:columns=2 lg:columns=3'):
                # Results limit
                with ui.card().classes('p-4 bg-gray-50 rounded-lg hover:bg-white transition-colors'):
                    ui.label('Results Limit').classes('font-medium')
                    limit_slider = ui.slider(min=1, max=20, value=5) \
                        .props('label-always') \
                        .classes('w-full')
                
                # NLP Engine selection
                with ui.card().classes('p-4 bg-gray-50 rounded-lg hover:bg-white transition-colors'):
                    ui.label('NLP Engine').classes('font-medium')
                    engine_select = ui.radio(
                        ['Regex', 'SpaCy', 'LLM'],
                        value='LLM'
                    ).props('inline')
                
                # Matching options
                with ui.card().classes('p-4 bg-gray-50 rounded-lg hover:bg-white transition-colors'):
                    ui.label('Matching Options').classes('font-medium')
                    with ui.row().classes('items-center gap-2'):
                        matching_toggle = ui.toggle([True, False], value=True) \
                            .props('left-label color=primary')
                        ui.label('Patient Matching')
                    with ui.row().classes('items-center gap-2'):
                        benchmark_toggle = ui.toggle([True, False], value=False) \
                            .props('left-label color=secondary')
                        ui.label('Benchmark Mode')

    # Results display
    trials_container = ui.column().classes('w-full p-4')

# Define search handler
async def on_search():
    trials_container.clear()
    with trials_container:
        # Show loading state
        loading = ui.spinner('dots', size='lg', color='primary')
        status_label = ui.label('Searching clinical trials...')
        
        try:
            ui.notify(f"Searching for: {search_input.value}")
            logger.info(f"Starting search for: {search_input.value}")
            
            # Perform async search
            def search_task():
                logger.info(f"Calling API with search term: '{search_input.value}'")
                results = api_client.search_trials(search_input.value, max_results=int(limit_slider.value))
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
            
            for study in results.get('studies', []):
                protocol_section = study.get('protocolSection', {})
                eligibility_module = protocol_section.get('eligibilityModule', {})
                criteria = eligibility_module.get('eligibilityCriteria', '')
                
                # Show processing status
                processing_label = ui.label(f"Processing trial {protocol_section.get('identificationModule', {}).get('nctId')}...")
                
                try:
                    # Check cache before processing
                    extraction_result = None
                    if criteria in extraction_cache:
                        extraction_result = extraction_cache[criteria]
                        logger.info(f"Using cached extraction for trial {protocol_section.get('identificationModule', {}).get('nctId')}")
                    elif criteria:
                        extraction_status = ui.label('Extracting mCODE features...')
                        ui.notify('Starting mCODE feature extraction...', type='info')
                        logger.info(f"Starting feature extraction for trial {protocol_section.get('identificationModule', {}).get('nctId')}")
                        
                        # Run extraction in executor to prevent blocking
                        def extraction_task():
                            try:
                                logger.info(f"Starting extraction for criteria: {criteria[:100]}...")
                                selected_engine = engine_select.value
                                
                                if benchmark_toggle.value:
                                    # Benchmark all engines
                                    results = {}
                                    for name, pipeline in engines.items():
                                        start = time.time()
                                        results[name] = {
                                            'result': pipeline.process_criteria(criteria),
                                            'time': time.time() - start
                                        }
                                    return {'benchmark': results}
                                else:
                                    # Use selected engine
                                    return {
                                        'single': {
                                            'engine': selected_engine,
                                            'result': engines[selected_engine].process_criteria(criteria)
                                        }
                                    }
                            except Exception as e:
                                logger.error(f"Extraction error: {str(e)}\n{traceback.format_exc()}")
                                raise
                            
                        extraction_result = await asyncio.get_event_loop().run_in_executor(None, extraction_task)
                        extraction_cache[criteria] = extraction_result
                        
                        extraction_status.set_text('Extraction complete')
                        ui.notify('mCODE features extracted successfully', type='positive')
                        logger.info(f"Extraction complete for trial {protocol_section.get('identificationModule', {}).get('nctId')}")
                    
                    # Display results in main thread
                    def display_task():
                        with trials_container:
                            if extraction_result.get('benchmark'):
                                # Show benchmark comparison
                                with ui.expansion('Engine Performance Comparison').classes('w-full'):
                                    # Performance metrics table
                                    with ui.table().classes('w-full'):
                                        columns = [
                                            {'name': 'engine', 'label': 'Engine', 'field': 'engine'},
                                            {'name': 'time', 'label': 'Time (ms)', 'field': 'time'},
                                            {'name': 'entities', 'label': 'Entities Found', 'field': 'entities'}
                                        ]
                                        rows = []
                                        for engine, data in extraction_result['benchmark'].items():
                                            rows.append({
                                                'engine': engine,
                                                'time': round(data['time']*1000, 2),
                                                'entities': len(data['result'].get('entities', []))
                                            })
                                        ui.table(columns=columns, rows=rows)
                                
                                # Show results from primary engine (LLM by default)
                                display_data = {
                                    'features': extraction_result['benchmark']['LLM']['result'].get('features', {}),
                                    'mcode_mappings': extraction_result['benchmark']['LLM']['result'].get('mcode_mappings', {})
                                }
                            else:
                                # Single engine mode
                                display_data = {
                                    'features': extraction_result['single']['result'].get('features', {}),
                                    'mcode_mappings': extraction_result['single']['result'].get('mcode_mappings', {})
                                }
                            display_trial_results(protocol_section, display_data)
                    
                    await asyncio.get_event_loop().run_in_executor(None, display_task)
                except Exception as e:
                    error_msg = f"Error processing trial: {str(e)}"
                    ui.notify(error_msg, type='negative')
                    processing_label.set_text(error_msg)
                    logger.error(f"Trial processing error: {error_msg}\n{traceback.format_exc()}")
                    # Log full exception details for debugging
                    logger.error(f"Full exception: {traceback.format_exc()}")
                    continue
                    
        except Exception as e:
            status_label.set_text(f"Search failed: {str(e)}")
            ui.notify(f"Search error: {str(e)}", type='negative')
            # Log full exception details for debugging
            logger.error(f"Search failed with error: {str(e)}\n{traceback.format_exc()}")
        finally:
            loading.set_visibility(False)


def display_trial_results(protocol_section, extraction_result):
    with ui.card().classes('w-full'):
        # Trial header with match strength
        with ui.row().classes('w-full items-center'):
            # Match strength indicator
            if extraction_result and 'features' in extraction_result and matching_toggle.value:
                from src.matcher import PatientMatcher
                matcher = PatientMatcher()
                
                # Prepare patient biomarkers as dict for matching
                patient_biomarkers = {b['name']: b['status'] for b in patient_profile['biomarkers']}
                
                # Handle both LLM and other engine formats
                features = extraction_result['features']
                if isinstance(features.get('biomarkers'), dict):
                    # Convert LLM format to standard format
                    biomarkers = [{'name': k, 'status': v} for k,v in features['biomarkers'].items()]
                    variants = features.get('genomic_variants', [])
                else:
                    biomarkers = features.get('biomarkers', [])
                    variants = features.get('genomic_variants', [])
                
                trial_features = {
                    'cancer_type': patient_profile['cancer_type'],
                    'biomarkers': {b['name']: b['status'] for b in biomarkers},
                    'genomic_variants': [v['gene'] if isinstance(v, dict) else v for v in variants]
                }
                # Build patient data for matching with fallbacks for missing fields
                patient_data = {
                    'cancer_type': patient_profile.get('cancer_type', ''),
                    'stage': patient_profile.get('stage', ''),
                    'biomarkers': patient_biomarkers,
                    'genomic_variants': [v.get('gene', '') for v in patient_profile.get('genomic_variants', [])],
                    'cancer_characteristics': {
                        'stage': patient_profile.get('stage', ''),
                        'histology': patient_profile.get('histology', ''),
                        'grade': patient_profile.get('grade', ''),
                        'tnm_staging': patient_profile.get('tnm_staging', {'t': '', 'n': '', 'm': ''})
                    }
                }
                
                # Ensure trial_features has required structure
                if not isinstance(trial_features, dict):
                    trial_features = {}
                if 'cancer_characteristics' not in trial_features:
                    trial_features['cancer_characteristics'] = {}
                
                match_score = matcher.calculate_match_score(patient_data, trial_features)
                match_desc = matcher.get_match_description(match_score)
                
                with ui.column().classes('items-center'):
                    ui.linear_progress(match_score/100).classes('w-16 h-2')
                    ui.label(f"{match_score}%").classes('text-xs')
                    ui.label(match_desc).classes('text-xs font-bold')
            
            # Trial info
            with ui.column().classes('flex-grow'):
                ui.label(protocol_section.get('identificationModule', {}).get('officialTitle')).classes('text-lg')
                ui.label(f"NCT ID: {protocol_section.get('identificationModule', {}).get('nctId')}")
        
        # Eligibility criteria
        with ui.expansion('View Eligibility Criteria').classes('w-full'):
            ui.markdown(f"```\n{protocol_section.get('eligibilityModule', {}).get('eligibilityCriteria', 'No criteria available')}\n```")
        
        # mCODE extraction results
        with ui.expansion('mCODE Extraction Results').classes('w-full'):
            if not extraction_result or not extraction_result.get('features'):
                ui.label('No mCODE data available').classes('text-red-500')
            else:
                features = extraction_result['features']
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
                                        with ui.card().classes('p-4 bg-gray-50 rounded-lg hover:bg-white transition-colors'):
                                            ui.label(key.replace('_', ' ').title()).classes('font-medium text-sm')
                                            ui.label(str(value) if value else 'Not specified')
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
                                        with ui.card().classes('p-4 bg-gray-50 rounded-lg hover:bg-white transition-colors'):
                                            ui.label(key.replace('_', ' ').title()).classes('font-medium text-sm')
                                            if isinstance(value, dict):
                                                for k, v in value.items():
                                                    ui.label(f"{k}: {v}").classes('text-xs')
                                            else:
                                                ui.label(str(value) if value else 'Not specified')
                        else:
                            ui.label('No cancer characteristics data').classes('text-gray-500')
                    
                    # Genomic Variants
                    with ui.tab_panel(genomics_tab):
                        variants = features.get('genomic_variants', [])
                        if variants:
                            with ui.card().classes('w-full p-4'):
                                ui.label('GENOMIC VARIANTS').classes('text-xl font-bold mb-4 text-primary')
                                with ui.grid(columns=2).classes('w-full gap-4'):
                                    for variant in variants:
                                        with ui.card().classes('p-4 bg-gray-50 rounded-lg hover:bg-white transition-colors'):
                                            with ui.column().classes('gap-1'):
                                                ui.label(f"Gene: {variant.get('gene', 'Unknown')}").classes('font-bold')
                                                ui.label(f"Variant: {variant.get('variant', 'N/A')}").classes('text-sm')
                                                ui.label(f"Significance: {variant.get('significance', 'N/A')}").classes('text-sm text-blue-600')
                        else:
                            ui.label('No genomic variants found').classes('text-gray-500')
                    
                    # Biomarkers
                    with ui.tab_panel(biomarkers_tab):
                        biomarkers = features.get('biomarkers', [])
                        if biomarkers:
                            with ui.card().classes('w-full p-4'):
                                ui.label('BIOMARKERS').classes('text-xl font-bold mb-4 text-primary')
                                with ui.grid(columns=2).classes('w-full gap-4'):
                                    for biomarker in biomarkers:
                                        with ui.card().classes('p-4 bg-gray-50 rounded-lg hover:bg-white transition-colors'):
                                            with ui.column().classes('gap-1'):
                                                ui.label(biomarker.get('name', 'Unknown')).classes('font-bold')
                                                with ui.row().classes('items-center gap-2'):
                                                    ui.label('Status:').classes('text-sm font-medium')
                                                    ui.label(biomarker.get('status', 'N/A')).classes('text-sm')
                                                with ui.row().classes('items-center gap-2'):
                                                    ui.label('Value:').classes('text-sm font-medium')
                                                    ui.label(biomarker.get('value', 'N/A')).classes('text-sm')
                        else:
                            ui.label('No biomarkers found').classes('text-gray-500')
                    
                    # Treatment History
                    with ui.tab_panel(treatment_tab):
                        treatment = features.get('treatment_history', {})
                        if treatment:
                            with ui.card().classes('w-full p-4'):
                                ui.label('TREATMENT HISTORY').classes('text-xl font-bold mb-4 text-primary')
                                with ui.grid(columns=2).classes('w-full gap-4'):
                                    for key, value in treatment.items():
                                        with ui.card().classes('p-4 bg-gray-50 rounded-lg hover:bg-white transition-colors'):
                                            ui.label(key.replace('_', ' ').title()).classes('font-medium text-sm')
                                            ui.label(str(value) if value else 'Not specified')
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
                                        with ui.card().classes('p-2 bg-gray-50'):
                                            ui.label(key.replace('_', ' ').title()).classes('font-medium text-sm')
                                            ui.label(str(value) if value else 'Not specified')
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

ui.run(title='mCODE Clinical Trials Search')