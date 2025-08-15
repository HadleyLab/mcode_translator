import json
from nicegui import ui
import sys
import os
import asyncio
import logging
import traceback

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

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Initialize extraction pipelines with caching
extraction_cache = {}
engines = {
    'Regex': ExtractionPipeline(RegexNLPEngine()),
    'SpaCy': ExtractionPipeline(SpacyNLPEngine()),
    'LLM': ExtractionPipeline(LLMNLPEngine())
}

# Patient profile section (aligned with test_llm_mcode_extraction.py)
patient_profile = {
    'cancer_type': 'breast cancer',
    'age': 55,
    'gender': 'female',
    'stage': 'II',
    'biomarkers': [
        {'name': 'ER', 'status': 'Positive'},
        {'name': 'PR', 'status': 'Positive'},
        {'name': 'HER2', 'status': 'Negative'}
    ],
    'genomic_variants': [
        {'gene': 'PIK3CA'},
        {'gene': 'TP53'}
    ],
    'cancer_characteristics': {
        'stage': 'II'
    }
}

# Update patient profile function
def update_patient_profile():
    # Convert UI inputs to test file's structure
    new_biomarkers = [
        {'name': 'ER', 'status': er_select.value},
        {'name': 'PR', 'status': pr_select.value},
        {'name': 'HER2', 'status': her2_select.value}
    ]
    
    new_variants = [{'gene': v.strip()} for v in variants_input.value.split(',') if v.strip()]
    
    patient_profile.update({
        'cancer_type': cancer_type_select.value,
        'age': int(age_input.value),
        'gender': gender_select.value,
        'stage': stage_select.value,
        'biomarkers': new_biomarkers,
        'genomic_variants': new_variants,
        'cancer_characteristics': {
            'stage': stage_select.value
        }
    })
    ui.notify('Patient profile updated!')

# Patient profile UI
with ui.expansion('Patient Profile', icon='person').classes('w-full mb-4'):
    with ui.row().classes('w-full items-start'):
        # Demographics
        with ui.column().classes('w-1/3 pr-4'):
            ui.label('Demographics').classes('text-lg font-bold')
            cancer_type_select = ui.select(
                ['breast cancer', 'lung cancer', 'colorectal cancer'],
                value=patient_profile['cancer_type'],
                label='Cancer Type'
            )
            age_input = ui.number(
                label='Age',
                value=patient_profile['age'],
                min=18,
                max=100
            )
            gender_select = ui.select(
                ['female', 'male', 'other', 'unknown'],
                value=patient_profile['gender'],
                label='Gender'
            )
            stage_select = ui.select(
                ['I', 'II', 'III', 'IV'],
                value=patient_profile['stage'],
                label='Stage'
            )
        
        # Biomarkers
        with ui.column().classes('w-1/3 px-4'):
            ui.label('Biomarkers').classes('text-lg font-bold')
            # Find biomarker status by name
            er_status = next((b['status'] for b in patient_profile['biomarkers'] if b['name'] == 'ER'), 'Unknown')
            pr_status = next((b['status'] for b in patient_profile['biomarkers'] if b['name'] == 'PR'), 'Unknown')
            her2_status = next((b['status'] for b in patient_profile['biomarkers'] if b['name'] == 'HER2'), 'Unknown')
            
            er_select = ui.select(
                ['Positive', 'Negative', 'Unknown'],
                value=er_status,
                label='ER Status'
            )
            pr_select = ui.select(
                ['Positive', 'Negative', 'Unknown'],
                value=pr_status,
                label='PR Status'
            )
            her2_select = ui.select(
                ['Positive', 'Negative', 'Unknown'],
                value=her2_status,
                label='HER2 Status'
            )
        
        # Genomic Variants
        with ui.column().classes('w-1/3 pl-4'):
            ui.label('Genomic Variants').classes('text-lg font-bold')
            # Extract gene names from variants list
            variant_genes = [v['gene'] for v in patient_profile['genomic_variants']]
            variants_input = ui.textarea(
                label='Variants (comma separated)',
                value=', '.join(variant_genes)
            )
            ui.button('Update Profile', icon='save', on_click=update_patient_profile)
            
# Basic search interface
with ui.column().classes('w-full items-center'):
    ui.label('mCODE Clinical Trials Search').classes('text-2xl')
    
    # Search controls
    with ui.row().classes('w-full justify-center items-center'):
        search_input = ui.input('Search term', value=patient_profile['cancer_type']).classes('w-64')
        
        # Results limit control
        with ui.column().classes('w-64'):
            ui.label('Results limit').classes('text-sm')
            limit_slider = ui.slider(min=1, max=20, value=5).classes('w-full')
        
        # Engine selection
        with ui.column().classes('w-64'):
            ui.label('NLP Engine').classes('text-sm')
            engine_select = ui.radio(
                ['Regex', 'SpaCy', 'LLM'],
                value='LLM'
            ).props('inline')
            ui.label('Benchmark Mode')
            benchmark_toggle = ui.toggle([True, False], value=False)
        
        # Matching toggle
        with ui.column().classes('w-64'):
            ui.label('Patient Matching').classes('text-sm')
            matching_toggle = ui.toggle([True, False], value=True).props('left-label')
            ui.label('Enable patient matching')
        
        search_button = ui.button('Search', icon='search').classes('w-32')

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
                match_score = matcher.calculate_match_score({
                    'cancer_type': patient_profile['cancer_type'],
                    'stage': patient_profile['stage'],
                    'biomarkers': patient_biomarkers,
                    'genomic_variants': [v['gene'] for v in patient_profile['genomic_variants']],
                    'cancer_characteristics': patient_profile['cancer_characteristics']
                }, trial_features)
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
                            ui.code(json.dumps(demographics, indent=2)).classes('w-full')
                        else:
                            ui.label('No demographics data')
                    
                    # Cancer Condition
                    with ui.tab_panel(cancer_tab):
                        cancer_data = features.get('cancer_characteristics', {})
                        if cancer_data:
                            ui.code(json.dumps(cancer_data, indent=2)).classes('w-full')
                        else:
                            ui.label('No cancer characteristics data')
                    
                    # Genomic Variants
                    with ui.tab_panel(genomics_tab):
                        variants = features.get('genomic_variants', [])
                        if variants:
                            with ui.grid(columns=2).classes('w-full'):
                                for variant in variants:
                                    with ui.card().classes('p-2'):
                                        ui.label(f"Gene: {variant.get('gene', 'Unknown')}").classes('font-bold')
                                        ui.label(f"Variant: {variant.get('variant', '')}")
                                        ui.label(f"Significance: {variant.get('significance', '')}")
                        else:
                            ui.label('No genomic variants found')
                    
                    # Biomarkers
                    with ui.tab_panel(biomarkers_tab):
                        biomarkers = features.get('biomarkers', [])
                        if biomarkers:
                            with ui.grid(columns=2).classes('w-full'):
                                for biomarker in biomarkers:
                                    with ui.card().classes('p-2'):
                                        ui.label(f"Name: {biomarker.get('name', 'Unknown')}").classes('font-bold')
                                        ui.label(f"Status: {biomarker.get('status', '')}")
                                        ui.label(f"Value: {biomarker.get('value', '')}")
                        else:
                            ui.label('No biomarkers found')
                    
                    # Treatment History
                    with ui.tab_panel(treatment_tab):
                        treatment = features.get('treatment_history', {})
                        if treatment:
                            ui.code(json.dumps(treatment, indent=2)).classes('w-full')
                        else:
                            ui.label('No treatment history data')
                    
                    # Performance Status
                    with ui.tab_panel(performance_tab):
                        performance = features.get('performance_status', {})
                        if performance:
                            ui.code(json.dumps(performance, indent=2)).classes('w-full')
                        else:
                            ui.label('No performance status data')
# Bind handler to button click
search_button.on('click', on_search)

ui.run(title='mCODE Clinical Trials Search')