from nicegui import ui
import sys
import os
import asyncio

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clinical_trials_api import ClinicalTrialsAPI
from src.extraction_pipeline import ExtractionPipeline
import time

# Initialize API client
api_client = ClinicalTrialsAPI()

# Initialize extraction pipeline with caching
extraction_cache = {}
pipeline = ExtractionPipeline(use_llm=True, model="deepseek-coder")

# Basic search interface
with ui.column().classes('w-full items-center'):
    ui.label('mCODE Clinical Trials Search').classes('text-2xl')
    
    # Search controls
    with ui.row().classes('w-full justify-center items-center'):
        search_input = ui.input('Search term', value='breast cancer').classes('w-64')
        
        # Results limit control
        with ui.column().classes('w-64'):
            ui.label('Results limit').classes('text-sm')
            limit_slider = ui.slider(min=1, max=20, value=5).classes('w-full')
        
        # Extraction controls
        with ui.column().classes('w-64'):
            ui.label('Extraction').classes('text-sm')
            llm_toggle = ui.toggle([True, False], value=True).props('left-label').bind_value(pipeline, 'use_llm')
            ui.label('Enable LLM extraction')
        
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
            # Perform async search
            def search_task():
                return api_client.search_trials(search_input.value, max_results=int(limit_slider.value))
            
            results = await asyncio.get_event_loop().run_in_executor(None, search_task)
            
            if not results:
                status_label.set_text('No results found')
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
                    elif criteria:
                        extraction_status = ui.label('Extracting genomic features...')
                        
                        # Run extraction in executor to prevent blocking
                        def extraction_task():
                            return pipeline.process_criteria(criteria)
                            
                        extraction_result = await asyncio.get_event_loop().run_in_executor(None, extraction_task)
                        extraction_cache[criteria] = extraction_result
                        
                        extraction_status.set_text('Extraction complete')
                        ui.notify('Genomic features extracted successfully', type='positive')
                    
                    # Display results in main thread
                    # Run display in main thread with explicit container context
                    def display_task():
                        with trials_container:
                            display_trial_results(protocol_section, extraction_result)
                    
                    await asyncio.get_event_loop().run_in_executor(None, display_task)
                except Exception as e:
                    ui.notify(f"Error processing trial: {str(e)}", type='negative')
                    processing_label.set_text(f"Error: {str(e)}")
                    continue
                    
        except Exception as e:
            status_label.set_text(f"Search failed: {str(e)}")
            ui.notify(f"Search error: {str(e)}", type='negative')
        finally:
            loading.set_visibility(False)

def display_trial_results(protocol_section, extraction_result):
    with ui.card().classes('w-full'):
        # Basic trial info
        ui.label(protocol_section.get('identificationModule', {}).get('officialTitle')).classes('text-lg')
        ui.label(f"NCT ID: {protocol_section.get('identificationModule', {}).get('nctId')}")
        
        # Show genomic features if available
        if extraction_result and extraction_result.get('genomic_features'):
            with ui.row().classes('w-full'):
                ui.label("Genomic Features:").classes('font-bold')
                with ui.column():
                    if extraction_result['genomic_features'].get('genomic_variants'):
                        ui.label(f"Variants: {len(extraction_result['genomic_features']['genomic_variants'])}")
                        for variant in extraction_result['genomic_features']['genomic_variants']:
                            ui.label(f"- {variant.get('gene', variant)} {variant.get('variant', '')}")
                    
                    if extraction_result['genomic_features'].get('biomarkers'):
                        ui.label(f"Biomarkers: {len(extraction_result['genomic_features']['biomarkers'])}")
                        for biomarker in extraction_result['genomic_features']['biomarkers']:
                            ui.label(f"- {biomarker.get('name', biomarker)}: {biomarker.get('status', '')}")
        

# Bind handler to button click
search_button.on('click', on_search)

ui.run(title='mCODE Clinical Trials Search')