import asyncio
import json
from typing import Dict, List, Optional
from nicegui import ui, run
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("✅ Loaded environment variables from .env file")
except ImportError:
    logger.warning("⚠️  python-dotenv not installed, using system environment variables")

from src.utils.logging_config import get_logger, Loggable, setup_logging
import logging

# Setup logging at the start
setup_logging(logging.INFO)
logger = get_logger(__name__)

from src.pipeline.fetcher import search_trials, get_full_study
from src.pipeline.strict_dynamic_extraction_pipeline import StrictDynamicExtractionPipeline

# Test logging at startup
logger.info("PatientMatcherApp starting up...")
logger.debug("Debug logging enabled")

class PatientMatcherApp(Loggable):
    def __init__(self):
        super().__init__()
        self.current_trial = None
        self.search_results = []
        self.loading = False
        self.dynamic_pipeline = StrictDynamicExtractionPipeline()
        
        # Pagination variables
        self.current_page = 1
        self.page_token = None
        self.total_pages = 1
        self.total_studies = 0
        self.page_tokens = {}  # Store page tokens for navigation
        self.current_search_term = None
        
        # Sample patient profile (would be dynamic in real app)
        self.patient_profile = {
            'cancer_type': 'breast cancer',
            'stage': 'II',
            'biomarkers': [
                {'name': 'ER', 'status': 'Positive'},
                {'name': 'PR', 'status': 'Positive'},
                {'name': 'HER2', 'status': 'Negative'}
            ],
            'genomic_variants': [
                {'gene': 'BRCA1', 'variant': 'c.68_69delAG'}
            ]
        }
        
        self.setup_ui()

    def setup_ui(self):
        """Setup the main application UI using NiceGUI grid system"""
        with ui.grid(columns=12).classes('w-full h-screen gap-0'):
            # Left panel - Patient profile (3 columns)
            with ui.column().classes('col-span-3 p-4 border-r h-full flex flex-col'):
                self.patient_panel_container = ui.column().classes('w-full')
                self.setup_patient_panel()
            
            # Middle panel - Search and results list (4 columns)
            with ui.column().classes('col-span-4 p-4 border-r h-full flex flex-col'):
                self.setup_search_panel()
                self.results_container = ui.column().classes('w-full flex-1 overflow-y-auto')
                self.setup_pagination()
            
            # Right panel - Detailed view (5 columns)
            self.details_panel = ui.column().classes('col-span-5 p-4 h-full flex flex-col overflow-y-auto')

    def setup_search_panel(self):
        """Setup search controls"""
        with ui.column().classes('w-full gap-4'):
            ui.label('Patient-Trial Matcher').classes('text-2xl font-bold')
            
            # Search inputs grid
            with ui.grid(columns=2).classes('w-full gap-4'):
                # Primary search input
                self.search_input = ui.input(
                    placeholder='Search trials by cancer type or biomarker',
                    value=self.patient_profile['cancer_type']
                ).classes('w-full')
                
                # Additional search fields
                self.date_filter = ui.select(
                    options=['All Dates', 'Last 30 days', 'Last 6 months', 'Last year', 'Last 5 years'],
                    value='All Dates',
                    label='Date Range'
                ).classes('w-full')
            
            # Advanced search options
            with ui.expansion('Advanced Search', icon='tune').classes('w-full'):
                with ui.grid(columns=2).classes('w-full gap-4'):
                    self.status_filter = ui.select(
                        options=['All Statuses', 'Recruiting', 'Active, not recruiting', 'Completed', 'Terminated'],
                        value='All Statuses',
                        label='Trial Status'
                    ).classes('w-full')
                    
                    self.phase_filter = ui.select(
                        options=['All Phases', 'Phase 1', 'Phase 2', 'Phase 3', 'Phase 4'],
                        value='All Phases',
                        label='Trial Phase'
                    ).classes('w-full')
            
            # Search button
            with ui.row().classes('w-full justify-between'):
                ui.button('Search', icon='search', on_click=self.on_search)\
                    .props('color=primary').classes('w-32')
                ui.button('Reset', icon='refresh', on_click=self.reset_search)\
                    .props('outline').classes('w-32')
            
            # Status indicator
            self.status_label = ui.label('Ready').classes('text-sm text-gray-600')

    async def on_search(self):
        """Handle search operation"""
        self.loading = True
        self.status_label.text = 'Searching...'
        self.results_container.clear()
        
        try:
            # Perform async search with pagination
            search_term = self.search_input.value
            self.current_search_term = search_term
            
            self.logger.info(f"Starting search for term: {search_term}")
            
            # Calculate total studies and pages if this is a new search
            if self.page_token is None and self.current_page == 1:
                # Get total count and pages
                self.logger.debug("Calculating total studies for new search")
                stats = await run.io_bound(
                    self._calculate_total_studies,
                    search_term
                )
                self.total_studies = stats['total_studies']
                self.total_pages = stats['total_pages']
                self.logger.info(f"Total studies: {self.total_studies}, Total pages: {self.total_pages}")
            
            self.logger.debug(f"Fetching search results - page_token: {self.page_token}")
            results = await run.io_bound(
                search_trials,
                search_term,
                fields=None,  # Use default fields from fetcher.py
                max_results=10,
                page_token=self.page_token,
                use_cache=True
            )
            
            # Handle both old and new API response formats
            self.search_results = []
            studies = results.get('StudyFields', results.get('studies', []))
            
            self.logger.info(f"Received {len(studies)} studies from API")
            
            # Update pagination info
            if 'nextPageToken' in results:
                next_page_token = results['nextPageToken']
                self.page_token = next_page_token
                # Store the token for this page
                self.page_tokens[self.current_page + 1] = next_page_token
                self.logger.debug(f"Next page token stored: {next_page_token[:10]}...")
            else:
                self.page_token = None
                self.logger.debug("No more pages available")
            
            # Update page label and button states
            self._update_pagination_ui()
            
            for study in studies:
                if isinstance(study, dict):
                    # New format (studies array) - forward-facing robust implementation
                    if 'protocolSection' in study:
                        ident = study.get('protocolSection', {}).get('identificationModule', {})
                        status = study.get('protocolSection', {}).get('statusModule', {})
                        conditions = study.get('protocolSection', {}).get('conditionsModule', {}).get('conditions', [])
                        design = study.get('protocolSection', {}).get('designModule', {})
                        description = study.get('protocolSection', {}).get('descriptionModule', {})
                        
                        # Extract dates from structured date objects
                        start_date_struct = status.get('startDateStruct', {})
                        completion_date_struct = status.get('completionDateStruct', {})
                        
                        start_date = start_date_struct.get('date', '')
                        completion_date = completion_date_struct.get('date', '')
                        
                        # Extract phase information from design module
                        phase_info = design.get('phases', [])
                        phase = phase_info[0] if phase_info and isinstance(phase_info, list) else design.get('phase', 'Not specified')
                        
                        nct_id = ident.get('nctId', 'N/A')
                        title = ident.get('briefTitle', 'No title')
                        self.search_results.append({
                            'NCTId': nct_id,
                            'BriefTitle': title,
                            'OfficialTitle': ident.get('officialTitle', ''),
                            'Condition': conditions,
                            'OverallStatus': status.get('overallStatus', 'Unknown'),
                            'Phase': phase,
                            'StudyType': design.get('studyType', 'Not specified'),
                            'BriefSummary': description.get('briefSummary', ''),
                            'DetailedDescription': description.get('detailedDescription', ''),
                            'StartDate': start_date,
                            'CompletionDate': completion_date,
                            'StartDateType': start_date_struct.get('type', ''),
                            'CompletionDateType': completion_date_struct.get('type', '')
                        })
                        self.logger.debug(f"Added study: {nct_id} - {title}")
                    # Old format (StudyFields) - maintain for backward compatibility
                    elif 'NCTId' in study:
                        conditions = study.get('Condition', [''])
                        # Convert to list if it's a string from old format
                        if isinstance(conditions, str):
                            conditions = [conditions]
                        elif isinstance(conditions, list) and len(conditions) > 0 and isinstance(conditions[0], str):
                            conditions = conditions
                        else:
                            conditions = ['Unknown']
                            
                        nct_id = study.get('NCTId', [''])[0]
                        title = study.get('BriefTitle', [''])[0]
                        self.search_results.append({
                            'NCTId': nct_id,
                            'BriefTitle': title,
                            'OfficialTitle': study.get('OfficialTitle', [''])[0] if study.get('OfficialTitle') else '',
                            'Condition': conditions,
                            'OverallStatus': study.get('OverallStatus', [''])[0],
                            'Phase': study.get('Phase', ['Not specified'])[0],
                            'StudyType': study.get('StudyType', ['Not specified'])[0],
                            'BriefSummary': study.get('BriefSummary', [''])[0],
                            'StartDate': study.get('StartDate', [''])[0] if study.get('StartDate') else '',
                            'CompletionDate': study.get('CompletionDate', [''])[0] if study.get('CompletionDate') else '',
                            'StartDateType': '',
                            'CompletionDateType': ''
                        })
                        self.logger.debug(f"Added study: {nct_id} - {title}")
            
            self.logger.info(f"Processed {len(self.search_results)} studies for display")
            self.display_search_results()
            self.status_label.text = f'Found {len(self.search_results)} trials'
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}", exc_info=True)
            self.status_label.text = f'Error: {str(e)}'
            ui.notify(f'Search failed: {str(e)}', type='negative')
        finally:
            self.loading = False

    def display_search_results(self):
        """Display search results in left panel"""
        for trial in self.search_results:
            with self.results_container:
                with ui.card().classes('w-full p-4 mb-3 cursor-pointer hover:bg-gray-50 hover:shadow-md transition-all') as card:
                    # Trial title and ID
                    ui.label(trial.get('BriefTitle', 'No title')).classes('font-bold text-lg')
                    
                    # Official title if available (collapsed by default)
                    if trial.get('OfficialTitle'):
                        with ui.expansion('View Official Title', icon='title').classes('w-full text-sm'):
                            ui.label(trial['OfficialTitle']).classes('text-gray-600 italic')
                    
                    # Trial metadata badges
                    with ui.row().classes('flex-wrap gap-2 mt-3'):
                        # Status badge
                        status = trial.get('OverallStatus', 'Unknown')
                        status_color = 'green' if status.lower() in ['recruiting', 'active'] else 'orange' if status.lower() in ['completed'] else 'red'
                        ui.badge(status).props(f'color={status_color}')
                        
                        # Phase badge
                        phase = trial.get('Phase', 'Not specified')
                        if phase != 'Not specified':
                            ui.badge(f"Phase {phase}").props('color=blue outline')
                        
                        # Study type badge
                        study_type = trial.get('StudyType', 'Not specified')
                        if study_type != 'Not specified':
                            ui.badge(study_type).props('color=purple outline')
                    
                    # Condition badges
                    if 'Condition' in trial and trial['Condition']:
                        conditions = trial['Condition']
                        with ui.row().classes('flex-wrap gap-1 mt-3'):
                            for condition in conditions[:3]:  # Show first 3 conditions
                                ui.badge(condition.strip()).props('color=blue outline')
                            if len(conditions) > 3:
                                ui.badge(f"+{len(conditions) - 3} more").props('color=grey outline')
                    
                    # Trial dates with type indicators
                    with ui.row().classes('items-center justify-between mt-3 text-xs text-gray-500'):
                        start_date = trial.get('StartDate', '')
                        completion_date = trial.get('CompletionDate', '')
                        start_type = trial.get('StartDateType', '')
                        completion_type = trial.get('CompletionDateType', '')
                        
                        if start_date:
                            date_text = f"Start: {start_date}"
                            if start_type:
                                date_text += f" ({start_type.lower()})"
                            ui.label(date_text)
                        if completion_date:
                            date_text = f"End: {completion_date}"
                            if completion_type:
                                date_text += f" ({completion_type.lower()})"
                            ui.label(date_text)
                    
                    # Brief summary (truncated)
                    brief_summary = trial.get('BriefSummary', '')
                    if brief_summary:
                        truncated_summary = brief_summary[:120] + '...' if len(brief_summary) > 120 else brief_summary
                        ui.label(truncated_summary).classes('text-sm text-gray-600 mt-3 line-clamp-2')
                    
                    # NCT ID at bottom
                    with ui.row().classes('items-center justify-between mt-3'):
                        ui.label(trial.get('NCTId', 'N/A')).classes('text-xs text-gray-400 font-mono')
                        ui.icon('arrow_forward').classes('text-gray-400')
                    
                    # Click handler to show details
                    card.on('click', lambda t=trial: self.show_trial_details(t))

    async def show_trial_details(self, trial):
        """Show detailed trial view with matching analysis"""
        self.current_trial = trial
        self.details_panel.clear()
        
        nct_id = trial.get('NCTId', 'Unknown')
        self.logger.info(f"Loading details for trial: {nct_id}")
        
        with self.details_panel:
            # Loading state
            loading = ui.spinner('dots', size='lg', color='primary')
            status = ui.label('Loading trial details...')
            
            try:
                # Fetch full trial details
                self.logger.debug(f"Fetching full study details for {nct_id}")
                full_study = await run.io_bound(
                    get_full_study,
                    nct_id
                )
                
                if not full_study:
                    self.logger.warning(f"No trial details found for {nct_id}")
                    raise Exception('No trial details found')
                
                self.logger.debug(f"Processing trial through dynamic extraction pipeline for {nct_id}")
                # Process trial through dynamic extraction pipeline
                pipeline_result = await run.io_bound(
                    self.dynamic_pipeline.process_clinical_trial,
                    full_study
                )
                
                if pipeline_result is None:
                    self.logger.warning(f"Pipeline returned None for trial {nct_id}")
                    raise Exception('Pipeline processing failed - no results returned')
                
                self.logger.info(f"Pipeline processing complete for {nct_id}. "
                                f"Entities: {len(pipeline_result.extracted_entities)}, "
                                f"Mappings: {len(pipeline_result.mcode_mappings)}, "
                                f"Source References: {len(pipeline_result.source_references)}")

                # Extract results from pipeline with source tracking
                mapped_mcode = pipeline_result.mcode_mappings
                nlp_result = type('NLPResult', (), {
                    'entities': pipeline_result.extracted_entities,
                    'error': None
                })()
                source_references = pipeline_result.source_references

                # Add source tracking to mapped mCODE elements
                nct_id_from_trial = trial.get('protocolSection', {}).get('identificationModule', {}).get('nctId', nct_id)
                for element in mapped_mcode:
                    element['source'] = nct_id_from_trial

                # Clear loading state
                loading.set_visibility(False)
                status.set_visibility(False)

                # Update entities with mapping information for visualization
                if nlp_result and hasattr(nlp_result, 'entities') and nlp_result.entities and mapped_mcode:
                    self.logger.debug(f"Updating entities with mapping information for {nct_id}")
                    self._update_entities_with_mapping(nlp_result.entities, mapped_mcode, source_references)

                # Display detailed view with NLP results and source provenance
                self.logger.debug(f"Displaying trial analysis for {nct_id}")
                await self.display_trial_analysis(full_study, mapped_mcode, nlp_result, source_references)
                
            except Exception as e:
                self.logger.error(f"Failed to load trial details for {nct_id}: {str(e)}", exc_info=True)
                loading.set_visibility(False)
                status.text = f'Error loading details: {str(e)}'
                ui.notify(f'Failed to load trial details: {str(e)}', type='negative')

    async def display_trial_analysis(self, trial, mapped_mcode, nlp_result=None, source_references=None):
        """Display trial analysis with enhanced mapping visualization and source provenance
        
        Args:
            trial: Dictionary containing trial details from ClinicalTrials.gov API
            mapped_mcode: List of mCODE elements mapped from trial criteria
            nlp_result: NLP processing result containing raw extracted features
            source_references: List of SourceReference objects for provenance tracking
            
        Data Binding Notes:
            - Trial details are bound directly from API response
            - mCODE elements are grouped by type for display
            - Shows many-to-one mapping relationships between NLP entities and mCODE elements
            - Confidence scores are color-coded (green >70%, orange >50%, red <=50%)
            - Biomarker matching is calculated against patient profile
            - Source provenance is shown with comprehensive tooltips
        """
        with ui.column().classes('w-full gap-4'):
            # Trial header with bound data
            with ui.card().classes('w-full p-4 bg-blue-50'):
                trial_title = trial.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle', 'No title')
                ui.label(trial_title).classes('text-xl font-bold')
                with ui.row().classes('items-center justify-between'):
                    nct_id = trial.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'N/A')
                    ui.label(nct_id).classes('text-sm text-gray-600')
                    status = trial.get('protocolSection', {}).get('statusModule', {}).get('overallStatus', 'Unknown')
                    ui.badge(status).props('color=green')
            
            # Matching visualization
            with ui.expansion('Patient Match Analysis', icon='person', value=True).classes('w-full'):
                with ui.card().classes('w-full p-4'):
                    ui.label('Matching Score').classes('text-lg font-bold mb-2')
                    
                    # Calculate actual match score based on mapped mCODE elements
                    total_biomarkers = len(self.patient_profile['biomarkers'])
                    matched_biomarkers = 0
                    biomarker_details = []
                    
                    for mapped in mapped_mcode:
                        if mapped.get('element_type') == 'Biomarker':
                            patient_bm = next((b for b in self.patient_profile['biomarkers']
                                             if b['name'].lower() == mapped['element_name'].lower()), None)
                            if patient_bm and patient_bm['status'].lower() == mapped.get('value', '').lower():
                                matched_biomarkers += 1
                                biomarker_details.append(f"{mapped['element_name']}: {mapped['value']} (✓)")
                            else:
                                biomarker_details.append(f"{mapped['element_name']}: {mapped['value']} (✗)")
                    
                    match_score = matched_biomarkers / total_biomarkers if total_biomarkers > 0 else 0
                    ui.linear_progress(match_score).classes('w-full h-6')
                    ui.label(f'{int(match_score*100)}% match').classes('text-lg font-bold text-center')
                    
                    with ui.grid(columns=2).classes('w-full gap-4 mt-4'):
                        # Cancer type match
                        with ui.card().classes('p-3 bg-green-50'):
                            ui.label('Cancer Type').classes('font-bold')
                            ui.label(f"Patient: {self.patient_profile['cancer_type']}")
                            ui.label(f"Trial: {trial.get('protocolSection', {}).get('conditionsModule', {}).get('conditions', [''])[0]}")
                        
                        # Biomarkers match
                        with ui.card().classes('p-3 bg-blue-50'):
                            ui.label('Biomarkers').classes('font-bold')
                            ui.label(f"{matched_biomarkers}/{total_biomarkers} biomarkers match").classes('text-sm mb-2')
                            for detail in biomarker_details:
                                ui.label(detail).classes('text-xs')
            
            # NLP to mCODE Mapping Visualization with Many-to-One Relationships
            with ui.expansion('NLP to mCODE Mapping', icon='compare_arrows', value=True).classes('w-full'):
                
                # Summary statistics with mapping relationships
                with ui.row().classes('w-full justify-between items-center mb-4'):
                    ui.label('Extraction & Mapping Summary').classes('text-lg font-bold')
                    with ui.row().classes('gap-4'):
                        if nlp_result and hasattr(nlp_result, 'entities'):
                            ui.badge(f"NLP: {len(nlp_result.entities)}").props('color=blue')
                        if mapped_mcode:
                            ui.badge(f"mCODE: {len(mapped_mcode)}").props('color=green')
                        if source_references:
                            ui.badge(f"Sources: {len(source_references)}").props('color=purple')
                
                # Show many-to-one mapping relationships
                if mapped_mcode and nlp_result and hasattr(nlp_result, 'entities') and nlp_result.entities:
                    # Create mapping visualization showing relationships
                    self._display_mapping_relationships(nlp_result.entities, mapped_mcode, source_references)
                else:
                    # Fallback to simple badge display if no relationships can be established
                    with ui.card().classes('w-full p-4 mb-4'):
                        ui.label('NLP Entities').classes('text-md font-bold mb-3 text-blue-600')
                        
                        if nlp_result and not nlp_result.error and hasattr(nlp_result, 'entities') and nlp_result.entities:
                            with ui.row().classes('flex-wrap gap-2'):
                                for entity in nlp_result.entities:
                                    entity_text = entity.get('text', 'Unknown')
                                    entity_type = entity.get('type', 'Unknown')
                                    confidence = entity.get('confidence', 0.8)
                                    
                                    # Create badge with tooltip showing source provenance
                                    badge = ui.badge(f"{entity_text}").props(
                                        f'color={"green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"}'
                                    )
                                    
                                    # Tooltip with comprehensive source information
                                    tooltip_content = self._create_entity_tooltip(entity, entity_type, source_references)
                                    badge.tooltip(tooltip_content)
                        else:
                            ui.label('No NLP entities extracted').classes('text-gray-500 text-center')
                    
                    # mCODE Mappings Badges
                    with ui.card().classes('w-full p-4'):
                        ui.label('mCODE Mappings').classes('text-md font-bold mb-3 text-green-600')
                        
                        if mapped_mcode:
                            with ui.row().classes('flex-wrap gap-2'):
                                for element in mapped_mcode:
                                    element_name = element.get('element_name', 'Unknown')
                                    element_type = element.get('element_type', 'Unknown')
                                    confidence = element.get('confidence', 0.8)
                                    
                                    # Create badge with tooltip showing source provenance
                                    badge = ui.badge(f"{element_name}").props(
                                        f'color={"green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"}'
                                    )
                                    
                                    # Tooltip with comprehensive source information
                                    tooltip_content = self._create_mcode_tooltip(element, element_type, source_references)
                                    badge.tooltip(tooltip_content)
                        else:
                            ui.label('No mCODE elements were mapped').classes('text-gray-500 text-center')
            
            # Raw eligibility criteria
            with ui.expansion('Eligibility Criteria', icon='list').classes('w-full'):
                criteria = trial.get('protocolSection', {}).get('eligibilityModule', {}).get('eligibilityCriteria', 'No criteria available')
                ui.markdown(f"```\n{criteria}\n```").classes('text-xs')

    def reset_search(self):
        """Reset search results and pagination state"""
        self.logger.info("Resetting search results and pagination state")
        self.search_results = []
        self.results_container.clear()
        self.details_panel.clear()
        self.status_label.text = 'Ready'
        self.current_page = 1
        self.page_token = None
        self.total_studies = 0
        self.total_pages = 1
        self.page_tokens = {}
        self.current_search_term = None
        
        # Reset pagination UI
        self._update_pagination_ui()
        self.logger.debug("Search reset completed")

    def setup_patient_panel(self):
        """Setup patient profile visualization and editing"""
        with self.patient_panel_container:
            with ui.column().classes('w-full gap-4'):
                ui.label('Patient Profile').classes('text-2xl font-bold')
                
                # Cancer type
                with ui.card().classes('w-full p-4 bg-blue-50'):
                    ui.label('Cancer Type').classes('font-bold')
                    self.cancer_type_input = ui.input(
                        label='Cancer Type',
                        value=self.patient_profile['cancer_type'],
                        on_change=lambda e: self.update_patient_profile('cancer_type', e.value)
                    ).classes('w-full')
                
                # Stage
                with ui.card().classes('w-full p-4 bg-blue-50'):
                    ui.label('Stage').classes('font-bold')
                    self.stage_input = ui.input(
                        label='Stage',
                        value=self.patient_profile['stage'],
                        on_change=lambda e: self.update_patient_profile('stage', e.value)
                    ).classes('w-full')
                
                # Biomarkers
                with ui.expansion('Biomarkers', icon='science', value=True).classes('w-full'):
                    with ui.card().classes('w-full p-4'):
                        # Add button to add new biomarker
                        ui.button('Add Biomarker', icon='add', on_click=self.add_biomarker).classes('mb-2')
                        
                        # Display existing biomarkers
                        for i, bm in enumerate(self.patient_profile['biomarkers']):
                            with ui.row().classes('w-full items-center gap-2 mb-2'):
                                ui.input(
                                    label='Name',
                                    value=bm['name'],
                                    on_change=lambda e, idx=i: self.update_biomarker(idx, 'name', e.value)
                                ).classes('flex-1')
                                ui.input(
                                    label='Status',
                                    value=bm['status'],
                                    on_change=lambda e, idx=i: self.update_biomarker(idx, 'status', e.value)
                                ).classes('flex-1')
                                ui.button('Remove', icon='delete', on_click=lambda idx=i: self.remove_biomarker(idx)).classes('w-24')
                
                # Genomic Variants
                with ui.expansion('Genomic Variants', icon='dna', value=True).classes('w-full'):
                    with ui.card().classes('w-full p-4'):
                        # Add button to add new variant
                        ui.button('Add Variant', icon='add', on_click=self.add_variant).classes('mb-2')
                        
                        # Display existing variants
                        for i, var in enumerate(self.patient_profile['genomic_variants']):
                            with ui.row().classes('w-full items-center gap-2 mb-2'):
                                ui.input(
                                    label='Gene',
                                    value=var['gene'],
                                    on_change=lambda e, idx=i: self.update_variant(idx, 'gene', e.value)
                                ).classes('flex-1')
                                ui.input(
                                    label='Variant',
                                    value=var['variant'],
                                    on_change=lambda e, idx=i: self.update_variant(idx, 'variant', e.value)
                                ).classes('flex-1')
                                ui.button('Remove', icon='delete', on_click=lambda idx=i: self.remove_variant(idx)).classes('w-24')

    def update_patient_profile(self, field: str, value: str):
        """Update patient profile field"""
        self.logger.info(f"Updating patient profile field '{field}' to '{value}'")
        self.patient_profile[field] = value
        ui.notify(f"Updated patient {field} to {value}")

    def update_biomarker(self, index: int, field: str, value: str):
        """Update biomarker at given index"""
        self.patient_profile['biomarkers'][index][field] = value
        ui.notify(f"Updated biomarker {index} {field} to {value}")

    def update_variant(self, index: int, field: str, value: str):
        """Update genomic variant at given index"""
        self.patient_profile['genomic_variants'][index][field] = value
        ui.notify(f"Updated variant {index} {field} to {value}")

    def add_biomarker(self):
        """Add a new biomarker to the patient profile"""
        self.logger.info("Adding new biomarker to patient profile")
        new_biomarker = {'name': '', 'status': ''}
        self.patient_profile['biomarkers'].append(new_biomarker)
        # Refresh the patient panel to show the new biomarker
        self.refresh_patient_panel()

    def remove_biomarker(self, index: int):
        """Remove a biomarker from the patient profile"""
        self.logger.info(f"Removing biomarker at index {index}")
        if 0 <= index < len(self.patient_profile['biomarkers']):
            removed = self.patient_profile['biomarkers'].pop(index)
            self.logger.debug(f"Removed biomarker: {removed['name']}")
            ui.notify(f"Removed biomarker: {removed['name']}")
            # Refresh the patient panel to update the display
            self.refresh_patient_panel()
        else:
            self.logger.warning(f"Invalid biomarker index {index} for removal")

    def add_variant(self):
        """Add a new genomic variant to the patient profile"""
        new_variant = {'gene': '', 'variant': ''}
        self.patient_profile['genomic_variants'].append(new_variant)
        # Refresh the patient panel to show the new variant
        self.refresh_patient_panel()

    def remove_variant(self, index: int):
        """Remove a genomic variant from the patient profile"""
        if 0 <= index < len(self.patient_profile['genomic_variants']):
            removed = self.patient_profile['genomic_variants'].pop(index)
            ui.notify(f"Removed variant: {removed['gene']}")
            # Refresh the patient panel to update the display
            self.refresh_patient_panel()

    def refresh_patient_panel(self):
        """Refresh the patient panel to reflect changes"""
        # Clear the patient panel container
        self.patient_panel_container.clear()
        
        # Rebuild the patient panel
        with self.patient_panel_container:
            self.setup_patient_panel()
        
        ui.notify("Patient profile updated.")

    def setup_pagination(self):
        """Setup pagination controls for search results"""
        with ui.column().classes('w-full gap-2 mt-4'):
            # Page navigation buttons
            with ui.row().classes('w-full justify-between'):
                self.prev_btn = ui.button('Previous', icon='chevron_left', on_click=self.prev_page)\
                    .props('flat').classes('w-32')
                self.next_btn = ui.button('Next', icon='chevron_right', on_click=self.next_page)\
                    .props('flat').classes('w-32')
            
            # Page numbers display for visited pages
            self.page_numbers_container = ui.row().classes('w-full justify-center gap-1')
            
            # Page info labels
            with ui.row().classes('w-full justify-center'):
                self.page_label = ui.label('Page 1').classes('text-sm text-gray-600')
                self.total_label = ui.label('').classes('text-sm text-gray-600')
        
        # Set initial button states
        self.prev_btn.set_enabled(False)
        self.next_btn.set_enabled(False)

    async def prev_page(self):
        """Navigate to previous page of results"""
        self.logger.info(f"Navigating to previous page from page {self.current_page}")
        if self.current_page > 1:
            self.current_page -= 1
            # For previous page navigation, we need to re-search from the beginning
            # and iterate through pages to reach the desired page
            self.page_token = None
            target_page = self.current_page
            self.current_page = 1
            
            # Store current search term and reset
            search_term = self.current_search_term
            self.current_search_term = None
            
            self.logger.debug(f"Re-searching to reach target page {target_page}")
            # Perform search to get to target page
            for page in range(1, target_page):
                results = await run.io_bound(
                    search_trials,
                    search_term,
                    fields=None,
                    max_results=10,
                    page_token=self.page_token,
                    use_cache=True
                )
                
                if 'nextPageToken' in results:
                    self.page_token = results['nextPageToken']
                    self.page_tokens[page + 1] = self.page_token
                else:
                    break
            
            # Restore current page and search term
            self.current_page = target_page
            self.current_search_term = search_term
            self.page_token = self.page_tokens.get(target_page)
            
            self.logger.debug(f"Restored to page {self.current_page}")
            # Update UI and perform final search
            self._update_pagination_ui()
            await self.on_search()
        else:
            self.logger.debug("Already on first page")
            ui.notify("Already on first page", type='warning')

    async def next_page(self):
        """Navigate to next page of results"""
        self.logger.info(f"Navigating to next page from page {self.current_page}")
        if self.page_token:
            self.current_page += 1
            await self.on_search()
        else:
            self.logger.debug("No more results available")
            ui.notify("No more results available", type='warning')

    def _update_pagination_ui(self):
        """Update pagination UI elements including page numbers for visited pages"""
        self.page_label.text = f'Page {self.current_page} of {self.total_pages}'
        self.total_label.text = f'Total: {self.total_studies} studies'
        self.prev_btn.set_enabled(self.current_page > 1)
        self.next_btn.set_enabled(self.page_token is not None)
        
        # Update page numbers for visited pages (show up to 5 previous pages)
        self.page_numbers_container.clear()
        with self.page_numbers_container:
            # Show current page and up to 4 previous pages
            start_page = max(1, self.current_page - 4)
            end_page = self.current_page
            
            for page_num in range(start_page, end_page + 1):
                if page_num in self.page_tokens or page_num == 1:
                    btn = ui.button(str(page_num), on_click=lambda p=page_num: self.go_to_page(p))\
                        .props('flat size=sm')
                    if page_num == self.current_page:
                        btn.props('color=primary')
    
    async def go_to_page(self, page_num):
        """Navigate directly to a specific page that has been visited"""
        self.logger.info(f"Navigating to page {page_num} from page {self.current_page}")
        if page_num == self.current_page:
            self.logger.debug("Already on target page")
            return
            
        if page_num in self.page_tokens or page_num == 1:
            self.current_page = page_num
            self.page_token = self.page_tokens.get(page_num)
            self.logger.debug(f"Page token for page {page_num}: {self.page_token[:10] if self.page_token else None}")
            await self.on_search()
        else:
            self.logger.warning(f"Page {page_num} not available in page tokens")
    
    def _create_feature_mapping(self, nlp_result, mapped_mcode):
        """Create a mapping between NLP features and mCODE elements with strict 1:1 biomarker matching
        
        Args:
            nlp_result: NLP processing result containing extracted features
            mapped_mcode: List of mapped mCODE elements
            
        Returns:
            Dictionary mapping feature types to paired NLP features and mCODE elements,
            ensuring exact 1:1 matches for biomarkers based on both name and status
        
        Args:
            nlp_result: NLP processing result
            mapped_mcode: List of mapped mCODE elements
            
        Returns:
            Dictionary mapping feature types to paired NLP features and mCODE elements
        """
        feature_mapping = {}
        
        # Special handling for biomarkers to ensure 1:1 mapping
        biomarker_mapping = []
        nlp_biomarkers = [e for e in (nlp_result.entities if nlp_result else []) if e.get('type') == 'Biomarker']
        mcode_biomarkers = [e for e in mapped_mcode if e.get('element_type') == 'Biomarker']
        
        # Pair biomarkers by name
        for nlp_bm in nlp_biomarkers:
            for mcode_bm in mcode_biomarkers:
                # Match biomarkers by name AND status for exact 1:1 mapping
                if (nlp_bm.get('text', '').lower() == mcode_bm.get('element_name', '').lower() and
                    nlp_bm.get('status', '').lower() == mcode_bm.get('value', '').lower()):
                    biomarker_mapping.append({
                        'nlp_feature': nlp_bm,
                        'mcode_element': mcode_bm
                    })
                    break
        
        # Add biomarker pairs to feature mapping
        if biomarker_mapping:
            feature_mapping['Biomarker'] = biomarker_mapping
        
        # Group remaining features by type
        for entity in [e for e in (nlp_result.entities if nlp_result else []) if e.get('type') != 'Biomarker']:
            feature_type = entity.get('type', 'Other')
            if feature_type not in feature_mapping:
                feature_mapping[feature_type] = []
            feature_mapping[feature_type].append({
                'nlp_feature': entity,
                'mcode_element': None  # Will be matched later
            })
        
        # Match remaining mCODE elements
        for element in [e for e in mapped_mcode if e.get('element_type') != 'Biomarker']:
            element_type = element.get('element_type', 'Other')
            if element_type in feature_mapping:
                # Try to find matching NLP feature
                matched = False
                for item in feature_mapping[element_type]:
                    if (item['nlp_feature'] and
                        item['nlp_feature'].get('text', '').lower() == element.get('element_name', '').lower()):
                        item['mcode_element'] = element
                        matched = True
                        break
                
                if not matched:
                    feature_mapping[element_type].append({
                        'nlp_feature': None,
                        'mcode_element': element
                    })
            else:
                feature_mapping[element_type] = [{
                    'nlp_feature': None,
                    'mcode_element': element
                }]
        
        return feature_mapping

    def _update_entities_with_mapping(self, entities, mapped_mcode, source_references=None):
        """Update NLP entities with mapping information for visualization
        
        Args:
            entities: List of NLP entities
            mapped_mcode: List of mapped mCODE elements
            source_references: List of SourceReference objects for provenance tracking
        """
        # Create a mapping from entity text to mCODE element
        mapping_dict = {}
        for mcode_element in mapped_mcode:
            if isinstance(mcode_element, dict):
                element_name = mcode_element.get('element_name', '')
                element_type = mcode_element.get('element_type', '')
                element_value = mcode_element.get('value', '')
                
                # Try multiple ways to match back to source entity
                keys_to_try = [
                    mcode_element.get('mapped_from', ''),
                    element_name.lower(),
                    f"{element_name.lower()}-{element_value.lower()}" if element_value else ''
                ]
                
                for key in keys_to_try:
                    if key:
                        mapping_dict[key] = {
                            'name': element_name,
                            'type': element_type,
                            'value': element_value
                        }
        
        # Update entities with mapping information
        for entity in entities:
            entity_text = entity.get('text', '').lower()
            entity_with_value = f"{entity_text}-{entity.get('value', '').lower()}"
            
            # Try to find mapping with multiple attempts
            mapping = (mapping_dict.get(entity_text) or
                      mapping_dict.get(entity_with_value))
            
            if mapping:
                entity['mapped_to'] = mapping['name']
                entity['mapped_type'] = mapping['type']
                entity['mapped_value'] = mapping['value']
                entity['has_mapping'] = True
    
    def _display_mapping_relationships(self, entities, mapped_mcode, source_references):
        """Display many-to-one mapping relationships between NLP entities and mCODE elements
        
        Args:
            entities: List of NLP entities
            mapped_mcode: List of mapped mCODE elements
            source_references: List of SourceReference objects for provenance tracking
        """
        # Group mCODE elements by type for better organization
        mcode_by_type = {}
        for element in mapped_mcode:
            element_type = element.get('element_type', 'Other')
            if element_type not in mcode_by_type:
                mcode_by_type[element_type] = []
            mcode_by_type[element_type].append(element)
        
        # Display mCODE elements grouped by type with their source relationships
        for element_type, elements in mcode_by_type.items():
            with ui.expansion(f"{element_type} Mappings", icon='category', value=True).classes('w-full mb-4'):
                for element in elements:
                    with ui.card().classes('w-full p-4 mb-3 bg-gray-50'):
                        # mCODE element header
                        with ui.row().classes('w-full justify-between items-center mb-3'):
                            element_name = element.get('element_name', 'Unknown')
                            confidence = element.get('confidence', 0.8)
                            
                            ui.label(element_name).classes('font-bold text-lg')
                            ui.badge(f"{confidence:.2f}").props(
                                f'color={"green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"}'
                            )
                        
                        # Show value if available
                        value = element.get('value', '')
                        if value:
                            with ui.row().classes('items-center gap-2 mb-3'):
                                ui.label('Value:').classes('font-semibold')
                                ui.label(value).classes('text-blue-600')
                        
                        # Show source references and connected NLP entities
                        element_sources = element.get('source_references', [])
                        if element_sources:
                            with ui.expansion('Source Provenance', icon='source').classes('w-full'):
                                for source_ref in element_sources:
                                    with ui.card().classes('w-full p-3 mb-2 bg-white'):
                                        # Source reference details
                                        with ui.row().classes('items-center gap-2 mb-2'):
                                            ui.icon('description').classes('text-blue-500')
                                            ui.label(f"{source_ref.get('section_name', 'Unknown')} ({source_ref.get('section_type', 'Unknown')})").classes('font-medium')
                                        
                                        # Text fragment
                                        text_fragment = source_ref.get('text_fragment', '')
                                        if text_fragment:
                                            ui.markdown(f"**Text:** `{text_fragment[:100]}{'...' if len(text_fragment) > 100 else ''}`").classes('text-sm mb-2')
                                        
                                        # Confidence and method
                                        with ui.row().classes('items-center gap-4 text-xs text-gray-500'):
                                            ui.label(f"Confidence: {source_ref.get('confidence', 0.8):.2f}")
                                            ui.label(f"Method: {source_ref.get('extraction_method', 'Unknown')}")
                        
                        # Find and show connected NLP entities
                        connected_entities = []
                        text_fragments = [ref.get('text_fragment', '').lower() for ref in element_sources]
                        
                        for entity in entities:
                            entity_text = entity.get('text', '').lower()
                            if any(fragment in entity_text or entity_text in fragment for fragment in text_fragments if fragment):
                                connected_entities.append(entity)
                        
                        if connected_entities:
                            with ui.expansion(f"Connected NLP Entities ({len(connected_entities)})", icon='link').classes('w-full'):
                                with ui.row().classes('flex-wrap gap-2'):
                                    for entity in connected_entities:
                                        entity_text = entity.get('text', 'Unknown')
                                        entity_type = entity.get('type', 'Unknown')
                                        confidence = entity.get('confidence', 0.8)
                                        
                                        badge = ui.badge(f"{entity_text}").props(
                                            f'color={"green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"}'
                                        )
                                        badge.tooltip(self._create_entity_tooltip(entity, entity_type, source_references))

    def _calculate_total_studies(self, search_term):
        """Calculate total studies and pages for a search term"""
        from src.pipeline.fetcher import calculate_total_studies
        try:
            return calculate_total_studies(search_term, page_size=10)
        except Exception as e:
            logger.error(f"Error calculating total studies: {str(e)}")
            return {'total_studies': 0, 'total_pages': 1, 'page_size': 10}

    def _create_entity_tooltip(self, entity, entity_type, source_references=None):
        """Create comprehensive tooltip content for NLP entity showing source provenance
        
        Args:
            entity: NLP entity dictionary
            entity_type: Type of the entity
            source_references: List of SourceReference objects for provenance tracking
            
        Returns:
            String with detailed source information
        """
        text = entity.get('text', 'Unknown')
        confidence = entity.get('confidence', 0.8)
        source = entity.get('source', 'NLP extraction')
        extraction_method = entity.get('extraction_method', 'Pattern matching')
        context = entity.get('context', 'No context available')
        
        # Find source references for this entity
        entity_sources = []
        if source_references:
            entity_text = text.lower()
            for ref in source_references:
                if ref.get('text_fragment', '').lower() in entity_text or entity_text in ref.get('text_fragment', '').lower():
                    entity_sources.append(ref)
        
        tooltip_content = f"""
{text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Type: {entity_type}
Confidence: {confidence:.2f}
Source: {source}
Method: {extraction_method}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Context: {context[:80]}{'...' if len(context) > 80 else ''}
        """
        
        # Add source provenance information if available
        if entity_sources:
            tooltip_content += "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            tooltip_content += "\nSource Provenance:"
            for i, source_ref in enumerate(entity_sources[:3]):  # Show up to 3 sources
                section = source_ref.get('section_name', 'Unknown')
                fragment = source_ref.get('text_fragment', '')
                tooltip_content += f"\n{i+1}. {section}: {fragment[:60]}{'...' if len(fragment) > 60 else ''}"
            if len(entity_sources) > 3:
                tooltip_content += f"\n... and {len(entity_sources) - 3} more sources"
        
        return tooltip_content.strip()

    def _create_mcode_tooltip(self, element, element_type, source_references=None):
        """Create comprehensive tooltip content for mCODE element showing source provenance
        
        Args:
            element: mCODE element dictionary
            element_type: Type of the element
            source_references: List of SourceReference objects for provenance tracking
            
        Returns:
            String with detailed source information
        """
        element_name = element.get('element_name', 'Unknown')
        value = element.get('value', 'Not specified')
        confidence = element.get('confidence', 0.8)
        source = element.get('source', 'Clinical trial criteria')
        mapping_method = element.get('mapping_method', 'Rule-based mapping')
        criteria_context = element.get('criteria_context', 'No context available')
        
        # Get source references from the element itself (connected by LLMMappingEngine)
        element_sources = element.get('source_references', [])
        
        tooltip_content = f"""
{element_name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Type: {element_type}
Value: {value}
Confidence: {confidence:.2f}
Source: {source}
Method: {mapping_method}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Criteria Context: {criteria_context[:80]}{'...' if len(criteria_context) > 80 else ''}
        """
        
        # Add source provenance information if available
        if element_sources:
            tooltip_content += "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            tooltip_content += "\nSource Provenance:"
            for i, source_ref in enumerate(element_sources[:3]):  # Show up to 3 sources
                section = source_ref.get('section_name', 'Unknown')
                fragment = source_ref.get('text_fragment', '')
                tooltip_content += f"\n{i+1}. {section}: {fragment[:60]}{'...' if len(fragment) > 60 else ''}"
            if len(element_sources) > 3:
                tooltip_content += f"\n... and {len(element_sources) - 3} more sources"
        
        return tooltip_content.strip()

# Run the application
if __name__ in {'__main__', '__mp_main__', 'patient_matcher'}:
    app = PatientMatcherApp()
    ui.run(title='Patient-Trial Matcher', dark=False)