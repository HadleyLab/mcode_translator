import asyncio
import json
from typing import Dict, List, Optional
from nicegui import ui, run
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher.fetcher import search_trials, get_full_study
from mcode_mapper.mcode_mapping_engine import MCODEMappingEngine
from nlp_engine.llm_nlp_engine import LLMNLPEngine
from code_extraction.code_extraction import CodeExtractionModule

class PatientMatcherApp:
    def __init__(self):
        self.current_trial = None
        self.search_results = []
        self.loading = False
        self.mcode_mapper = MCODEMappingEngine()
        self.nlp_engine = LLMNLPEngine()
        self.code_extractor = CodeExtractionModule()
        
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
            
            # Search input
            self.search_input = ui.input(
                placeholder='Search trials by cancer type or biomarker',
                value=self.patient_profile['cancer_type']
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
            
            # Calculate total studies and pages if this is a new search
            if self.page_token is None and self.current_page == 1:
                # Get total count and pages
                stats = await run.io_bound(
                    self._calculate_total_studies,
                    search_term
                )
                self.total_studies = stats['total_studies']
                self.total_pages = stats['total_pages']
            
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
            
            # Update pagination info
            if 'nextPageToken' in results:
                next_page_token = results['nextPageToken']
                self.page_token = next_page_token
                # Store the token for this page
                self.page_tokens[self.current_page + 1] = next_page_token
            else:
                self.page_token = None
            
            # Update page label and button states
            self._update_pagination_ui()
            
            for study in studies:
                if isinstance(study, dict):
                    # New format (studies array)
                    if 'protocolSection' in study:
                        ident = study.get('protocolSection', {}).get('identificationModule', {})
                        status = study.get('protocolSection', {}).get('statusModule', {})
                        self.search_results.append({
                            'NCTId': ident.get('nctId', 'N/A'),
                            'BriefTitle': ident.get('briefTitle', 'No title'),
                            'Condition': ', '.join(study.get('protocolSection', {}).get('conditionsModule', {}).get('conditions', [])),
                            'OverallStatus': status.get('overallStatus', 'Unknown')
                        })
                    # Old format (StudyFields)
                    elif 'NCTId' in study:
                        self.search_results.append({
                            'NCTId': study.get('NCTId', [''])[0],
                            'BriefTitle': study.get('BriefTitle', [''])[0],
                            'Condition': ', '.join(study.get('Condition', [''])),
                            'OverallStatus': study.get('OverallStatus', [''])[0]
                        })
            self.display_search_results()
            self.status_label.text = f'Found {len(self.search_results)} trials'
            
        except Exception as e:
            self.status_label.text = f'Error: {str(e)}'
            ui.notify(f'Search failed: {str(e)}', type='negative')
        finally:
            self.loading = False

    def display_search_results(self):
        """Display search results in left panel"""
        for trial in self.search_results:
            with self.results_container:
                with ui.card().classes('w-full p-3 mb-2 cursor-pointer hover:bg-gray-50') as card:
                    ui.label(trial.get('BriefTitle', 'No title')).classes('font-bold')
                    with ui.row().classes('items-center justify-between'):
                        ui.label(trial.get('NCTId', 'N/A')).classes('text-xs text-gray-600')
                        ui.badge(trial.get('OverallStatus', 'Unknown')).props('color=green')
                    
                    # Click handler to show details
                    card.on('click', lambda t=trial: self.show_trial_details(t))

    async def show_trial_details(self, trial):
        """Show detailed trial view with matching analysis"""
        self.current_trial = trial
        self.details_panel.clear()
        
        with self.details_panel:
            # Loading state
            loading = ui.spinner('dots', size='lg', color='primary')
            status = ui.label('Loading trial details...')
            
            try:
                # Fetch full trial details
                full_study = await run.io_bound(
                    get_full_study,
                    trial['NCTId']
                )
                
                if not full_study:
                    raise Exception('No trial details found')
                
                # Extract eligibility criteria
                criteria = full_study.get('protocolSection', {}).get('eligibilityModule', {}).get('eligibilityCriteria', '')
                
                # Process with NLP engine using trial context
                nlp_result = await run.io_bound(
                    self.nlp_engine.process_trial_context,
                    criteria,
                    full_study
                )
                
                # Extract codes
                codes_result = await run.io_bound(
                    self.code_extractor.process_criteria_for_codes,
                    criteria,
                    nlp_result.entities if nlp_result else None
                )
                
                # Enhanced mCODE mapping with fallbacks
                mapped_mcode = []
                
                # 1. Try full NLP + code extraction pipeline
                try:
                    all_entities = []
                    if nlp_result and not nlp_result.error:
                        all_entities.extend(nlp_result.entities)
                    if codes_result and 'extracted_codes' in codes_result:
                        for system, codes in codes_result['extracted_codes'].items():
                            all_entities.extend([{
                                'text': c.get('text', ''),
                                'confidence': c.get('confidence', 0.8),
                                'codes': {system: c.get('code', '')}
                            } for c in codes])
                    
                    mapped_mcode = await run.io_bound(
                        self.mcode_mapper.map_entities_to_mcode,
                        all_entities,
                        full_study  # Pass trial information for context-aware mapping
                    )
                except Exception as e:
                    print(f"Full mapping failed: {str(e)}")
                
                # 2. Fallback to basic cancer type mapping if no elements found
                if not mapped_mcode:
                    cancer_type = trial.get('protocolSection', {}).get('conditionsModule', {}).get('conditions', [''])[0]
                    if cancer_type.lower() == 'breast cancer':
                        mapped_mcode = [{
                            'element_name': 'Breast Cancer',
                            'element_type': 'Condition',
                            'value': cancer_type,
                            'confidence': 0.9,
                            'primary_code': {
                                'system': 'ICD10CM',
                                'code': 'C50.911'
                            }
                        }]
                    elif cancer_type:
                        mapped_mcode = [{
                            'element_name': cancer_type,
                            'element_type': 'Condition',
                            'value': cancer_type,
                            'confidence': 0.7
                        }]
                
                # 3. Add study-specific biomarkers only if relevant
                if not mapped_mcode:
                    # Get study conditions
                    study_conditions = trial.get('protocolSection', {}).get('conditionsModule', {}).get('conditions', [])
                    study_conditions_lower = [c.lower() for c in study_conditions]
                    
                    # Only add patient biomarkers if they match study conditions
                    if (self.patient_profile.get('cancer_type', '').lower() in study_conditions_lower and
                        self.patient_profile.get('biomarkers')):
                        relevant_biomarkers = []
                        for b in self.patient_profile['biomarkers']:
                            # Check if biomarker is mentioned in eligibility criteria
                            criteria = trial.get('protocolSection', {}).get('eligibilityModule', {}).get('eligibilityCriteria', '').lower()
                            if b['name'].lower() in criteria:
                                relevant_biomarkers.append({
                                    'element_name': b['name'],
                                    'element_type': 'Biomarker',
                                    'status': b['status'],
                                    'confidence': 0.8,
                                    'source': 'patient'
                                })
                        
                        if relevant_biomarkers:
                            mapped_mcode.extend(relevant_biomarkers)
                
                # Clear loading state
                loading.set_visibility(False)
                status.set_visibility(False)
                
                # Display detailed view with NLP results
                self.display_trial_analysis(full_study, mapped_mcode, nlp_result)
                
            except Exception as e:
                loading.set_visibility(False)
                status.text = f'Error loading details: {str(e)}'
                ui.notify(f'Failed to load trial details: {str(e)}', type='negative')

    def display_trial_analysis(self, trial, mapped_mcode, nlp_result=None):
        """Display trial analysis with matching visualization
        
        Args:
            trial: Dictionary containing trial details from ClinicalTrials.gov API
            mapped_mcode: List of mCODE elements mapped from trial criteria
            nlp_result: NLP processing result containing raw extracted features
            
        Data Binding Notes:
            - Trial details are bound directly from API response
            - mCODE elements are grouped by type for display
            - Confidence scores are color-coded (green >70%, orange >50%, red <=50%)
            - Biomarker matching is calculated against patient profile
            - Shows side-by-side comparison of NLP vs mCODE results
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
            
            # Data Visualization
            with ui.expansion('Data Analysis', icon='analytics', value=True).classes('w-full'):
                # NLP Extractions
                with ui.expansion('NLP Extractions', icon='text_snippet').classes('w-full'):
                    if nlp_result and not nlp_result.error and hasattr(nlp_result, 'features'):
                        features = nlp_result.features
                        for feature_type, feature_data in features.items():
                            with ui.expansion(feature_type.title(), icon='category').classes('w-full'):
                                if isinstance(feature_data, list):
                                    for item in feature_data:
                                        if isinstance(item, dict):
                                            with ui.card().classes('w-full p-2 mb-2'):
                                                for key, value in item.items():
                                                    ui.label(f"{key}: {value}").classes('text-xs')
                                elif isinstance(feature_data, dict):
                                    for key, value in feature_data.items():
                                        ui.label(f"{key}: {value}").classes('text-sm')
                    else:
                        ui.label('No NLP features extracted').classes('text-gray-500')
                
                # mCODE Elements
                with ui.expansion('mCODE Elements', icon='dna').classes('w-full'):
                    if mapped_mcode:
                        # Group elements by type
                        element_groups = {}
                        for element in mapped_mcode:
                            element_type = element.get('element_type', 'Other')
                            if element_type not in element_groups:
                                element_groups[element_type] = []
                            element_groups[element_type].append(element)
                        
                        # Display each group
                        for element_type, elements in element_groups.items():
                            with ui.expansion(element_type, icon='category').classes('w-full'):
                                for element in elements:
                                    with ui.card().classes('w-full p-3 mb-2 hover:shadow-md'):
                                        # Main element info
                                        with ui.row().classes('items-center justify-between'):
                                            ui.label(element.get('element_name', 'Unknown')).classes('font-bold')
                                            if 'confidence' in element:
                                                ui.badge(f"{element['confidence']*100:.0f}%")\
                                                    .props(f'color={"green" if element["confidence"] > 0.7 else "orange" if element["confidence"] > 0.5 else "red"}')
                                        
                                        # Detailed fields
                                        with ui.column().classes('mt-2 gap-1'):
                                            if 'value' in element:
                                                with ui.row().classes('items-center gap-2'):
                                                    ui.icon('check_circle').classes('text-green-500')
                                                    ui.label(f"Value: {element['value']}").classes('text-sm')
                                            if 'status' in element:
                                                with ui.row().classes('items-center gap-2'):
                                                    ui.icon('info').classes('text-blue-500')
                                                    ui.label(f"Status: {element['status']}").classes('text-sm')
                                            
                                            # Code information
                                            if 'primary_code' in element:
                                                primary_code = element['primary_code']
                                                with ui.row().classes('items-center gap-2'):
                                                    ui.icon('code').classes('text-purple-500')
                                                    ui.label(f"{primary_code.get('system', 'N/A')}: {primary_code.get('code', 'N/A')}").classes('text-xs font-mono')
                                            
                                            # Additional codes
                                            if 'mapped_codes' in element:
                                                mapped_codes = element['mapped_codes']
                                                for system, code in mapped_codes.items():
                                                    with ui.row().classes('items-center gap-2'):
                                                        ui.icon('link').classes('text-gray-500')
                                                        ui.label(f"{system}: {code}").classes('text-xs font-mono')
                                            
                                            # Source information
                                            if 'source' in element:
                                                with ui.row().classes('items-center gap-2'):
                                                    ui.icon('source').classes('text-yellow-500')
                                                    ui.label(f"Source: {element['source']}").classes('text-xs')
                    else:
                        ui.label('No mCODE elements were mapped').classes('text-gray-500')
            
            # # Original mCODE visualization panel (kept for backward compatibility)
            # with ui.expansion('mCODE Detailed View', icon='dna').classes('w-full'):
            #     if mapped_mcode:
            #         # Summary stats
            #         with ui.card().classes('w-full p-4 bg-blue-50'):
            #             with ui.row().classes('items-center justify-between'):
            #                 ui.label('Mapped Elements').classes('font-bold')
            #                 ui.badge(f"{len(mapped_mcode)} elements").props('color=blue')
                    
            #         # Group elements by type
            #         element_groups = {}
            #         for element in mapped_mcode:
            #             element_type = element.get('element_type', 'Other')
            #             if element_type not in element_groups:
            #                 element_groups[element_type] = []
            #             element_groups[element_type].append(element)
                    
            #         # Display each group with enhanced details
            #         for element_type, elements in element_groups.items():
            #             with ui.expansion(element_type, icon='category').classes('w-full'):
            #                 with ui.grid(columns=1).classes('w-full gap-4'):
            #                     for element in elements:
            #                         with ui.card().classes('w-full p-4 hover:shadow-md'):
            #                             # Main element info
            #                             with ui.row().classes('items-center justify-between'):
            #                                 ui.label(element.get('element_name', 'Unknown')).classes('font-bold')
            #                                 if 'confidence' in element:
            #                                     ui.badge(f"{element['confidence']*100:.0f}%")\
            #                                         .props(f'color={"green" if element["confidence"] > 0.7 else "orange" if element["confidence"] > 0.5 else "red"}')
                                        
            #                             # Detailed fields with better organization
            #                             with ui.column().classes('mt-2 gap-1'):
            #                                 # Value information
            #                                 if 'value' in element:
            #                                     with ui.row().classes('items-center gap-2'):
            #                                         ui.icon('check_circle').classes('text-green-500')
            #                                         ui.label(f"Value: {element['value']}").classes('text-sm')
            #                                 if 'status' in element:
            #                                     with ui.row().classes('items-center gap-2'):
            #                                         ui.icon('info').classes('text-blue-500')
            #                                         ui.label(f"Status: {element['status']}").classes('text-sm')
                                            
            #                                 # Code information
            #                                 if 'primary_code' in element:
            #                                     primary_code = element['primary_code']
            #                                     with ui.row().classes('items-center gap-2'):
            #                                         ui.icon('code').classes('text-purple-500')
            #                                         ui.label(f"{primary_code.get('system', 'N/A')}: {primary_code.get('code', 'N/A')}").classes('text-xs font-mono')
                                            
            #                                 # Additional codes
            #                                 if 'mapped_codes' in element:
            #                                     mapped_codes = element['mapped_codes']
            #                                     for system, code in mapped_codes.items():
            #                                         with ui.row().classes('items-center gap-2'):
            #                                             ui.icon('link').classes('text-gray-500')
            #                                             ui.label(f"{system}: {code}").classes('text-xs font-mono')
                                            
            #                                 # Source information
            #                                 if 'source' in element:
            #                                     with ui.row().classes('items-center gap-2'):
            #                                         ui.icon('source').classes('text-yellow-500')
            #                                         ui.label(f"Source: {element['source']}").classes('text-xs')
            #     else:
            #         ui.label('No mCODE elements were mapped from this trial').classes('text-gray-500 p-4 text-center')
            
            # Raw eligibility criteria
            with ui.expansion('Eligibility Criteria', icon='list').classes('w-full'):
                criteria = trial.get('protocolSection', {}).get('eligibilityModule', {}).get('eligibilityCriteria', 'No criteria available')
                ui.markdown(f"```\n{criteria}\n```").classes('text-xs')

    def reset_search(self):
        """Reset search results and pagination state"""
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
        new_biomarker = {'name': '', 'status': ''}
        self.patient_profile['biomarkers'].append(new_biomarker)
        # Refresh the patient panel to show the new biomarker
        self.refresh_patient_panel()

    def remove_biomarker(self, index: int):
        """Remove a biomarker from the patient profile"""
        if 0 <= index < len(self.patient_profile['biomarkers']):
            removed = self.patient_profile['biomarkers'].pop(index)
            ui.notify(f"Removed biomarker: {removed['name']}")
            # Refresh the patient panel to update the display
            self.refresh_patient_panel()

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
            
            # Update UI and perform final search
            self._update_pagination_ui()
            await self.on_search()
        else:
            ui.notify("Already on first page", type='warning')

    async def next_page(self):
        """Navigate to next page of results"""
        if self.page_token:
            self.current_page += 1
            await self.on_search()
        else:
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
        if page_num == self.current_page:
            return
            
        if page_num in self.page_tokens or page_num == 1:
            self.current_page = page_num
            self.page_token = self.page_tokens.get(page_num)
            await self.on_search()
    
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

    def _calculate_total_studies(self, search_term):
        """Calculate total studies and pages for a search term"""
        from data_fetcher.fetcher import calculate_total_studies
        try:
            return calculate_total_studies(search_term, page_size=10)
        except Exception as e:
            print(f"Error calculating total studies: {str(e)}")
            return {'total_studies': 0, 'total_pages': 1, 'page_size': 10}

# Run the application
if __name__ in {'__main__', '__mp_main__', 'patient_matcher'}:
    app = PatientMatcherApp()
    ui.run(title='Patient-Trial Matcher', dark=False)