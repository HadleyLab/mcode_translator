"""
Clinical Trial Data Fetcher Demo Application
Demonstrates search capabilities of the fetcher.py module using NiceGUI
"""
import json
import sys
import os
from pathlib import Path
from nicegui import ui

# Add src directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import data fetcher functions
from src.data_fetcher.fetcher import (
    search_trials,
    get_full_study,
    calculate_total_studies,
    display_single_study
)
from collections import Counter
import logging

# Import mCODE modules
from src.code_extraction.code_extraction import CodeExtractionModule
from src.mcode_mapper.mcode_mapping_engine import MCODEMappingEngine
from src.nlp_engine.llm_nlp_engine import LLMNLPEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
current_search_results = []
current_study_details = None
current_page = 1
results_per_page = 10
total_studies = 0
total_pages = 1
search_expression = "breast cancer"
page_tokens = [None]  # Store page tokens for each page, index 0 is for page 1
current_view = "table"  # Default view: table, cards, list
results_container = None  # Global container for results

# Color mappings for consistency
STATUS_COLORS = {
    'ACTIVE_NOT_RECRUITING': 'blue',
    'RECRUITING': 'green',
    'COMPLETED': 'grey',
    'TERMINATED': 'red',
    'WITHDRAWN': 'orange',
    'UNKNOWN': 'gray'
}

# Store condition colors for consistency between charts and badges
condition_colors = {}

def fetch_search_results():
    """Fetch search results using the search expression."""
    global current_search_results, current_page, total_studies, total_pages, page_tokens
    
    try:
        ui.notify(f"Fetching search results for page {current_page}...", type='info')
        
        # Get total count first
        stats = calculate_total_studies(search_expression, page_size=results_per_page)
        total_studies = stats.get('total_studies', 0)
        total_pages = stats.get('total_pages', 1)
        
        # Ensure page_tokens array is large enough
        while len(page_tokens) <= total_pages:
            page_tokens.append(None)
        
        # Check if we have a page token for the current page
        # If not, reset to page 1
        if current_page > 1 and (current_page - 1 >= len(page_tokens) or page_tokens[current_page - 1] is None):
            logger.info(f"No page token for page {current_page}, resetting to page 1")
            current_page = 1
        
        # Fetch data for current page
        results = search_trials(
            search_expr=search_expression,
            fields=["NCTId", "BriefTitle", "Condition", "OverallStatus", "BriefSummary"],
            max_results=results_per_page,
            page_token=page_tokens[current_page - 1],
            use_cache=False
        )
        
        current_search_results = results.get('studies', [])
        
        # Store nextPageToken for the next page if it exists
        if 'nextPageToken' in results and results['nextPageToken']:
            # Ensure we have enough space in the page_tokens array
            if current_page >= len(page_tokens):
                page_tokens.extend([None] * (current_page - len(page_tokens) + 1))
            page_tokens[current_page] = results['nextPageToken']
        
        ui.notify(f"Found {total_studies} total studies, showing page {current_page} of {total_pages}", type='positive')
        logger.info(f"Successfully fetched {len(current_search_results)} studies for page {current_page}")
        
        # Log the NCT IDs of the fetched studies for debugging
        nct_ids = []
        for study in current_search_results:
            protocol_section = study.get('protocolSection', {})
            identification_module = protocol_section.get('identificationModule', {})
            nct_id = identification_module.get('nctId', 'Unknown')
            nct_ids.append(nct_id)
        logger.info(f"Page {current_page} NCT IDs: {nct_ids}")
        
        return True
        
    except Exception as e:
        error_msg = f"Error fetching search results: {str(e)}"
        ui.notify(error_msg, type='negative')
        logger.error(error_msg)
        current_search_results = []
        return False

def fetch_study_details(nct_id: str):
    """Fetch detailed information for a specific study."""
    global current_study_details
    
    try:
        ui.notify(f"Fetching details for study {nct_id}...", type='info')
        study_data = get_full_study(nct_id)
        # Handle case where get_full_study returns a list of studies
        if isinstance(study_data, dict) and 'studies' in study_data:
            current_study_details = study_data['studies'][0] if study_data['studies'] else None
        else:
            current_study_details = study_data
        ui.notify(f"Loaded details for study {nct_id}", type='positive')
        logger.info(f"Successfully fetched details for study {nct_id}")
        return True
    except Exception as e:
        error_msg = f"Error fetching study details: {str(e)}"
        ui.notify(error_msg, type='negative')
        logger.error(error_msg)
        current_study_details = None
        return False

def export_to_json(data, filename: str):
    """Export data to JSON file."""
    try:
        ui.notify(f"Exporting data to {filename}...", type='info')
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        ui.notify(f"Data exported to {filename}", type='positive')
        logger.info(f"Successfully exported data to {filename}")
        return True
    except Exception as e:
        error_msg = f"Error exporting data: {str(e)}"
        ui.notify(error_msg, type='negative')
        logger.error(error_msg)
        return False

def extract_mcode_data(study):
    """Extract mCODE data from a study."""
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
        
        # Initialize mCODE modules
        code_extractor = CodeExtractionModule()
        mcode_mapper = MCODEMappingEngine()
        nlp_engine = LLMNLPEngine()
        
        # Process eligibility criteria through the full mCODE pipeline
        # Step 1: NLP processing
        # Handle empty or invalid eligibility criteria
        if not eligibility_criteria or not isinstance(eligibility_criteria, str) or len(eligibility_criteria.strip()) == 0:
            logger.warning(f"No eligibility criteria found for study {nct_id}")
            nlp_result = None
        else:
            try:
                nlp_result = nlp_engine.process_criteria(eligibility_criteria)
                # Check if processing failed
                if nlp_result.error:
                    logger.warning(f"LLM NLP processing failed for study {nct_id}: {nlp_result.error}")
                    nlp_result = None
            except Exception as e:
                logger.error(f"Error processing criteria with LLM NLP engine for study {nct_id}: {str(e)}")
                nlp_result = None
        
        # Step 2: Code extraction
        extracted_codes = code_extractor.process_criteria_for_codes(
            eligibility_criteria if eligibility_criteria else "",
            nlp_result.entities if nlp_result and not nlp_result.error else None
        )
        
        # Step 3: mCODE mapping
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
        
        mapped_mcode = mcode_mapper.map_entities_to_mcode(all_entities)
        
        # Step 4: Generate structured data
        demographics = {}
        if nlp_result and not nlp_result.error and hasattr(nlp_result, 'features'):
            demographics = nlp_result.features.get('demographics', {})
        
        structured_data = mcode_mapper.generate_mcode_structure(mapped_mcode, demographics)
        
        # Step 5: Validate mCODE compliance
        validation_result = mcode_mapper.validate_mcode_compliance({
            'mapped_elements': mapped_mcode,
            'demographics': demographics
        })
        
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


def display_mcode_visualization(study_data):
    """Display mCODE visualization for a study."""
    with ui.card().classes('w-full'):
        if not study_data:
            ui.label('No study data available.').classes('text-gray-500 italic')
            return
        
        # Display basic study information
        ui.label('Study Information').classes('text-h6 text-primary')
        with ui.grid(columns=2).classes('w-full gap-4'):
            # Display conditions
            with ui.card().classes('w-full'):
                ui.label('Conditions').classes('font-weight-bold text-primary')
                conditions = study_data.get('conditions', [])
                if conditions:
                    with ui.column().classes('w-full'):
                        for condition in conditions:
                            ui.label(f'• {condition}').classes('text-body2')
                else:
                    ui.label('No conditions specified').classes('text-gray-500 italic')
            
            # Display status
            with ui.card().classes('w-full'):
                ui.label('Study Status').classes('font-weight-bold text-primary')
                status = study_data.get('status', 'N/A')
                ui.label(status).classes('text-body2')
        
        # Display NLP results
        nlp_result = study_data.get('nlp_result', {})
        if nlp_result:
            ui.label('NLP Analysis').classes('text-h6 text-primary mt-4')
            with ui.card().classes('w-full'):
                entities = nlp_result.get('entities', [])
                if entities:
                    ui.label('Extracted Entities:').classes('font-weight-bold text-primary')
                    with ui.column().classes('w-full'):
                        for entity in entities[:10]:  # Show first 10 entities
                            ui.label(f"• {entity.get('text', 'N/A')} (Confidence: {entity.get('confidence', 0):.2f})").classes('text-body2')
                        if len(entities) > 10:
                            ui.label(f'... and {len(entities) - 10} more entities').classes('text-body2 text-secondary')
                else:
                    ui.label('No entities extracted').classes('text-gray-500 italic')
        
        # Display extracted codes
        extracted_codes = study_data.get('extracted_codes', {})
        if extracted_codes:
            ui.label('Extracted Codes').classes('text-h6 text-primary mt-4')
            # Check if we have extracted_codes in the new format
            codes_data = extracted_codes.get('extracted_codes', extracted_codes)
            if codes_data:
                with ui.grid(columns=2).classes('w-full gap-4'):
                    # Handle both old and new format
                    if isinstance(codes_data, dict):
                        for system, codes in codes_data.items():
                            if codes:
                                with ui.card().classes('w-full'):
                                    ui.label(f'{system} Codes').classes('font-weight-bold text-primary')
                                    with ui.column().classes('w-full'):
                                        # Handle both list of codes and list of code info
                                        if isinstance(codes, list) and codes:
                                            if isinstance(codes[0], dict):
                                                # New format with code info
                                                for code_info in codes[:5]:  # Show first 5 codes
                                                    ui.label(f"• {code_info.get('code', 'N/A')} ({code_info.get('text', 'N/A')})").classes('text-body2')
                                                if len(codes) > 5:
                                                    ui.label(f'... and {len(codes) - 5} more codes').classes('text-body2 text-secondary')
                                            else:
                                                # Old format with just codes
                                                for code in codes[:5]:
                                                    ui.label(f"• {code}").classes('text-body2')
                                                if len(codes) > 5:
                                                    ui.label(f'... and {len(codes) - 5} more codes').classes('text-body2 text-secondary')
                    else:
                        # Handle case where codes_data is not a dict
                        with ui.card().classes('w-full'):
                            ui.label('Codes').classes('font-weight-bold text-primary')
                            ui.label(str(codes_data)).classes('text-body2')
            else:
                ui.label('No codes extracted').classes('text-gray-500 italic text-body2')
        else:
            ui.label('No codes extracted').classes('text-gray-500 italic text-body2')
        
        # Display mCODE mappings
        mapped_mcode = study_data.get('mapped_mcode', [])
        if mapped_mcode:
            ui.label('mCODE Mappings').classes('text-h6 text-primary mt-4')
            with ui.card().classes('w-full'):
                with ui.column().classes('w-full'):
                    for element in mapped_mcode[:10]:  # Show first 10 mappings
                        if element:
                            with ui.card().classes('w-full mb-2'):
                                ui.label(f"{element.get('mcode_element', 'N/A')}").classes('font-weight-bold text-primary')
                                primary_code = element.get('primary_code', {})
                                if primary_code:
                                    ui.label(f"Primary Code: {primary_code.get('code', 'N/A')} ({primary_code.get('system', 'N/A')})").classes('text-body2')
                                else:
                                    ui.label('No primary code').classes('text-body2 text-secondary')
                                
                                mapped_codes = element.get('mapped_codes', {})
                                if mapped_codes:
                                    ui.label('Mapped Codes:').classes('text-body2 font-weight-bold mt-1')
                                    with ui.column().classes('w-full pl-4'):
                                        for system, code in mapped_codes.items():
                                            ui.label(f"• {system}: {code}").classes('text-body2')
                    if len(mapped_mcode) > 10:
                        ui.label(f'... and {len(mapped_mcode) - 10} more mappings').classes('text-body2 text-secondary')
        else:
            ui.label('No mCODE mappings found').classes('text-gray-500 italic text-body2')
        
        # Display structured data
        structured_data = study_data.get('structured_data', {})
        if structured_data:
            ui.label('Structured Data').classes('text-h6 text-primary mt-4')
            with ui.card().classes('w-full'):
                ui.label('Generated mCODE Resources:').classes('font-weight-bold text-primary')
                entry_list = structured_data.get('entry', [])
                if entry_list:
                    ui.label(f"Generated {len(entry_list)} resources").classes('text-body2')
                    # Show count of each resource type
                    resource_types = {}
                    for entry in entry_list:
                        resource = entry.get('resource', {})
                        resource_type = resource.get('resourceType', 'Unknown')
                        resource_types[resource_type] = resource_types.get(resource_type, 0) + 1
                    for resource_type, count in resource_types.items():
                        ui.label(f"• {resource_type}: {count}").classes('text-body2')
                else:
                    ui.label('No resources generated').classes('text-gray-500 italic text-body2')
        
        # Display validation results
        validation = study_data.get('validation', {})
        if validation:
            ui.label('mCODE Validation').classes('text-h6 text-primary mt-4')
            with ui.card().classes('w-full'):
                valid = validation.get('valid', False)
                compliance_score = validation.get('compliance_score', 0)
                errors = validation.get('errors', [])
                
                with ui.column().classes('w-full'):
                    ui.label(f"Valid: {'Yes' if valid else 'No'}").classes('text-body2')
                    ui.label(f"Compliance Score: {compliance_score:.2f}").classes('text-body2')
                    if errors:
                        ui.label('Validation Errors:').classes('font-weight-bold text-primary mt-2')
                        with ui.column().classes('w-full pl-4'):
                            for error in errors[:5]:  # Show first 5 errors
                                ui.label(f"• {error}").classes('text-body2 text-negative')
                            if len(errors) > 5:
                                ui.label(f'... and {len(errors) - 5} more errors').classes('text-body2 text-secondary')
                    else:
                        ui.label('No validation errors').classes('text-body2 text-positive')

def generate_condition_colors(conditions):
    """Generate consistent colors for conditions."""
    # Predefined color palette that matches the visual style
    colors = [
        'blue', 'orange', 'green', 'red', 'purple',
        'brown', 'pink', 'grey', 'lime', 'cyan'
    ]
    
    condition_colors = {}
    for i, condition in enumerate(conditions):
        condition_colors[condition] = colors[i % len(colors)]
    
    return condition_colors


def generate_condition_colors_for_current_data():
    """Generate condition colors based on current search results."""
    global condition_colors
    
    # Collect all conditions from current results
    all_conditions = []
    for study in current_search_results:
        protocol_section = study.get('protocolSection', {})
        conditions_module = protocol_section.get('conditionsModule', {})
        conditions = conditions_module.get('conditions', [])
        all_conditions.extend(conditions)
    
    # Get unique conditions
    unique_conditions = list(set(all_conditions))
    
    # Generate colors for these conditions
    condition_colors = generate_condition_colors(unique_conditions)


def get_condition_badge_color(condition):
    """Get the appropriate badge color for a condition."""
    return condition_colors.get(condition, 'blue')


def get_status_badge_color(status):
    """Get the appropriate badge color for a status."""
    return STATUS_COLORS.get(status, 'purple')


def create_trial_cards():
    """Create trial cards for each study."""
    # Clear previous content
    results_container.clear()
    
    with results_container:
        if not current_search_results:
            ui.label('No studies found. Try a different search term.')
            return
            
        # Use grid layout for consistent card alignment
        with ui.grid(columns=1).classes('w-full gap-4'):
            for study in current_search_results:
                protocol_section = study.get('protocolSection', {})
                identification_module = protocol_section.get('identificationModule', {})
                status_module = protocol_section.get('statusModule', {})
                description_module = protocol_section.get('descriptionModule', {})
                conditions_module = protocol_section.get('conditionsModule', {})
                
                # Create card for each trial with improved styling
                with ui.card().classes('w-full shadow-md hover:shadow-lg transition-shadow'):
                    # Header with title and NCT ID
                    with ui.row().classes('w-full justify-between items-start mb-2'):
                        ui.label(identification_module.get('briefTitle', 'N/A')).classes('text-lg font-weight-bold')
                        ui.label(identification_module.get('nctId', 'N/A')).classes('text-caption text-secondary')
                    
                    # Status badge with improved styling
                    status = status_module.get('overallStatus', 'N/A')
                    ui.badge(status).props(f'color={get_status_badge_color(status)}').classes('mb-2')
                    
                    # Brief description with better formatting
                    brief_summary = description_module.get('briefSummary', 'No description available')
                    if brief_summary and brief_summary != 'No description available':
                        ui.markdown(brief_summary[:200] + '...' if len(brief_summary) > 200 else brief_summary).classes('text-body2')
                    else:
                        ui.label('No description available').classes('text-gray-500 italic text-body2')
                    
                    # Expansion panel for more details with mCODE focus
                    with ui.expansion('Study Details', icon='info').classes('w-full mt-2'):
                        with ui.card().classes('w-full'):
                            # Conditions with better organization
                            conditions = conditions_module.get('conditions', [])
                            if conditions:
                                ui.label('Conditions:').classes('font-weight-bold text-primary')
                                with ui.row():
                                    for condition in conditions[:5]:  # Show first 5 conditions
                                        ui.badge(condition).props(f'color={get_condition_badge_color(condition)}')
                                    if len(conditions) > 5:
                                        ui.badge(f'+{len(conditions) - 5} more').props('color=grey')
                            else:
                                ui.label('No conditions specified').classes('text-gray-500 italic')
                    
                    # Action buttons with better styling
                    with ui.row().classes('w-full justify-end mt-3'):
                        ui.button('View Details', icon='visibility',
                                 on_click=lambda nct_id=identification_module.get('nctId', ''): show_study_details(nct_id)).props('outline color=primary')


def create_trial_list():
    """Create a list view of trials."""
    # Clear previous content
    results_container.clear()
    
    with results_container:
        if not current_search_results:
            ui.label('No studies found. Try a different search term.')
            return
            
        # Use grid layout for consistent list alignment
        with ui.grid(columns=1).classes('w-full gap-3'):
            for study in current_search_results:
                protocol_section = study.get('protocolSection', {})
                identification_module = protocol_section.get('identificationModule', {})
                status_module = protocol_section.get('statusModule', {})
                description_module = protocol_section.get('descriptionModule', {})
                conditions_module = protocol_section.get('conditionsModule', {})
                
                # Create list item for each trial with improved styling
                with ui.card().classes('w-full shadow-sm hover:shadow-md transition-shadow'):
                    with ui.row().classes('w-full items-center justify-between'):
                        with ui.column():
                            with ui.row().classes('items-center gap-2 mb-1'):
                                ui.label(identification_module.get('briefTitle', 'N/A')).classes('font-weight-bold')
                                ui.label(identification_module.get('nctId', 'N/A')).classes('text-caption text-secondary')
                            
                            # Status badge with improved styling
                            status = status_module.get('overallStatus', 'N/A')
                            ui.badge(status).props(f'color={get_status_badge_color(status)}').classes('mb-1')
                            
                            # Brief description with better formatting
                            brief_summary = description_module.get('briefSummary', 'No description available')
                            if brief_summary and brief_summary != 'No description available':
                                ui.label(brief_summary[:100] + '...' if len(brief_summary) > 100 else brief_summary).classes('text-body2 text-secondary')
                            else:
                                ui.label('No description available').classes('text-body2 text-gray-500 italic')
                            
                            # Conditions as badges
                            conditions = conditions_module.get('conditions', [])
                            if conditions:
                                with ui.row().classes('mt-1'):
                                    for condition in conditions[:2]:  # Show only first 2 in list view
                                        ui.badge(condition).props(f'color={get_condition_badge_color(condition)}').classes('text-caption')
                            else:
                                ui.label('No conditions').classes('text-caption text-gray-500')
                        
                        # Action button with better styling
                        ui.button('View Details', icon='visibility',
                                 on_click=lambda nct_id=identification_module.get('nctId', ''): show_study_details(nct_id)).props('outline color=primary')


def create_trial_table():
    """Create a table view of trials with badges for status and conditions."""
    # Clear previous content
    results_container.clear()
    
    with results_container:
        if not current_search_results:
            ui.label('No studies found. Try a different search term.')
            return
            
        # Create table with proper column alignment and improved styling
        columns = [
            {'name': 'nct_id', 'label': 'NCT ID', 'field': 'nct_id', 'sortable': True, 'align': 'left'},
            {'name': 'title', 'label': 'Title', 'field': 'title', 'sortable': True, 'align': 'left'},
            {'name': 'description', 'label': 'Description', 'field': 'description', 'align': 'left'},
            {'name': 'status', 'label': 'Status', 'field': 'status', 'sortable': True, 'align': 'left'},
            {'name': 'conditions', 'label': 'Conditions', 'field': 'conditions', 'align': 'left'},
            {'name': 'actions', 'label': 'Actions', 'field': 'actions', 'align': 'left'}
        ]
        
        rows = []
        for study in current_search_results:
            protocol_section = study.get('protocolSection', {})
            identification_module = protocol_section.get('identificationModule', {})
            status_module = protocol_section.get('statusModule', {})
            description_module = protocol_section.get('descriptionModule', {})
            conditions_module = protocol_section.get('conditionsModule', {})
            
            # Status for table
            status = status_module.get('overallStatus', 'N/A')
            
            # Brief description for table
            brief_summary = description_module.get('briefSummary', 'No description available')
            description = brief_summary[:100] + '...' if len(brief_summary) > 100 else brief_summary
            
            # Conditions for table
            conditions = conditions_module.get('conditions', [])
            conditions_display = ", ".join(conditions) if conditions else ""
            
            # Get status color
            status_color = STATUS_COLORS.get(status, 'purple')
            
            # Get condition colors
            condition_colors = {}
            for condition in conditions:
                condition_colors[condition] = get_condition_badge_color(condition)
            
            rows.append({
                'nct_id': identification_module.get('nctId', 'N/A'),
                'title': identification_module.get('briefTitle', 'N/A'),
                'description': description,
                'status': status,
                'status_color': status_color,
                'conditions': conditions_display,
                'condition_colors': condition_colors,
                'actions': identification_module.get('nctId', 'N/A')
            })
        
        # Create table with NiceGUI props for styling
        table = ui.table(columns=columns, rows=rows, pagination=False).classes('w-full')
        table.props('wrap-cells dense flat bordered')
        
        # Add color to status column using NiceGUI badge
        table.add_slot('body-cell-status', '''
            <q-td :props="props">
                <q-badge v-if="props.row.status" :color="props.row.status_color" class="text-weight-bold">{{ props.row.status }}</q-badge>
                <div v-else>N/A</div>
            </q-td>
        ''')
        
        # Add badges to conditions column
        table.add_slot('body-cell-conditions', '''
            <q-td :props="props">
                <div v-if="props.row.conditions">
                    <q-badge
                        v-for="(condition, index) in props.row.conditions.split(', ')"
                        :key="index"
                        :color="props.row.condition_colors[condition]"
                        class="q-mr-xs"
                    >
                        {{ condition }}
                    </q-badge>
                </div>
                <div v-else>No conditions</div>
            </q-td>
        ''')
        
        # Add action buttons to the table
        table.add_slot('body-cell-actions', '''
            <q-td :props="props">
                <q-btn flat size="sm" icon="visibility" @click="() => $parent.$emit('view-details', props.row.actions)" color="primary" title="View Details"></q-btn>
                <q-btn flat size="sm" icon="medical_services" @click="() => $parent.$emit('view-mcode', props.row.actions)" color="secondary" title="View mCODE Analysis"></q-btn>
            </q-td>
        ''')
        
        # Add event handlers for view details and mCODE analysis
        table.on('view-details', lambda e: show_study_details(e.args))
        table.on('view-mcode', lambda e: show_mcode_analysis(e.args))
        
        # No JavaScript needed - using Python functions for color mapping
        pass


def update_results_view():
    """Update results display based on current view mode."""
    if current_view == "cards":
        create_trial_cards()
    elif current_view == "list":
        create_trial_list()
    elif current_view == "table":
        create_trial_table()


def change_view(view):
    """Change the view mode."""
    global current_view
    current_view = view
    update_results_view()


def get_status_distribution(studies):
    """Get distribution of study statuses."""
    statuses = []
    for study in studies:
        protocol_section = study.get('protocolSection', {})
        status_module = protocol_section.get('statusModule', {})
        status = status_module.get('overallStatus', 'Unknown')
        statuses.append(status)
    return Counter(statuses)

def get_conditions_distribution(studies):
    """Get distribution of conditions."""
    conditions = []
    for study in studies:
        protocol_section = study.get('protocolSection', {})
        conditions_module = protocol_section.get('conditionsModule', {})
        conditions.extend(conditions_module.get('conditions', []))
    return Counter(conditions)

# UI Components
def create_search_interface():
    """Create the streamlined search interface."""
    
    with ui.column().classes('w-full p-4 max-w-6xl mx-auto'):
        # Header section with improved visual hierarchy
        with ui.column().classes('w-full items-center mb-6'):
            ui.label('mCODE Clinical Trial Search').classes('text-h3 text-primary font-bold')
            ui.markdown('Search for clinical trials and analyze mCODE data with enhanced visualization').classes('text-subtitle1 text-center text-secondary')
        
        # Search controls card with better organization
        with ui.card().classes('w-full mb-6 shadow-lg'):
            with ui.column().classes('w-full'):
                ui.label('Search Controls').classes('text-h6 text-primary')
                with ui.row().classes('w-full items-center gap-4'):
                    search_input = ui.input(
                        label='Search Expression',
                        placeholder='Enter search terms (e.g., "breast cancer")',
                        value=search_expression
                    ).classes('flex-grow')
                    
                    def update_search():
                        global search_expression, current_page
                        search_expression = search_input.value
                        current_page = 1
                        if fetch_search_results():
                            update_results_display()
                            create_pagination_controls()
                    
                    ui.button('Search', icon='search', on_click=update_search).tooltip('Search for clinical trials').props('color=primary')
                
                with ui.row().classes('w-full items-center mt-4 gap-4'):
                    # Results per page selector with better labeling
                    with ui.column():
                        ui.label('Results per page').classes('text-caption text-secondary')
                        results_per_page_select = ui.select([10, 20, 50, 100], value=results_per_page).tooltip('Select number of results per page')
                        
                        def update_results_per_page():
                            global results_per_page, current_page, page_tokens
                            results_per_page = results_per_page_select.value
                            current_page = 1  # Reset to first page when changing page size
                            page_tokens = [None]  # Reset page tokens when changing page size
                            if fetch_search_results():
                                update_results_display()
                                create_pagination_controls()
                        
                        results_per_page_select.on('update:model-value', lambda: update_results_per_page())
        
        # Results visualization section with clear headings
        with ui.card().classes('w-full mb-6 shadow-lg'):
            ui.label('Study Distribution').classes('text-h6 text-primary')
            # Results info and visualizations
            with ui.grid(columns=2).classes('w-full gap-6 mt-4'):
                with ui.card().classes('w-full'):
                    ui.label('Status Distribution').classes('text-subtitle1 font-weight-bold')
                    status_chart_container = ui.column().classes('w-full')
                
                with ui.card().classes('w-full'):
                    ui.label('Top Conditions').classes('text-subtitle1 font-weight-bold')
                    conditions_chart_container = ui.column().classes('w-full')
        
        # Results section with improved organization
        with ui.card().classes('w-full shadow-lg'):
            # Results header with info and view toggle
            with ui.row().classes('w-full items-center justify-between mb-4'):
                with ui.column():
                    ui.label('Search Results').classes('text-h6 text-primary')
                    results_info_label = ui.label('').classes('text-subtitle2 text-secondary')
                
                # View toggle with better visual design
                with ui.row().classes('items-center gap-2'):
                    ui.button(icon='view_module', on_click=lambda: change_view('cards')).props(
                        'flat round' + (' color=primary' if current_view == 'cards' else '')
                    ).tooltip('Card View')
                    ui.button(icon='view_list', on_click=lambda: change_view('list')).props(
                        'flat round' + (' color=primary' if current_view == 'list' else '')
                    ).tooltip('List View')
                    ui.button(icon='table_chart', on_click=lambda: change_view('table')).props(
                        'flat round' + (' color=primary' if current_view == 'table' else '')
                    ).tooltip('Table View')
        
            # Results container
            global results_container
            results_container = ui.column().classes('w-full')
        
        # Pagination controls with better styling
        pagination_controls_container = ui.column().classes('w-full mt-6')
        
        def create_pagination_controls():
            """Create or recreate the pagination controls."""
            pagination_controls_container.clear()
            with pagination_controls_container:
                with ui.row().classes('w-full justify-center items-center gap-2'):
                    first_button = ui.button('First', on_click=lambda: change_page(1)).tooltip('Go to first page').props('flat color=primary')
                    prev_button = ui.button('Previous', on_click=lambda: change_page(max(1, current_page - 1))).tooltip('Go to previous page').props('flat color=primary')
                    page_label = ui.label(f'Page {current_page} of {total_pages}').tooltip('Current page information').classes('text-subtitle2 mx-2')
                    next_button = ui.button('Next', on_click=lambda: change_page(min(total_pages, current_page + 1))).tooltip('Go to next page').props('flat color=primary')
                    last_button = ui.button('Last', on_click=lambda: change_page(total_pages)).tooltip('Go to last page').props('flat color=primary')
                    
                    # Set button states
                    if current_page == 1:
                        first_button.disable()
                        prev_button.disable()
                    else:
                        first_button.enable()
                        prev_button.enable()
                    
                    if current_page == total_pages:
                        next_button.disable()
                    else:
                        next_button.enable()
                    
                    # Disable last button unless total_pages is the next page
                    if current_page == total_pages or (total_pages > 1 and total_pages != current_page + 1):
                        last_button.disable()
                    else:
                        last_button.enable()
        
        # Initialize pagination controls
        create_pagination_controls()
        
        def update_results_display():
            """Update the results display with current data."""
            # Generate condition colors for current data
            generate_condition_colors_for_current_data()
            
            # Update results view based on current view mode
            update_results_view()
            
            # Update info label
            if not current_search_results:
                results_info_label.set_text('No studies found')
                update_charts(Counter(), Counter())
                return
            
            start_result = (current_page - 1) * results_per_page + 1
            end_result = min(current_page * results_per_page, total_studies)
            results_info_label.set_text(f'Showing {start_result}-{end_result} of {total_studies} studies')
            
            # Update charts
            status_dist = get_status_distribution(current_search_results)
            conditions_dist = get_conditions_distribution(current_search_results)
            update_charts(status_dist, conditions_dist)
        
        def update_charts(status_dist, conditions_dist):
            """Update the charts with current data."""
            # Clear previous charts
            status_chart_container.clear()
            conditions_chart_container.clear()
            
            # Status distribution chart with improved styling
            if status_dist:
                with status_chart_container:
                    ui.echart({
                        'title': {
                            'text': 'Study Status Distribution',
                            'textStyle': {'fontSize': 16, 'fontWeight': 'bold'}
                        },
                        'tooltip': {'trigger': 'item'},
                        'series': [{
                            'type': 'pie',
                            'data': [{'value': count, 'name': status} for status, count in status_dist.items()],
                            'label': {'show': True, 'formatter': '{b}: {c} ({d}%)'},
                            'emphasis': {
                                'itemStyle': {
                                    'shadowBlur': 10,
                                    'shadowOffsetX': 0,
                                    'shadowColor': 'rgba(0, 0, 0, 0.5)'
                                }
                            }
                        }]
                    }).classes('w-full h-64')
            
            # Conditions distribution chart with improved styling
            if conditions_dist:
                # Get top 10 conditions
                top_conditions = dict(conditions_dist.most_common(10))
                if top_conditions:
                    with conditions_chart_container:
                        ui.echart({
                            'title': {
                                'text': 'Top Conditions',
                                'textStyle': {'fontSize': 16, 'fontWeight': 'bold'}
                            },
                            'tooltip': {'trigger': 'axis'},
                            'xAxis': {
                                'type': 'category',
                                'data': list(top_conditions.keys()),
                                'axisLabel': {'rotate': 45, 'fontSize': 12}
                            },
                            'yAxis': {
                                'type': 'value',
                                'name': 'Number of Studies',
                                'nameLocation': 'middle',
                                'nameGap': 30
                            },
                            'series': [{
                                'data': [{'value': count, 'name': condition} for condition, count in top_conditions.items()],
                                'type': 'bar',
                                'itemStyle': {
                                    'color': '#4285f4'
                                },
                                'emphasis': {
                                    'itemStyle': {
                                        'color': '#3367d6'
                                    }
                                }
                            }]
                        }).classes('w-full h-64')
        
        def change_page(page):
            """Change to a specific page."""
            global current_page
            logger.info(f"Changing page from {current_page} to {page}")
            current_page = page
            logger.info(f"Current page is now {current_page}")
            if fetch_search_results():
                logger.info(f"Successfully fetched results for page {current_page}, updating display")
                update_results_display()
                create_pagination_controls()
                logger.info(f"Finished updating display for page {current_page}")
            else:
                # If fetch failed, still update pagination controls to reflect current state
                logger.info(f"Failed to fetch results for page {current_page}, updating pagination controls")
                create_pagination_controls()
        
        
        # Initialize
        if fetch_search_results():
            update_results_display()
            create_pagination_controls()

def show_study_details(nct_id: str):
    """Show detailed information for a specific study."""
    success = fetch_study_details(nct_id)
    
    if success and current_study_details:
        # Create a maximized dialog to show the study details
        with ui.dialog(value=True).classes('w-full h-full') as dialog, ui.card().classes('w-full h-full'):
            with ui.scroll_area().classes('w-full h-full'):
                with ui.column().classes('w-full'):
                    # Basic info with improved styling
                    protocol_section = current_study_details.get('protocolSection', {})
                    identification_module = protocol_section.get('identificationModule', {})
                    status_module = protocol_section.get('statusModule', {})
                    description_module = protocol_section.get('descriptionModule', {})
                    conditions_module = protocol_section.get('conditionsModule', {})
                    eligibility_module = protocol_section.get('eligibilityModule', {})
                    
                    with ui.card().classes('w-full'):
                        with ui.row().classes('w-full justify-between items-center mb-2'):
                            ui.label(identification_module.get('briefTitle', 'N/A')).classes('text-h6 font-weight-bold')
                            ui.label(identification_module.get('nctId', 'N/A')).classes('text-subtitle2 text-secondary')
                        
                        # Status badge with tooltip and improved styling
                        status = status_module.get('overallStatus', 'N/A')
                        status_color = {
                            'ACTIVE_NOT_RECRUITING': 'blue',
                            'RECRUITING': 'green',
                            'COMPLETED': 'green',
                            'TERMINATED': 'red',
                            'WITHDRAWN': 'red',
                            'UNKNOWN': 'gray'
                        }.get(status, 'gray')
                        ui.badge(status, color=status_color).tooltip(f'Study status: {status}').classes('self-end')
                    
                    # Process criteria toggle
                    process_criteria_toggle = ui.toggle(['Basic View', 'Process Criteria'], value='Basic View').classes('mb-4')
                    
                    # Expandable sections with improved styling
                    with ui.expansion('Description', icon='description').classes('w-full'):
                        brief_summary = description_module.get('briefSummary', '')
                        if brief_summary:
                            ui.markdown(brief_summary).classes('text-body2')
                        else:
                            ui.label('No description available').classes('text-gray-500 italic')
                    
                    with ui.expansion('Conditions', icon='local_hospital').classes('w-full'):
                        conditions = conditions_module.get('conditions', [])
                        if conditions:
                            with ui.column().classes('w-full h-full'):
                                for condition in conditions:
                                    ui.label(f'• {condition}').classes('text-body2')
                        else:
                            ui.label('No conditions specified').classes('text-gray-500 italic')
                    
                    with ui.expansion('Dates', icon='calendar_today').classes('w-full'):
                        with ui.grid(columns=2).classes('w-full'):
                            ui.label('Start Date:').classes('font-weight-bold')
                            ui.label(status_module.get('startDateStruct', {}).get('date', 'N/A')).classes('text-body2')
                            
                            ui.label('Completion Date:').classes('font-weight-bold')
                            ui.label(status_module.get('completionDateStruct', {}).get('date', 'N/A')).classes('text-body2')
                    
                    # Eligibility criteria section
                    with ui.expansion('Eligibility Criteria', icon='rule').classes('w-full'):
                        eligibility_criteria = eligibility_module.get('eligibilityCriteria', '')
                        if eligibility_criteria:
                            # Show raw criteria in a scrollable area
                            with ui.scroll_area().classes('h-40'):
                                ui.markdown(eligibility_criteria).classes('text-body2')
                        else:
                            ui.label('No eligibility criteria available').classes('text-gray-500 italic')
                    
                    # mCODE analysis section
                    mcode_expansion = ui.expansion('mCODE Analysis', icon='medical_services').classes('w-full')
                    mcode_container = None
                    
                    def update_mcode_view():
                        nonlocal mcode_container
                        # Clear previous content
                        if mcode_container:
                            mcode_container.clear()
                        
                        # Create new container
                        with mcode_expansion:
                            mcode_container = ui.column().classes('w-full')
                            with mcode_container:
                                if process_criteria_toggle.value == 'Process Criteria':
                                    ui.label('Processing criteria with NLP engine...').classes('text-body2 text-info')
                                    # Force UI update
                                    ui.run_javascript('''
                                        return new Promise(resolve => setTimeout(resolve, 100));
                                    ''')
                                    
                                    # Extract mCODE data with processing
                                    logger.info(f"Extracting mCODE data for study details with criteria processing")
                                    mcode_data = extract_mcode_data(current_study_details)
                                    logger.info(f"mCODE data extracted: {mcode_data}")
                                    
                                    if mcode_data:
                                        display_mcode_visualization(mcode_data)
                                    else:
                                        ui.label('Failed to process mCODE data').classes('text-body2 text-negative')
                                else:
                                    ui.label('Switch to "Process Criteria" view to see mCODE analysis').classes('text-body2 text-info')
                    
                    # Set up the initial mCODE view
                    update_mcode_view()
                    
                    # Update mCODE view when toggle changes
                    process_criteria_toggle.on('update:model-value', lambda: update_mcode_view())
                    
                    # Export button with improved styling
                    def export_study():
                        if current_study_details:
                            filename = f"{nct_id}_details.json"
                            export_to_json(current_study_details, filename)
                            dialog.close()
                    
                    with ui.row().classes('w-full justify-end mt-4'):
                        ui.button('Export Study Details', icon='download', on_click=export_study).tooltip('Export study details to JSON file').props('color=primary')
                        ui.button('Close', on_click=dialog.close).tooltip('Close this dialog').props('outline color=secondary')
            
            dialog.open()
    else:
        ui.notify('Failed to load study details', type='negative')


def show_mcode_analysis(nct_id: str):
    """Show mCODE analysis for a specific study in a maximized popup."""
    success = fetch_study_details(nct_id)
    
    if success and current_study_details:
        # Create a maximized dialog to show the mCODE analysis
        with ui.dialog(value=True).classes('w-full h-full') as dialog, ui.card().classes('w-full h-full'):
            with ui.scroll_area().classes('w-full h-full'):
                with ui.column().classes('w-full'):
                    # Header with study info
                    with ui.row().classes('w-full justify-between items-center mb-4'):
                        ui.label(f'mCODE Analysis: {nct_id}').classes('text-h5 font-weight-bold')
                    
                    # Process criteria toggle
                    process_criteria_toggle = ui.toggle(['Basic View', 'Process Criteria'], value='Process Criteria').classes('mb-4')
                    
                    # mCODE visualization container
                    mcode_container = ui.column().classes('w-full')
                    
                    def update_mcode_view():
                        # Clear previous content
                        mcode_container.clear()
                        
                        with mcode_container:
                            if process_criteria_toggle.value == 'Process Criteria':
                                ui.label('Processing criteria with NLP engine...').classes('text-body2 text-info')
                                # Force UI update
                                ui.run_javascript('''
                                    return new Promise(resolve => setTimeout(resolve, 100));
                                ''')
                                
                                # Extract mCODE data with processing
                                logger.info(f"Extracting mCODE data for analysis with criteria processing")
                                mcode_data = extract_mcode_data(current_study_details)
                                logger.info(f"mCODE data for analysis: {mcode_data}")
                                
                                if mcode_data:
                                    display_mcode_visualization(mcode_data)
                                else:
                                    ui.label('Failed to process mCODE data').classes('text-body2 text-negative')
                            else:
                                ui.label('Switch to "Process Criteria" view to see mCODE analysis').classes('text-body2 text-info')
                    
                    # Set up the initial mCODE view
                    update_mcode_view()
                    
                    # Update mCODE view when toggle changes
                    process_criteria_toggle.on('update:model-value', lambda: update_mcode_view())
                    
                    with ui.row().classes('w-full justify-end mt-4'):
                        ui.button('Close', on_click=dialog.close).props('outline color=secondary')
            
            dialog.open()
    else:
        ui.notify('Failed to load study details for mCODE analysis', type='negative')

# Main UI - Simplified to only include search functionality
with ui.column().classes('w-full'):
    create_search_interface()

# Run the app
ui.run(title='Clinical Trial Search', port=8084)