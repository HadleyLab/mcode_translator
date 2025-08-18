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
    calculate_total_studies
)
from collections import Counter
import logging

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
            fields=["NCTId", "BriefTitle", "Condition", "OverallStatus"],
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
    
    with ui.column().classes('w-full p-4'):
        ui.label('Clinical Trial Search').classes('text-h4 text-center')
        ui.markdown('Search for clinical trials and view results with enhanced visualization').classes('text-center')
        
        # Search controls
        with ui.row().classes('w-full items-center'):
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
            
            ui.button('Search', icon='search', on_click=update_search).tooltip('Search for clinical trials')
            
            # Results per page selector
            ui.label('Results per page:').tooltip('Number of results to display per page')
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
        
        # Results info and visualizations
        with ui.grid(columns=2).classes('w-full gap-4 mt-4'):
            with ui.card().classes('w-full'):
                status_chart_container = ui.column().classes('w-full')
            
            with ui.card().classes('w-full'):
                conditions_chart_container = ui.column().classes('w-full')
        
        # Results info
        results_info_label = ui.label('').classes('text-subtitle2')
        
        # Results table
        results_table = ui.table(columns=[], rows=[], pagination=False).classes('w-full mt-4')
        results_table.props('wrap-cells dense')
        
        # Pagination controls
        pagination_controls_container = ui.column().classes('w-full mt-4')
        
        def create_pagination_controls():
            """Create or recreate the pagination controls."""
            pagination_controls_container.clear()
            with pagination_controls_container:
                with ui.row().classes('w-full justify-center items-center'):
                    first_button = ui.button('First', on_click=lambda: change_page(1)).tooltip('Go to first page')
                    prev_button = ui.button('Previous', on_click=lambda: change_page(max(1, current_page - 1))).tooltip('Go to previous page')
                    page_label = ui.label(f'Page {current_page} of {total_pages}').tooltip('Current page information')
                    next_button = ui.button('Next', on_click=lambda: change_page(min(total_pages, current_page + 1))).tooltip('Go to next page')
                    last_button = ui.button('Last', on_click=lambda: change_page(total_pages)).tooltip('Go to last page')
                    
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
            """Update the results table with current data."""
            if not current_search_results:
                results_table.columns = []
                results_table.rows = []
                results_info_label.set_text('No studies found')
                update_charts(Counter(), Counter())
                return
            
            # Set columns with width constraints
            results_table.columns = [
                {'name': 'nct_id', 'label': 'NCT ID', 'field': 'nct_id', 'sortable': True, 'style': 'width: 15%'},
                {'name': 'title', 'label': 'Title', 'field': 'title', 'sortable': True, 'style': 'width: 30%'},
                {'name': 'conditions', 'label': 'Conditions', 'field': 'conditions', 'style': 'width: 30%'},
                {'name': 'status', 'label': 'Status', 'field': 'status', 'sortable': True, 'style': 'width: 15%'},
                {'name': 'actions', 'label': 'Actions', 'field': 'actions', 'style': 'width: 10%'}
            ]
            
            # Set rows
            rows = []
            for study in current_search_results:
                protocol_section = study.get('protocolSection', {})
                identification_module = protocol_section.get('identificationModule', {})
                status_module = protocol_section.get('statusModule', {})
                conditions_module = protocol_section.get('conditionsModule', {})
                nct_id = identification_module.get('nctId', 'N/A')
                
                # Get status for badge
                status = status_module.get('overallStatus', 'N/A')
                
                # Get status color
                status_color = {
                    'ACTIVE_NOT_RECRUITING': 'blue',
                    'RECRUITING': 'green',
                    'COMPLETED': 'grey',
                    'TERMINATED': 'red',
                    'WITHDRAWN': 'orange',
                    'UNKNOWN': 'gray'
                }.get(status, 'purple')
                
                rows.append({
                    'nct_id': nct_id,
                    'title': identification_module.get('briefTitle', 'N/A'),
                    'conditions': ', '.join(conditions_module.get('conditions', [])[:3]),
                    'status': status,
                    'status_color': status_color,
                    'actions': nct_id  # Pass NCT ID for the action button
                })
            
            results_table.rows = rows
            
            async def handle_row_click(e):
                """Handle table row click to show study details."""
                try:
                    # Log the event structure for debugging
                    logger.info(f"Row click event args: {e.args}")
                    
                    # Get the clicked row data - handle different event structures
                    if isinstance(e.args, list) and len(e.args) > 1:
                        # Newer NiceGUI versions use e.args[1] for row data
                        row_data = e.args[1]
                    elif isinstance(e.args, dict) and 'args' in e.args:
                        # Some versions nest args in a dict
                        row_data = e.args['args'][1] if len(e.args['args']) > 1 else e.args['args'][0]
                    else:
                        # Fallback to first item
                        row_data = e.args[0] if isinstance(e.args, list) and len(e.args) > 0 else e.args
                    
                    # Extract NCT ID from row data
                    if isinstance(row_data, dict) and 'nct_id' in row_data:
                        nct_id = row_data['nct_id']
                    else:
                        # Fallback - try to find NCT ID in the row data structure
                        nct_id = None
                        if isinstance(row_data, dict):
                            nct_id = row_data.get('nct_id') or row_data.get('key') or row_data.get('id')
                    
                    if nct_id:
                        logger.info(f"Showing details for study: {nct_id}")
                        show_study_details(nct_id)
                    else:
                        logger.error(f"Could not extract NCT ID from row data: {row_data}")
                        ui.notify("Failed to extract study ID", type='negative')
                except Exception as ex:
                    logger.error(f"Error handling row click: {str(ex)}")
                    ui.notify("Failed to show study details", type='negative')
            
            results_table.on('rowClick', handle_row_click)
            
            # Log the rows being set for debugging
            logger.info(f"Setting table rows for page {current_page}: {len(rows)} rows")
            for i, row in enumerate(rows):
                logger.info(f"  Row {i+1}: {row['nct_id']} - {row['title']}")
            
            # Force update the table
            results_table.update()
            
            # Add custom slot for status column to show badges
            results_table.add_slot('body-cell-status', '''
                <q-td :props="props">
                    <q-badge
                        :color="props.row.status_color"
                        :label="props.row.status"
                    />
                </q-td>
            ''')
            
            # Add custom slot for actions column to show a button
            results_table.add_slot('body-cell-actions', '''
                <q-td :props="props">
                    <q-btn
                        flat
                        size="sm"
                        label="View Details"
                        icon="visibility"
                        @click="() => window.showStudyDetails(props.row.actions)"
                    />
                </q-td>
            ''')
            
            # Add JavaScript function to show study details
            results_table.add_slot('bottom', '''
                <script>
                    window.showStudyDetails = function(nctId) {
                        // This will be handled by NiceGUI's native event system
                        document.dispatchEvent(new CustomEvent('show-studies', {detail: nctId}));
                    }
                </script>
            ''')
            
            # Add event handler for view details
            # results_table.on('show-studies', lambda e: show_study_details(e.args))  # Removed duplicate event handler
            
            # Update info label
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
            
            # Status distribution chart
            if status_dist:
                with status_chart_container:
                    ui.echart({
                        'title': {'text': 'Study Status Distribution'},
                        'tooltip': {'trigger': 'item'},
                        'series': [{
                            'type': 'pie',
                            'data': [{'value': count, 'name': status} for status, count in status_dist.items()],
                            'label': {'show': True, 'formatter': '{b}: {c} ({d}%)'}
                        }]
                    }).classes('w-full h-64')
            
            # Conditions distribution chart
            if conditions_dist:
                # Get top 10 conditions
                top_conditions = dict(conditions_dist.most_common(10))
                if top_conditions:
                    with conditions_chart_container:
                        ui.echart({
                            'title': {'text': 'Top Conditions'},
                            'tooltip': {'trigger': 'axis'},
                            'xAxis': {
                                'type': 'category',
                                'data': list(top_conditions.keys())
                            },
                            'yAxis': {'type': 'value'},
                            'series': [{
                                'data': [{'value': count, 'name': condition} for condition, count in top_conditions.items()],
                                'type': 'bar'
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
        # Create a dialog to show the study details
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-4xl'):
            with ui.scroll_area().classes('h-96 w-full'):
                with ui.column().classes('w-full'):
                    # Basic info
                    protocol_section = current_study_details.get('protocolSection', {})
                    identification_module = protocol_section.get('identificationModule', {})
                    status_module = protocol_section.get('statusModule', {})
                    description_module = protocol_section.get('descriptionModule', {})
                    conditions_module = protocol_section.get('conditionsModule', {})
                    
                    with ui.card().classes('w-full'):
                        with ui.row().classes('w-full justify-between items-center'):
                            ui.label(identification_module.get('briefTitle', 'N/A')).classes('text-h6')
                            ui.label(identification_module.get('nctId', 'N/A')).classes('text-subtitle2')
                        
                        # Status badge with tooltip
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
                    
                    # Expandable sections
                    with ui.expansion('Description', icon='description').classes('w-full'):
                        brief_summary = description_module.get('briefSummary', '')
                        if brief_summary:
                            ui.markdown(brief_summary)
                        else:
                            ui.label('No description available').classes('text-gray-500 italic')
                    
                    with ui.expansion('Conditions', icon='local_hospital').classes('w-full'):
                        conditions = conditions_module.get('conditions', [])
                        if conditions:
                            with ui.column():
                                for condition in conditions:
                                    ui.label(f'• {condition}')
                        else:
                            ui.label('No conditions specified').classes('text-gray-500 italic')
                    
                    with ui.expansion('Dates', icon='calendar_today').classes('w-full'):
                        with ui.grid(columns=2).classes('w-full'):
                            ui.label('Start Date:').classes('font-bold')
                            ui.label(status_module.get('startDateStruct', {}).get('date', 'N/A'))
                            
                            ui.label('Completion Date:').classes('font-bold')
                            ui.label(status_module.get('completionDateStruct', {}).get('date', 'N/A'))
                    
                    # Export button
                    def export_study():
                        if current_study_details:
                            filename = f"{nct_id}_details.json"
                            export_to_json(current_study_details, filename)
                            dialog.close()
                    
                    with ui.row().classes('w-full justify-end'):
                        ui.button('Export Study Details', icon='download', on_click=export_study).tooltip('Export study details to JSON file')
                        ui.button('Close', on_click=dialog.close).tooltip('Close this dialog')
            
            dialog.open()
    else:
        ui.notify('Failed to load study details', type='negative')

# Main UI - Simplified to only include search functionality
with ui.column().classes('w-full'):
    create_search_interface()

# Run the app
ui.run(title='Clinical Trial Search', port=8084)