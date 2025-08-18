"""
Enhanced Data Fetcher Dashboard for mCODE Translator
Clean, modular interface with search, visualizations, and view toggles.
"""
import json
import sys
import os
from functools import partial
from collections import Counter

from nicegui import ui

# Add src directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import data fetcher functions
from src.data_fetcher.fetcher import search_trials, get_full_study, calculate_total_studies


# Global state
search_expr = "breast cancer"
search_results = []
current_page = 1
actual_total_studies = 0  # Actual total from the API
total_pages = 1
search_input = None
results_container = None
pagination_controls = None
status_filter = "All"
condition_filter = "All"
current_view = "cards"  # Default view: cards, list, table
total_count_label = None

# Color mappings for consistency
STATUS_COLORS = {
    'Active, not recruiting': 'blue',
    'Recruiting': 'green',
    'Completed': 'grey',
    'Terminated': 'red',
    'Withdrawn': 'orange',
    'Suspended': 'yellow'
}

# Store condition colors for consistency between charts and badges
condition_colors = {}


def fetch_data():
    """Fetch data using the search expression."""
    global search_expr, search_results, current_page, actual_total_studies, total_pages
    
    try:
        # Reset to first page
        current_page = 1
        
        # Get current search expression from input
        search_expr = search_input.value
        
        # Get actual total count of studies
        try:
            stats = calculate_total_studies(search_expr)
            actual_total_studies = stats.get('total_studies', 0)
            total_pages = stats.get('total_pages', 1)
        except Exception as e:
            actual_total_studies = 0
            total_pages = 1
            ui.notify(f"Could not fetch total count: {str(e)}", type='warning')
        
        # Fetch data for current page
        fetch_page_data()
        
        # Update visualizations (only for first page to avoid too many API calls)
        if current_page == 1:
            update_visualizations()
        
        # Show notification
        if actual_total_studies > 0:
            ui.notify(f"Found {actual_total_studies} total studies", type='positive')
        else:
            ui.notify(f"Search completed", type='positive')
        
    except Exception as e:
        ui.notify(f"Error: {str(e)}", type='negative')


def reset_search():
    """Reset search to default state."""
    global search_expr
    search_expr = ""
    search_input.set_value("")
    fetch_data()


def fetch_page_data():
    """Fetch data for the current page."""
    global search_results, current_page, actual_total_studies, total_pages
    
    try:
        # Calculate min_rank for pagination
        min_rank = (current_page - 1) * results_per_page + 1
        print(f"DEBUG: fetch_page_data called, current_page={current_page}, min_rank={min_rank}, results_per_page={results_per_page}")
        print(f"DEBUG: Calculated min_rank: ({current_page} - 1) * {results_per_page} + 1 = {min_rank}")
        
        # Fetch data for current page with proper fields
        results = search_trials(
            search_expr=search_expr,
            fields=["NCTId", "BriefTitle", "Condition", "OverallStatus", "BriefSummary"],
            max_results=results_per_page,
            min_rank=min_rank
        )
        
        # Update results
        search_results = results.get('studies', [])
        print(f"DEBUG: fetch_page_data got {len(search_results)} results")
        
        # Generate condition colors for current data
        generate_condition_colors_for_current_data()
        
        # Update total count display
        update_total_count()
        
        # Update displayed results based on current view
        update_results_view()
        update_pagination_controls()
        
    except Exception as e:
        ui.notify(f"Error fetching page data: {str(e)}", type='negative')


def update_total_count():
    """Update the total count display."""
    if total_count_label:
        start_result = (current_page - 1) * results_per_page + 1
        end_result = min(current_page * results_per_page, actual_total_studies)
        
        if actual_total_studies > 0:
            total_count_label.set_text(f"Showing {start_result}-{end_result} of {actual_total_studies} total studies")
        else:
            total_count_label.set_text(f"Showing {len(search_results)} studies")


def update_results_view():
    """Update results display based on current view mode."""
    if current_view == "cards":
        create_trial_cards()
    elif current_view == "list":
        create_trial_list()
    elif current_view == "table":
        create_trial_table()


def generate_condition_colors_for_current_data():
    """Generate condition colors based on current search results."""
    global condition_colors
    
    # Collect all conditions from current results
    all_conditions = []
    for study in search_results:
        protocol_section = study.get('protocolSection', {})
        conditions_module = protocol_section.get('conditionsModule', {})
        conditions = conditions_module.get('conditions', [])
        all_conditions.extend(conditions)
    
    # Get unique conditions
    unique_conditions = list(set(all_conditions))
    
    # Generate colors for these conditions
    condition_colors = generate_condition_colors(unique_conditions)


def update_visualizations():
    """Update visualization components."""
    # Update status chart
    update_status_chart()
    
    # Update conditions chart
    update_conditions_chart()


def update_status_chart():
    """Update the status distribution chart."""
    # Clear previous chart
    status_chart_container.clear()
    
    # For visualization, we'll fetch a larger sample
    try:
        sample_results = search_trials(
            search_expr=search_expr,
            fields=["NCTId", "OverallStatus"],
            max_results=100
        )
        
        # Count status distribution
        status_counts = Counter()
        for study in sample_results.get('studies', []):
            protocol_section = study.get('protocolSection', {})
            status = protocol_section.get('statusModule', {}).get('overallStatus', 'Unknown')
            status_counts[status] += 1
        
        # Create chart data
        if status_counts:
            with status_chart_container:
                ui.echart({
                    'title': {'text': 'Study Status Distribution'},
                    'tooltip': {'trigger': 'item'},
                    'series': [{
                        'type': 'pie',
                        'data': [{'value': count, 'name': status} for status, count in status_counts.items()],
                        'label': {'show': True, 'formatter': '{b}: {c} ({d}%)'}
                    }]
                }).classes('w-full h-64')
    except Exception as e:
        with status_chart_container:
            ui.label('Could not load status chart')


def update_conditions_chart():
    """Update the conditions distribution chart."""
    global condition_colors
    
    # Clear previous chart
    conditions_chart_container.clear()
    
    # For visualization, we'll fetch a larger sample
    try:
        sample_results = search_trials(
            search_expr=search_expr,
            fields=["NCTId", "Condition"],
            max_results=100
        )
        
        # Count conditions distribution
        condition_counts = Counter()
        for study in sample_results.get('studies', []):
            protocol_section = study.get('protocolSection', {})
            conditions = protocol_section.get('conditionsModule', {}).get('conditions', [])
            for condition in conditions:
                condition_counts[condition] += 1
        
        # Get top 10 conditions
        top_conditions = dict(condition_counts.most_common(10))
        
        # Generate consistent colors for conditions
        condition_colors.update(generate_condition_colors(list(top_conditions.keys())))
        
        # Create chart data with explicit colors
        if top_conditions:
            chart_data = []
            for condition, count in top_conditions.items():
                chart_data.append({
                    'value': count,
                    'name': condition,
                    'itemStyle': {'color': condition_colors.get(condition, '#1f77b4')}
                })
            
            with conditions_chart_container:
                ui.echart({
                    'title': {'text': 'Top Conditions'},
                    'tooltip': {'trigger': 'axis'},
                    'xAxis': {
                        'type': 'category',
                        'data': list(top_conditions.keys()),
                        'axisLabel': {'rotate': 45}
                    },
                    'yAxis': {'type': 'value'},
                    'series': [{
                        'data': chart_data,
                        'type': 'bar'
                    }]
                }).classes('w-full h-64')
    except Exception as e:
        with conditions_chart_container:
            ui.label('Could not load conditions chart')


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


def get_condition_badge_color(condition):
    """Get the appropriate badge color for a condition."""
    return condition_colors.get(condition, 'blue')


def get_status_badge_color(status):
    """Get the appropriate badge color for a status."""
    return STATUS_COLORS.get(status, 'purple')


def create_trial_cards():
    """Create trial cards for each study."""
    print(f"DEBUG: create_trial_cards called, current_page={current_page}, len(search_results)={len(search_results)}")
    # Clear previous content
    results_container.clear()
    
    with results_container:
        if not search_results:
            ui.label('No studies found. Try a different search term.')
            return
            
        # Use grid layout for consistent card alignment
        with ui.grid(columns=1).classes('w-full gap-4'):
            for study in search_results:
                protocol_section = study.get('protocolSection', {})
                identification_module = protocol_section.get('identificationModule', {})
                status_module = protocol_section.get('statusModule', {})
                description_module = protocol_section.get('descriptionModule', {})
                conditions_module = protocol_section.get('conditionsModule', {})
                
                # Create card for each trial
                with ui.card().classes('w-full'):
                    # Header with title and NCT ID
                    with ui.row().classes('w-full justify-between items-start'):
                        ui.label(identification_module.get('briefTitle', 'N/A')).classes('text-lg')
                        ui.label(identification_module.get('nctId', 'N/A')).classes('text-caption')
                    
                    # Status badge
                    status = status_module.get('overallStatus', 'N/A')
                    ui.badge(status).props(f'color={get_status_badge_color(status)}')
                    
                    # Brief description
                    brief_summary = description_module.get('briefSummary', 'No description available')
                    if brief_summary and brief_summary != 'No description available':
                        ui.markdown(brief_summary[:200] + '...' if len(brief_summary) > 200 else brief_summary)
                    else:
                        ui.label('No description available').classes('text-gray-500 italic')
                    
                    # Expansion panel for more details
                    with ui.expansion('More Details', icon='info').classes('w-full'):
                        with ui.card().classes('w-full'):
                            # Conditions
                            conditions = conditions_module.get('conditions', [])
                            if conditions:
                                ui.label('Conditions:').classes('font-bold')
                                with ui.row():
                                    for condition in conditions[:5]:  # Show first 5 conditions
                                        ui.badge(condition).props(f'color={get_condition_badge_color(condition)}')
                                    if len(conditions) > 5:
                                        ui.badge(f'+{len(conditions) - 5} more').props('color=grey')
                    
                    # Action buttons
                    with ui.row().classes('w-full justify-end mt-2'):
                        ui.button('View Details', icon='visibility',
                                 on_click=lambda nct_id=identification_module.get('nctId', ''): show_study_details(nct_id)).props('outline')


def create_trial_list():
    """Create a list view of trials."""
    print(f"DEBUG: create_trial_list called, current_page={current_page}, len(search_results)={len(search_results)}")
    # Clear previous content
    results_container.clear()
    
    with results_container:
        if not search_results:
            ui.label('No studies found. Try a different search term.')
            return
            
        # Use grid layout for consistent list alignment
        with ui.grid(columns=1).classes('w-full gap-2'):
            for study in search_results:
                protocol_section = study.get('protocolSection', {})
                identification_module = protocol_section.get('identificationModule', {})
                status_module = protocol_section.get('statusModule', {})
                description_module = protocol_section.get('descriptionModule', {})
                conditions_module = protocol_section.get('conditionsModule', {})
                
                # Create list item for each trial
                with ui.card().classes('w-full'):
                    with ui.row().classes('w-full items-center justify-between'):
                        with ui.column():
                            with ui.row().classes('items-center gap-2'):
                                ui.label(identification_module.get('briefTitle', 'N/A'))
                                ui.label(identification_module.get('nctId', 'N/A')).classes('text-caption')
                            
                            # Status badge
                            status = status_module.get('overallStatus', 'N/A')
                            ui.badge(status).props(f'color={get_status_badge_color(status)}')
                            
                            # Brief description
                            brief_summary = description_module.get('briefSummary', 'No description available')
                            if brief_summary and brief_summary != 'No description available':
                                ui.label(brief_summary[:100] + '...' if len(brief_summary) > 100 else brief_summary).classes('text-sm text-gray-600')
                            else:
                                ui.label('No description available').classes('text-sm text-gray-500 italic')
                            
                            # Conditions as badges
                            conditions = conditions_module.get('conditions', [])
                            if conditions:
                                with ui.row():
                                    for condition in conditions[:2]:  # Show only first 2 in list view
                                        ui.badge(condition).props(f'color={get_condition_badge_color(condition)}')
                        
                        # Action button
                        ui.button('View Details', icon='visibility',
                                 on_click=lambda nct_id=identification_module.get('nctId', ''): show_study_details(nct_id)).props('outline')


def create_trial_table():
    """Create a table view of trials with badges for status and conditions."""
    print(f"DEBUG: create_trial_table called, current_page={current_page}, len(search_results)={len(search_results)}")
    # Clear previous content
    results_container.clear()
    
    with results_container:
        if not search_results:
            ui.label('No studies found. Try a different search term.')
            return
            
        # Create table with proper column alignment
        columns = [
            {'name': 'nct_id', 'label': 'NCT ID', 'field': 'nct_id', 'sortable': True, 'align': 'left'},
            {'name': 'title', 'label': 'Title', 'field': 'title', 'sortable': True, 'align': 'left'},
            {'name': 'description', 'label': 'Description', 'field': 'description', 'align': 'left'},
            {'name': 'status', 'label': 'Status', 'field': 'status', 'sortable': True, 'align': 'left'},
            {'name': 'conditions', 'label': 'Conditions', 'field': 'conditions', 'align': 'left'},
            {'name': 'actions', 'label': 'Actions', 'field': 'actions', 'align': 'left'}
        ]
        
        rows = []
        for study in search_results:
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
            conditions_display = ""
            if conditions:
                conditions_display = ", ".join(conditions[:3])
                if len(conditions) > 3:
                    conditions_display += f" (+{len(conditions) - 3} more)"
            
            rows.append({
                'nct_id': identification_module.get('nctId', 'N/A'),
                'title': identification_module.get('briefTitle', 'N/A'),
                'description': description,
                'status': status,
                'conditions': conditions_display,
                'actions': identification_module.get('nctId', 'N/A')
            })
        
        # Create table with NiceGUI props for styling
        table = ui.table(columns=columns, rows=rows, pagination=False).classes('w-full')
        table.props('wrap-cells dense')
        
        # Add color to status column using NiceGUI props
        table.add_slot('body-cell-status', '''
            <q-td :props="props">
                <q-badge :color="get_status_color(props.row.status)">{{ props.row.status }}</q-badge>
            </q-td>
        ''')
        
        # Add badges to conditions column
        table.add_slot('body-cell-conditions', '''
            <q-td :props="props">
                <div v-if="props.row.conditions">
                    <q-badge 
                        v-for="(condition, index) in props.row.conditions.split(', ')" 
                        :key="index" 
                        :color="get_condition_color(condition)"
                        class="q-mr-xs"
                    >
                        {{ condition }}
                    </q-badge>
                </div>
                <div v-else>
                    No conditions
                </div>
            </q-td>
        ''')
        
        # Add action buttons to the table
        table.add_slot('body-cell-actions', '''
            <q-td :props="props">
                <q-btn flat size="sm" icon="visibility" @click="() => $parent.$emit('view-details', props.row.actions)"></q-btn>
            </q-td>
        ''')
        
        # Add event handler for view details
        table.on('view-details', lambda e: show_study_details(e.args))
        
        # Add JavaScript function to get status color
        table.add_slot('bottom', '''
            <script>
                function get_status_color(status) {
                    const colors = {
                        'Active, not recruiting': 'blue',
                        'Recruiting': 'green',
                        'Completed': 'grey',
                        'Terminated': 'red',
                        'Withdrawn': 'orange',
                        'Suspended': 'yellow'
                    };
                    return colors[status] || 'purple';
                }
                
                function get_condition_color(condition) {
                    // Simple hash function to generate consistent colors
                    let hash = 0;
                    for (let i = 0; i < condition.length; i++) {
                        hash = condition.charCodeAt(i) + ((hash << 5) - hash);
                    }
                    const colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey'];
                    return colors[Math.abs(hash) % colors.length];
                }
            </script>
        ''')


def update_pagination_controls():
    """Update pagination controls."""
    print(f"DEBUG: update_pagination_controls called, current_page={current_page}, total_pages={total_pages}")
    pagination_controls.clear()
    
    # Create a separate function for each page button
    page_changers = {}
    for page_num in range(1, total_pages + 1):
        def make_page_changer(page_num=page_num):
            def page_changer():
                try:
                    change_page(page_num)
                except Exception as e:
                    print(f"DEBUG: Error in change_page: {str(e)}")
                    import traceback
                    traceback.print_exc()
            return page_changer
        page_changers[page_num] = make_page_changer()
    
    with pagination_controls:
        if total_pages > 1:
            with ui.row().classes('w-full justify-center items-center'):
                ui.button('First', on_click=lambda: change_page(1)).props('flat' if current_page == 1 else '')
                ui.button('Previous', on_click=lambda: change_page(max(1, current_page - 1))).props('flat' if current_page == 1 else '')
                
                
                # Page numbers
                start_page = max(1, current_page - 2)
                end_page = min(total_pages, start_page + 4)
                if end_page - start_page < 4:
                    start_page = max(1, end_page - 4)
                
                print(f"DEBUG: Creating page buttons from {start_page} to {end_page}")
                for page in range(start_page, end_page + 1):
                    print(f"DEBUG: Creating button for page {page}, current_page={current_page}")
                    if page == current_page:
                        ui.button(str(page), on_click=page_changers[page]).props('unelevated color=primary')
                    else:
                        ui.button(str(page), on_click=page_changers[page]).props('flat')
                ui.button('Next', on_click=lambda: change_page(min(total_pages, current_page + 1))).props('flat' if current_page == total_pages else '')
                ui.button('Last', on_click=lambda: change_page(total_pages)).props('flat' if current_page == total_pages else '')
                
                ui.label(f'Page {current_page} of {total_pages}').classes('text-caption')


def change_page(page):
    """Change to a specific page."""
    global current_page
    print(f"DEBUG: change_page called with page={page}, current_page was {current_page}")
    print(f"DEBUG: change_page function called with page={page}")
    print(f"DEBUG: About to update current_page from {current_page} to {page}")
    try:
        current_page = page
        print(f"DEBUG: current_page is now {current_page}")
        fetch_page_data()
        print(f"DEBUG: fetch_page_data completed for page {current_page}")
    except Exception as e:
        print(f"DEBUG: Error in change_page: {str(e)}")
        import traceback
        traceback.print_exc()


def change_view(view):
    """Change the view mode."""
    global current_view
    current_view = view
    update_results_view()


def show_study_details(nct_id: str):
    """Show full details for a specific study."""
    try:
        study = get_full_study(nct_id)
        
        # Create a dialog to show full details
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-4xl'):
            with ui.column().classes('w-full'):
                ui.label(f'Study Details: {nct_id}').classes('text-h6')
                
                # Tabs for different sections
                with ui.tabs() as tabs:
                    overview_tab = ui.tab('Overview')
                    details_tab = ui.tab('Full Details')
                
                with ui.tab_panels(tabs, value=overview_tab).classes('w-full'):
                    with ui.tab_panel(overview_tab):
                        display_study_overview(study)
                    
                    with ui.tab_panel(details_tab):
                        # Display full study data in a scrollable area
                        with ui.scroll_area().classes('h-96'):
                            ui.markdown(f"```json\n{json.dumps(study, indent=2)}\n```")
                
                ui.button('Close', on_click=dialog.close).classes('self-end')
        
        dialog.open()
        
    except Exception as e:
        ui.notify(f"Error fetching study details: {str(e)}", type='negative')


def display_study_overview(study):
    """Display a formatted overview of study details."""
    protocol_section = study.get('protocolSection', {})
    identification_module = protocol_section.get('identificationModule', {})
    status_module = protocol_section.get('statusModule', {})
    description_module = protocol_section.get('descriptionModule', {})
    conditions_module = protocol_section.get('conditionsModule', {})
    
    with ui.column().classes('w-full'):
        # Basic info
        with ui.card().classes('w-full'):
            ui.label('Basic Information').classes('text-subtitle1')
            with ui.grid(columns=2).classes('w-full'):
                ui.label('NCT ID:').classes('font-bold')
                ui.label(identification_module.get('nctId', 'N/A'))
                
                ui.label('Title:').classes('font-bold')
                ui.label(identification_module.get('briefTitle', 'N/A'))
                
                ui.label('Status:').classes('font-bold')
                status = status_module.get('overallStatus', 'N/A')
                ui.badge(status).props(f'color={get_status_badge_color(status)}')
                
                ui.label('Start Date:').classes('font-bold')
                ui.label(status_module.get('startDateStruct', {}).get('date', 'N/A'))
        
        # Description
        brief_summary = description_module.get('briefSummary', '')
        if brief_summary:
            with ui.card().classes('w-full'):
                ui.label('Description').classes('text-subtitle1')
                ui.markdown(brief_summary)
        else:
            with ui.card().classes('w-full'):
                ui.label('Description').classes('text-subtitle1')
                ui.label('No description available').classes('text-gray-500 italic')
        
        # Conditions
        conditions = conditions_module.get('conditions', [])
        if conditions:
            with ui.card().classes('w-full'):
                ui.label('Conditions').classes('text-subtitle1')
                with ui.row():
                    for condition in conditions:
                        ui.badge(condition).props(f'color={get_condition_badge_color(condition)}')


# Main UI
with ui.column().classes('w-full p-4 max-w-6xl mx-auto'):
    ui.label('Clinical Trial Search').classes('text-h4')
    
    # Visualizations at the top
    with ui.grid(columns=2).classes('w-full gap-6 mb-6'):
        with ui.card().classes('w-full'):
            status_chart_container = ui.column().classes('w-full')
        
        with ui.card().classes('w-full'):
            conditions_chart_container = ui.column().classes('w-full')
    
    # Search results panel (includes search controls, view options, and results)
    with ui.card().classes('w-full'):
        # Search controls
        with ui.column().classes('w-full gap-4'):
            with ui.row().classes('w-full items-center'):
                search_input = ui.input(
                    placeholder='Enter search terms (e.g., "breast cancer")',
                    value=search_expr
                ).classes('flex-grow')
                
                ui.button('Search', icon='search', on_click=fetch_data)
                ui.button('Reset', icon='refresh', on_click=reset_search).props('outline')
        
        # Results header with view toggle and total count
        with ui.row().classes('w-full items-center justify-between mt-4'):
            with ui.row().classes('items-center'):
                ui.label('Search Results').classes('text-h6')
                total_count_label = ui.label('')
            
            # View toggle
            with ui.row().classes('items-center'):
                ui.button(icon='view_module', on_click=lambda: change_view('cards')).props(
                    'flat' + (' unelevated color=primary' if current_view == 'cards' else '')
                ).tooltip('Card View')
                ui.button(icon='view_list', on_click=lambda: change_view('list')).props(
                    'flat' + (' unelevated color=primary' if current_view == 'list' else '')
                ).tooltip('List View')
                ui.button(icon='table_chart', on_click=lambda: change_view('table')).props(
                    'flat' + (' unelevated color=primary' if current_view == 'table' else '')
                ).tooltip('Table View')
        
        # Results container
        results_container = ui.column().classes('w-full mt-4')
        
        # Pagination controls
        pagination_controls = ui.column().classes('w-full mt-4')


# Fetch initial data
fetch_data()


# Run the app
ui.run(title='mCODE Clinical Trial Search', port=8083)