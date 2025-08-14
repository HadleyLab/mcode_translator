from nicegui import ui
from src.clinical_trials_api import ClinicalTrialsAPI

# Initialize API client
api_client = ClinicalTrialsAPI()

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
        
        search_button = ui.button('Search', icon='search').classes('w-32')

    # Results display
    trials_container = ui.column().classes('w-full p-4')

# Define search handler
def on_search():
    trials_container.clear()
    with trials_container:
        ui.spinner('dots', size='lg', color='primary')
        results = api_client.search_trials(search_input.value, max_results=int(limit_slider.value))
        
        for study in results.get('studies', []):
            protocol_section = study.get('protocolSection', {})
            with ui.card().classes('w-full'):
                ui.label(protocol_section.get('identificationModule', {}).get('officialTitle')).classes('text-lg')
                ui.label(f"NCT ID: {protocol_section.get('identificationModule', {}).get('nctId')}")
                ui.label(f"Status: {protocol_section.get('statusModule', {}).get('overallStatus')}")
                conditions = protocol_section.get('conditionsModule', {}).get('conditions', [])
                ui.label(f"Conditions: {', '.join(conditions)}")
                
                # Show detailed description on click
                with ui.expansion('View Details').classes('w-full'):
                    description = protocol_section.get('descriptionModule', {}).get('detailedDescription', '')
                    ui.markdown(description)

# Bind handler to button click
search_button.on('click', on_search)

ui.run(port=8082, title='mCODE Clinical Trials Search', reload=False)