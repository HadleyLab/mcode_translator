from nicegui import ui
from clinical_trials_api import ClinicalTrialsAPI
from extraction_pipeline import ExtractionPipeline
from config import Config
import json

def init_nicegui_interface():
    # Search input and controls
    with ui.header().classes('bg-blue-100 p-4'):
        ui.label('Clinical Trials Search').classes('text-2xl font-bold')
        search_input = ui.input('Search terms', placeholder='breast cancer').classes('w-96')
        search_button = ui.button('Search', icon='search')
        limit_slider = ui.slider(min=1, max=20, value=5, step=1).props('label-always')
        ui.label().bind_text_from(limit_slider, 'value', lambda v: f'Results limit: {v}')

    # Results display area
    results_container = ui.column().classes('w-full p-4 gap-4')

    async def perform_search():
        search_term = search_input.value or 'breast cancer'
        limit = int(limit_slider.value)
        
        results_container.clear()
        with results_container:
            ui.spinner('dots', size='lg', color='primary')
            ui.label('Searching...')

        try:
            # Call backend API
            api = ClinicalTrialsAPI(Config())
            pipeline = ExtractionPipeline()
            
            raw_results = api.search_trials(search_term, max_results=limit)
            processed_results = pipeline.process_search_results(raw_results['studies'])
            
            results_container.clear()
            with results_container:
                if not processed_results:
                    ui.label('No matching trials found').classes('text-lg')
                    return
                
                ui.label(f'Found {len(processed_results)} trials').classes('text-lg font-bold')
                
                for trial in processed_results:
                    with ui.card().classes('w-full p-4 gap-2'):
                        # Basic trial info
                        protocol = trial.get('protocolSection', {})
                        ident = protocol.get('identificationModule', {})
                        ui.label(ident.get('briefTitle', 'No title')).classes('text-xl font-bold')
                        ui.label(f"NCT ID: {ident.get('nctId', 'Unknown')}")
                        
                        # mCODE data display
                        mcode_data = trial.get('mcode_data', {})
                        if mcode_data:
                            with ui.expansion('View mCODE Data').classes('w-full'):
                                display_mcode_data(mcode_data)
                        else:
                            ui.label('No mCODE data extracted').classes('text-sm italic')

        except Exception as e:
            results_container.clear()
            with results_container:
                ui.notify(f"Error: {str(e)}", type='negative')
                ui.label(f"Error occurred: {str(e)}").classes('text-red')

    def display_mcode_data(mcode_data):
        """Display mCODE data in a structured format"""
        # Genomic Variants
        if mcode_data.get('mcode_mappings', {}).get('mapped_elements'):
            variants = [e for e in mcode_data['mcode_mappings']['mapped_elements'] 
                      if e.get('mcode_element') == 'GenomicVariant']
            if variants:
                with ui.card().classes('bg-blue-50 p-2'):
                    ui.label('Genomic Variants').classes('font-bold')
                    for var in variants:
                        ui.label(f"{var.get('geneStudied')}: {var.get('dnaChange')}")

        # Biomarkers
        if mcode_data.get('mcode_mappings', {}).get('mapped_elements'):
            biomarkers = [e for e in mcode_data['mcode_mappings']['mapped_elements']
                        if e.get('mcode_element') == 'Biomarker']
            if biomarkers:
                with ui.card().classes('bg-green-50 p-2'):
                    ui.label('Biomarkers').classes('font-bold')
                    for bio in biomarkers:
                        ui.label(f"{bio.get('code')}: {bio.get('value')}")

        # Full mCODE structure (collapsed by default)
        with ui.expansion('View Full mCODE Structure'):
            ui.json_editor({'content': {'json': mcode_data.get('mcode_mappings', {})}}, 
                         language='json', expanded=False)

    # Wire up search button
    search_button.on_click(perform_search)

    # Initialize with default search
    ui.timer(0.1, perform_search, once=True)

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(reload=False,
           title="mCODE Clinical Trials Search")