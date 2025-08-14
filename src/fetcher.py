import click
import json
import sys
from .clinical_trials_api import ClinicalTrialsAPI, ClinicalTrialsAPIError
from .config import Config
from .nlp_engine import NLPEngine
from .code_extraction import CodeExtractionModule
from .mcode_mapping_engine import MCODEMappingEngine
from .structured_data_generator import StructuredDataGenerator


@click.command()
@click.option('--condition', '-c', help='Condition to search for (e.g., "breast cancer")')
@click.option('--nct-id', '-n', help='Specific NCT ID to fetch (e.g., "NCT00000000")')
@click.option('--limit', '-l', default=10, help='Maximum number of results to return')
@click.option('--export', '-e', type=click.Path(), help='Export results to JSON file')
@click.option('--process-criteria', '-p', is_flag=True, help='Process eligibility criteria with NLP engine')
def main(condition, nct_id, limit, export, process_criteria):
    """
    Clinical Trial Data Fetcher for mCODE Translator
    
    Examples:
      python fetcher.py --condition "breast cancer" --limit 10
      python fetcher.py --nct-id NCT00000000
      python fetcher.py --condition "lung cancer" --export results.json
      python fetcher.py --nct-id NCT00000000 --process-criteria
    """
    # Initialize the API client
    config = Config()
    api = ClinicalTrialsAPI(config)
    
    try:
        if nct_id:
            # Fetch a specific trial by NCT ID
            click.echo(f"Fetching trial {nct_id}...")
            result = api.get_full_study(nct_id)
            display_single_study(result, export, process_criteria)
        elif condition:
            # Search for trials by condition
            click.echo(f"Searching for trials matching '{condition}'...")
            result = api.search_trials(condition, None, limit)
            display_search_results(result, export)
        else:
            # Show help if no arguments provided
            click.echo("Please specify either a condition to search for or an NCT ID to fetch.")
            click.echo("Use --help for more information.")
            sys.exit(1)
            
    except ClinicalTrialsAPIError as e:
        click.echo(f"API Error: {str(e)}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected Error: {str(e)}", err=True)
        sys.exit(1)


def display_single_study(result, export_path=None, process_criteria=False):
    """
    Display a single study result or export to file
    
    Args:
        result: Study result to display
        export_path: Optional path to export JSON file
        process_criteria: Whether to process eligibility criteria with NLP
    """
    # Process eligibility criteria with NLP if requested
    if process_criteria and 'protocolSection' in result:
        protocol_section = result['protocolSection']
        if 'eligibilityModule' in protocol_section:
            eligibility_module = protocol_section['eligibilityModule']
            if 'eligibilityCriteria' in eligibility_module:
                criteria_text = eligibility_module['eligibilityCriteria']
                if criteria_text:
                    try:
                        nlp_engine = NLPEngine()
                        # Ensure criteria_text is a string
                        if isinstance(criteria_text, list):
                            criteria_text = ' '.join(str(item) for item in criteria_text)
                        elif not isinstance(criteria_text, str):
                            criteria_text = str(criteria_text)
                        nlp_result = nlp_engine.process_criteria(criteria_text)
                        
                        # Process through the full pipeline
                        # Step 1: Extract codes
                        code_extractor = CodeExtractionModule()
                        code_result = code_extractor.process_criteria_for_codes(criteria_text, nlp_result['entities'])
                        
                        # Step 2: Map to mCODE
                        mapper = MCODEMappingEngine()
                        mapping_result = mapper.process_nlp_output(nlp_result)
                        
                        # Step 3: Generate structured data
                        generator = StructuredDataGenerator()
                        structured_result = generator.generate_mcode_resources(
                            mapping_result['mapped_elements'],
                            nlp_result.get('demographics', {})
                        )
                        
                        # Add all results to the study data
                        if 'mcodeResults' not in result:
                            result['mcodeResults'] = {}
                        result['mcodeResults']['nlp'] = nlp_result
                        result['mcodeResults']['codes'] = code_result
                        result['mcodeResults']['mappings'] = mapping_result
                        result['mcodeResults']['structured_data'] = structured_result
                    except Exception as e:
                        click.echo(f"Warning: Error processing criteria with NLP: {str(e)}", err=True)
    
    if export_path:
        # Export to JSON file
        with open(export_path, 'w') as f:
            json.dump(result, f, indent=2)
        click.echo(f"Study exported to {export_path}")
    else:
        # Display study to console
        click.echo(json.dumps(result, indent=2))


def display_search_results(result, export_path=None):
    """
    Display search results or export to file
    
    Args:
        result: Search results to display
        export_path: Optional path to export JSON file
    """
    if export_path:
        # Export to JSON file
        with open(export_path, 'w') as f:
            json.dump(result, f, indent=2)
        click.echo(f"Results exported to {export_path}")
    else:
        # Display results to console
        if 'studies' in result:
            studies = result['studies']
            click.echo(f"Found {len(studies)} trials:")
            for i, study in enumerate(studies[:10]):  # Show up to 10 results
                # Extract NCT ID and title
                nct_id = 'Unknown'
                title = 'No title'
                
                if 'protocolSection' in study:
                    protocol_section = study['protocolSection']
                    if 'identificationModule' in protocol_section:
                        identification_module = protocol_section['identificationModule']
                        if 'nctId' in identification_module:
                            nct_id = identification_module['nctId']
                        if 'briefTitle' in identification_module:
                            title = identification_module['briefTitle']
                
                click.echo(f"  {i+1}. {nct_id}: {title}")
        else:
            click.echo("No studies found in results")


def display_results(results, export_path=None):
    """
    Display results or export to file
    
    Args:
        results: Results to display
        export_path: Optional path to export JSON file
    """
    if export_path:
        # Export to JSON file
        with open(export_path, 'w') as f:
            json.dump(results, f, indent=2)
        click.echo(f"Results exported to {export_path}")
    else:
        # Display results to console
        click.echo(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()