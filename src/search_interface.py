import json
from flask import Flask, render_template, request, jsonify
from .clinical_trials_api import ClinicalTrialsAPI
from .config import Config
from .extraction_pipeline import ExtractionPipeline

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/')
def search_page():
    return render_template('search.html')

@app.route('/api/search', methods=['POST'])
def search_trials():
    try:
        search_term = request.json.get('query')
        limit = request.json.get('limit', 10)
        
        api = ClinicalTrialsAPI(Config())
        # Use more specific breast cancer search terms
        enhanced_search = f"{search_term} AND breast cancer AND (PIK3CA OR TP53 OR ESR1 OR HER2 OR ER OR PR)"
        app.logger.info(f"Searching for: {enhanced_search} with limit {limit}")
        api_response = api.search_trials(enhanced_search, max_results=limit)
        
        # Extract studies from API response
        if not isinstance(api_response, dict) or 'studies' not in api_response:
            app.logger.error(f"Invalid API response format: {type(api_response)}")
            app.logger.debug(f"Full API response: {json.dumps(api_response, indent=2)}")
            raise ValueError("Invalid API response format")
            
        results = api_response['studies']
        app.logger.info(f"Found {len(results)} raw results")
        
        # Detailed logging of first 3 results
        for i, trial in enumerate(results[:3]):
            app.logger.info(f"Result {i+1} structure keys: {list(trial.keys())}")
            protocol = trial.get('protocolSection', {})
            app.logger.info(f"Protocol section keys: {list(protocol.keys())}")
            
            eligibility = protocol.get('eligibilityModule', {})
            criteria = eligibility.get('eligibilityCriteria', '')
            app.logger.info(f"Trial {i+1} criteria found: {bool(criteria)}")
            
            if criteria:
                app.logger.debug(f"Criteria sample: {criteria[:200]}...")
                # Test extraction pipeline
                pipeline = ExtractionPipeline()
                extracted = pipeline.process_criteria(criteria)
                app.logger.info(f"Extraction result keys: {list(extracted.keys())}")
                if extracted.get('mcode_mappings'):
                    app.logger.info(f"Mapped elements: {len(extracted['mcode_mappings'].get('mapped_elements', []))}")
        
        # Process results through mCODE extraction pipeline
        pipeline = ExtractionPipeline()
        processed_results = pipeline.process_search_results(results)
        app.logger.info(f"Processed {len(processed_results)} results with mCODE data")
        app.logger.debug(f"First result mCODE structure: {json.dumps(processed_results[0]['mcode_data'], indent=2) if processed_results else 'No results'}")
        
        response_data = {
            'status': 'success',
            'results': processed_results,
            'count': len(processed_results),
            'mcode_version': '1.0'
        }
        app.logger.info(f"Returning response with {len(processed_results)} processed results")
        app.logger.debug(f"Sample processed result: {json.dumps(processed_results[0], indent=2) if processed_results else 'None'}")
        return jsonify(response_data)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/extract', methods=['POST'])
def extract_codes():
    try:
        criteria_text = request.json.get('criteria')
        if not criteria_text:
            raise ValueError("No criteria text provided")
            
        pipeline = ExtractionPipeline()
        result = pipeline.process_criteria(criteria_text)
        
        return jsonify({
            'status': 'success',
            'result': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)