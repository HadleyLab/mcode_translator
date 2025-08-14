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
        app.logger.info(f"Searching for: {search_term} with limit {limit}")
        api_response = api.search_trials(search_term, max_results=limit)
        
        # Extract studies from API response
        if not isinstance(api_response, dict) or 'studies' not in api_response:
            raise ValueError("Invalid API response format")
            
        results = api_response['studies']
        app.logger.info(f"Found {len(results)} raw results")
        app.logger.debug(f"First result structure: {json.dumps(results[0], indent=2) if results else 'No results'}")
        
        # Process results through mCODE extraction pipeline
        pipeline = ExtractionPipeline()
        processed_results = pipeline.process_search_results(results)
        app.logger.info(f"Processed {len(processed_results)} results with mCODE data")
        app.logger.debug(f"First result mCODE structure: {json.dumps(processed_results[0]['mcode_data'], indent=2) if processed_results else 'No results'}")
        
        return jsonify({
            'status': 'success',
            'results': processed_results,
            'count': len(processed_results)
        })
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