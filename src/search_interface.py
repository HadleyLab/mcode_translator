from flask import Flask, render_template, request, jsonify
from src.clinical_trials_api import ClinicalTrialsAPI
from src.config import Config

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
        results = api.search_trials(search_term, max_results=limit)
        
        return jsonify({
            'status': 'success',
            'results': results
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)