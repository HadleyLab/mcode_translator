# mCODE Translator

## Running the Clinical Trial Data Fetcher

The clinical trial data fetcher can be run from the command line with various options:

### Basic Usage

```bash
# Activate the conda environment first
conda activate mcode_translator

# Show help
python src/data_fetcher/fetcher.py --help

# Search for clinical trials
python src/data_fetcher/fetcher.py --condition "breast cancer" --limit 10

# Fetch a specific trial by NCT ID
python src/data_fetcher/fetcher.py --nct-id NCT00000000

# Export results to JSON file
python src/data_fetcher/fetcher.py --condition "lung cancer" --export results.json

# Process eligibility criteria with NLP engine
python src/data_fetcher/fetcher.py --nct-id NCT00000000 --process-criteria

# Search with pagination
python src/data_fetcher/fetcher.py --condition "breast cancer" --min-rank 50

# Calculate total number of studies
python src/data_fetcher/fetcher.py --condition "breast cancer" --count
```

## Running Tests

### Unit Tests

```bash
# Run unit tests for the fetcher
python -m pytest tests/unit/test_fetcher.py -v

# Run all unit tests
python -m pytest tests/unit/ -v
```

### Integration Tests

```bash
# Run integration tests (including the new TestAPIIntegration class)
python -m pytest tests/integration/test_api_integration.py -v

# Run all integration tests
python -m pytest tests/integration/ -v
```

### All Tests

```bash
# Run the complete test suite
python run_tests.py
```

## Project Structure

- `src/data_fetcher/fetcher.py` - Main clinical trial data fetcher script
- `tests/unit/test_fetcher.py` - Unit tests for the fetcher
- `tests/integration/test_api_integration.py` - Integration tests including TestAPIIntegration class
- `src/utils/config.py` - Configuration management
- `src/utils/cache.py` - Caching mechanism
- `src/nlp_engine/` - NLP processing components
- `src/code_extraction/` - Code extraction components
- `src/mcode_mapper/` - mCODE mapping components
- `src/structured_data_generator/` - Structured data generation components

## Dependencies

The project requires the following dependencies:

- `requests>=2.25.1`
- `click>=7.1.2`
- `python-dotenv>=0.15.0`
- `spacy==3.7.5`
- `scispacy==0.5.4`
- `pytrials==0.3.0`

Install with:
```bash
pip install -r requirements.txt
```

## Testing Dependencies

- `pytest>=6.0.0`
- `pytest-cov>=2.10.0`
- `pytest-html>=3.1.1`
- `coverage>=5.5`

Install with:
```bash
pip install -r requirements-test.txt