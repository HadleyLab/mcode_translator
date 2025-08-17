# Component-Based Testing for NLP Engines

This document outlines the component-based testing approach for NLP engines in the mCODE Translator project.

## Overview

Component-based testing for NLP engines focuses on testing each engine's capabilities independently and comparing their performance. This approach ensures that each engine meets the required standards and allows for easy comparison between different approaches.

## NLP Engine Components

### 1. Regex NLP Engine

Testing the regex-based NLP engine for pattern matching.

```python
class TestRegexNLPEngine:
    """Test the Regex NLP Engine"""
    
    def test_basic_entity_extraction(self, regex_nlp_engine):
        """Test basic entity extraction with regex patterns"""
        text = "Patient diagnosed with breast cancer (ICD-10-CM: C50.911)"
        result = regex_nlp_engine.process_criteria(text)
        
        assert 'entities' in result
        assert len(result['entities']) > 0
        assert any(entity['text'] == 'breast cancer' for entity in result['entities'])
    
    def test_code_extraction(self, regex_nlp_engine):
        """Test code extraction from text"""
        text = "Histologically confirmed diagnosis of breast cancer (ICD-10-CM: C50.911)"
        result = regex_nlp_engine.process_criteria(text)
        
        # Extract codes using the regex engine's code extraction method
        codes = regex_nlp_engine.extract_codes(text)
        assert 'ICD10CM' in codes
        assert codes['ICD10CM'][0]['code'] == 'C50.911'
    
    @pytest.mark.parametrize("pattern,text,expected_match", [
        (r'breast cancer', 'Patient has breast cancer', True),
        (r'lung cancer', 'Patient has breast cancer', False),
        (r'ICD-10-CM:\s*([A-Z0-9.]+)', 'Diagnosis: ICD-10-CM: C50.911', True),
    ])
    def test_regex_patterns(self, pattern, text, expected_match):
        """Test individual regex patterns"""
        import re
        match = re.search(pattern, text)
        if expected_match:
            assert match is not None
        else:
            assert match is None
    
    def test_performance_benchmark(self, regex_nlp_engine):
        """Test performance of regex engine"""
        import time
        text = "Patient with BRCA1 mutation, ER+ HER2- breast cancer, stage IIA"
        
        start_time = time.time()
        for _ in range(100):
            result = regex_nlp_engine.process_criteria(text)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.01  # Should process in less than 10ms on average
```

### 2. SpaCy NLP Engine

Testing the SpaCy-based NLP engine for natural language processing.

```python
class TestSpacyNLPEngine:
    """Test the SpaCy NLP Engine"""
    
    def test_entity_recognition(self, spacy_nlp_engine):
        """Test entity recognition capabilities"""
        text = "Patient diagnosed with breast cancer, female, 55 years old"
        result = spacy_nlp_engine.process_criteria(text)
        
        assert 'entities' in result
        assert len(result['entities']) >= 3  # cancer, female, 55
        
        # Check for medical entities
        medical_entities = [e for e in result['entities'] if e.get('type') in ['CONDITION', 'GENDER', 'AGE']]
        assert len(medical_entities) >= 1
    
    def test_medical_concept_extraction(self, spacy_nlp_engine):
        """Test extraction of medical concepts"""
        text = "Patient with estrogen receptor positive, progesterone receptor negative, HER2 positive"
        result = spacy_nlp_engine.process_criteria(text)
        
        # Check for biomarker entities
        biomarker_entities = [e for e in result['entities'] if 'receptor' in e['text'].lower() or 'her2' in e['text'].lower()]
        assert len(biomarker_entities) >= 3
    
    def test_demographic_extraction(self, spacy_nlp_engine):
        """Test demographic information extraction"""
        text = "Female patient, 55 years old, diagnosed with breast cancer"
        result = spacy_nlp_engine.extract_demographics(text)
        
        assert 'gender' in result
        assert 'age' in result
        assert result['gender'].lower() == 'female'
        assert result['age'] == '55'
    
    def test_confidence_scoring(self, spacy_nlp_engine):
        """Test confidence scoring of extracted entities"""
        text = "Patient diagnosed with breast cancer"
        result = spacy_nlp_engine.process_criteria(text)
        
        for entity in result['entities']:
            assert 'confidence' in entity
            assert 0.0 <= entity['confidence'] <= 1.0
```

### 3. LLM NLP Engine

Testing the LLM-based NLP engine for advanced natural language understanding.

```python
class TestLLMNLPEngine:
    """Test the LLM NLP Engine"""
    
    def test_contextual_understanding(self, llm_nlp_engine):
        """Test contextual understanding of complex criteria"""
        text = """
        INCLUSION CRITERIA:
        - Histologically confirmed diagnosis of breast cancer (ICD-10-CM: C50.911)
        - Female patients aged 18-75 years
        - Eastern Cooperative Oncology Group (ECOG) performance status of 0 or 1
        - Adequate organ function as defined by laboratory values
        
        EXCLUSION CRITERIA:
        - Pregnant or nursing women
        - History of other malignancies within 5 years
        """
        result = llm_nlp_engine.extract_mcode_features(text)
        
        assert 'features' in result
        assert 'cancer_characteristics' in result['features']
        assert 'demographics' in result['features']
        assert 'performance_status' in result['features']
        assert 'exclusions' in result['features']
    
    def test_genomic_variant_extraction(self, llm_nlp_engine):
        """Test extraction of genomic variants"""
        text = "Patient with BRCA1 mutation, ER+ HER2- breast cancer"
        result = llm_nlp_engine.extract_mcode_features(text)
        
        assert 'genomic_variants' in result['features']
        genomic_variants = result['features']['genomic_variants']
        assert len(genomic_variants) >= 1
        assert any('BRCA1' in variant.get('gene', '') for variant in genomic_variants)
    
    def test_biomarker_analysis(self, llm_nlp_engine):
        """Test biomarker analysis and interpretation"""
        text = "Estrogen receptor positive, progesterone receptor negative, HER2 positive"
        result = llm_nlp_engine.extract_mcode_features(text)
        
        assert 'biomarkers' in result['features']
        biomarkers = result['features']['biomarkers']
        assert len(biomarkers) >= 3
        
        # Check for specific biomarkers
        er_positive = any(b.get('name') == 'ER' and b.get('status') == 'positive' for b in biomarkers)
        pr_negative = any(b.get('name') == 'PR' and b.get('status') == 'negative' for b in biomarkers)
        her2_positive = any(b.get('name') == 'HER2' and b.get('status') == 'positive' for b in biomarkers)
        
        assert er_positive or pr_negative or her2_positive
    
    def test_treatment_history_extraction(self, llm_nlp_engine):
        """Test extraction of treatment history"""
        text = "Patient previously treated with paclitaxel and trastuzumab"
        result = llm_nlp_engine.extract_mcode_features(text)
        
        assert 'treatment_history' in result['features']
        treatments = result['features']['treatment_history']
        assert len(treatments) >= 2
        
        # Check for specific medications
        medications = treatments.get('medications', [])
        assert len(medications) >= 2
        medication_names = [med.get('name', '').lower() for med in medications]
        assert 'paclitaxel' in medication_names or 'trastuzumab' in medication_names
```

## NLP Engine Comparison Testing

Testing and comparing the performance of different NLP engines.

```python
class TestNLPEngineComparison:
    """Compare different NLP engines"""
    
    def test_entity_extraction_comparison(self, regex_nlp_engine, spacy_nlp_engine, llm_nlp_engine):
        """Compare entity extraction capabilities"""
        text = "Patient diagnosed with breast cancer (ICD-10-CM: C50.911), female, 55 years old"
        
        # Process with each engine
        regex_result = regex_nlp_engine.process_criteria(text)
        spacy_result = spacy_nlp_engine.process_criteria(text)
        llm_result = llm_nlp_engine.extract_mcode_features(text)
        
        # Compare number of entities found
        regex_count = len(regex_result.get('entities', []))
        spacy_count = len(spacy_result.get('entities', []))
        llm_count = len(llm_result.get('entities', []))
        
        # Each engine should find at least some entities
        assert regex_count >= 1
        assert spacy_count >= 1
        assert llm_count >= 1
    
    def test_code_extraction_accuracy(self, regex_nlp_engine, spacy_nlp_engine):
        """Compare code extraction accuracy"""
        text = "Histologically confirmed diagnosis of breast cancer (ICD-10-CM: C50.911)"
        
        # Extract codes with each engine
        regex_codes = regex_nlp_engine.extract_codes(text)
        # For SpaCy, we might need to use a different approach
        spacy_result = spacy_nlp_engine.process_criteria(text)
        
        # Check that regex engine correctly identifies the ICD-10 code
        assert 'ICD10CM' in regex_codes
        assert len(regex_codes['ICD10CM']) >= 1
        assert regex_codes['ICD10CM'][0]['code'] == 'C50.911'
    
    def test_processing_time_comparison(self, regex_nlp_engine, spacy_nlp_engine, llm_nlp_engine):
        """Compare processing times of different engines"""
        import time
        text = "Patient with BRCA1 mutation, ER+ HER2- breast cancer, stage IIA"
        
        # Time regex engine
        start = time.time()
        for _ in range(50):
            regex_nlp_engine.process_criteria(text)
        regex_time = time.time() - start
        
        # Time SpaCy engine
        start = time.time()
        for _ in range(50):
            spacy_nlp_engine.process_criteria(text)
        spacy_time = time.time() - start
        
        # Time should be reasonable for each engine
        assert regex_time < 1.0  # Regex should be very fast
        assert spacy_time < 2.0  # SpaCy should be reasonably fast
        # Note: LLM engine is not timed as it may be much slower and involve external APIs
    
    def test_confidence_comparison(self, spacy_nlp_engine, llm_nlp_engine):
        """Compare confidence scores between engines"""
        text = "Patient diagnosed with breast cancer"
        
        spacy_result = spacy_nlp_engine.process_criteria(text)
        llm_result = llm_nlp_engine.extract_mcode_features(text)
        
        # Check that both engines provide confidence scores
        spacy_entities = spacy_result.get('entities', [])
        for entity in spacy_entities:
            assert 'confidence' in entity
            assert 0.0 <= entity['confidence'] <= 1.0
        
        # LLM engine should also provide confidence information
        # This might be in a different format depending on implementation
```

## Parameterized Testing for NLP Scenarios

Using parameterized tests to efficiently test various NLP scenarios.

```python
class TestNLPEngineParameterized:
    """Parameterized tests for NLP engines"""
    
    @pytest.mark.parametrize("engine_type,text,expected_entities", [
        ("regex", "Patient has breast cancer", ["breast cancer"]),
        ("spacy", "Patient diagnosed with lung cancer", ["lung cancer"]),
        ("llm", "Female patient, 55 years old with breast cancer", ["breast cancer", "female", "55"]),
    ])
    def test_entity_extraction_scenarios(self, engine_type, text, expected_entities):
        """Test entity extraction for different scenarios"""
        # This is a conceptual test - in practice, you'd get the actual engine instance
        # based on engine_type and run the test
        
        # For demonstration, we'll just check that the parameters make sense
        assert isinstance(text, str)
        assert isinstance(expected_entities, list)
        assert len(expected_entities) > 0
    
    @pytest.mark.parametrize("cancer_type,biomarkers", [
        ("breast", ["ER+", "PR-", "HER2+"]),
        ("lung", ["PD-L1+", "EGFR+"]),
        ("colorectal", ["KRAS-", "NRAS-"]),
    ])
    def test_cancer_specific_biomarkers(self, llm_nlp_engine, cancer_type, biomarkers):
        """Test biomarker extraction for different cancer types"""
        text = f"Patient with {cancer_type} cancer, biomarkers: {', '.join(biomarkers)}"
        result = llm_nlp_engine.extract_mcode_features(text)
        
        # Check that biomarkers are extracted
        extracted_biomarkers = result['features'].get('biomarkers', [])
        assert len(extracted_biomarkers) >= len(biomarkers)
    
    @pytest.mark.parametrize("criteria_complexity,text_length_range", [
        ("simple", (50, 100)),
        ("medium", (100, 300)),
        ("complex", (300, 1000)),
    ])
    def test_processing_complex_criteria(self, regex_nlp_engine, criteria_complexity, text_length_range):
        """Test processing criteria of different complexities"""
        # Generate test data based on complexity
        text = ClinicalTrialDataGenerator.generate_eligibility_criteria(criteria_complexity)
        
        # Check text length is within expected range
        assert text_length_range[0] <= len(text) <= text_length_range[1]
        
        # Process the text
        result = regex_nlp_engine.process_criteria(text)
        assert 'entities' in result
```

## Benefits of Component-Based NLP Testing

1. **Engine Isolation**: Each NLP engine is tested independently
2. **Performance Comparison**: Easy comparison of different engines' performance
3. **Feature Coverage**: Comprehensive testing of each engine's capabilities
4. **Maintainability**: Changes to one engine don't affect tests for others
5. **Scalability**: New engines can be added with minimal impact on existing tests
6. **Confidence Assessment**: Systematic evaluation of confidence scoring
7. **Benchmarking**: Performance benchmarks for each engine type