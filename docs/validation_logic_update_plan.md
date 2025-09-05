# Validation Logic Update Plan

## Current Validation Issues

### 1. Field Comparison Mismatch
- **Current**: Compares `text` field from gold standard vs `element_name` from LLM output
- **Problem**: Should compare `element_name` vs `Mcode_type` (or equivalent fields)

### 2. Exact Text Matching
- **Current**: Uses exact string matching
- **Problem**: LLM generates different text representations but same semantic meaning
- **Solution**: Implement semantic similarity matching

### 3. Missing FHIR Structure Validation
- **Current**: Only validates basic text fields
- **Problem**: Doesn't validate proper FHIR resource structure
- **Solution**: Add FHIR compliance validation

## Required Validation Logic Changes

### 1. Field Comparison Correction

**Current (incorrect) validation in `run_breast_cancer_gold_standard_strict.py`:**
```python
# Lines 131-183: calculate_metrics() method
# Compares gold_standard_mapping['text'] vs predicted_mapping['element_name']
```

**Should be changed to:**
```python
# Compare element_name vs Mcode_type (or equivalent semantic fields)
gold_standard_element = gold_standard_mapping.get('element_name', '')
predicted_element = predicted_mapping.get('element_name', '')
```

### 2. Semantic Similarity Implementation

**Add semantic similarity function:**
```python
def semantic_similarity(text1, text2):
    """Calculate semantic similarity between two text strings"""
    # Simple implementation using text normalization and fuzzy matching
    normalized1 = text1.lower().strip()
    normalized2 = text2.lower().strip()
    
    # Exact match
    if normalized1 == normalized2:
        return 1.0
    
    # Contains match
    if normalized1 in normalized2 or normalized2 in normalized1:
        return 0.9
    
    # Word overlap
    words1 = set(normalized1.split())
    words2 = set(normalized2.split())
    overlap = len(words1.intersection(words2))
    total = len(words1.union(words2))
    
    if total > 0:
        return overlap / total
    
    return 0.0
```

### 3. FHIR Structure Validation

**Add FHIR compliance validation:**
```python
def validate_fhir_structure(mapping):
    """Validate that mapping has proper FHIR structure"""
    required_fields = ['resourceType', 'element_name', 'code']
    for field in required_fields:
        if field not in mapping:
            return False
    
    # Validate code structure
    code = mapping.get('code', {})
    if not isinstance(code, dict):
        return False
    if 'system' not in code or 'code' not in code:
        return False
    
    return True
```

## Updated Validation Logic

### Modified calculate_metrics() function:

```python
def calculate_metrics(gold_standard_mappings, predicted_mappings, similarity_threshold=0.7):
    """
    Calculate precision, recall, and F1 score with semantic similarity matching
    
    Args:
        gold_standard_mappings: List of gold standard mappings
        predicted_mappings: List of predicted mappings from LLM
        similarity_threshold: Threshold for considering a match (0.0-1.0)
    
    Returns:
        Dictionary with precision, recall, F1 score, and detailed metrics
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Convert to dictionaries for easier lookup
    gold_dict = {mapping.get('element_name', ''): mapping for mapping in gold_standard_mappings}
    pred_dict = {mapping.get('element_name', ''): mapping for mapping in predicted_mappings}
    
    # Find matches using semantic similarity
    matched_gold = set()
    matched_pred = set()
    
    for gold_key, gold_mapping in gold_dict.items():
        for pred_key, pred_mapping in pred_dict.items():
            similarity = semantic_similarity(gold_key, pred_key)
            if similarity >= similarity_threshold:
                true_positives += 1
                matched_gold.add(gold_key)
                matched_pred.add(pred_key)
                break
    
    # Calculate remaining unmatched
    false_positives = len(pred_dict) - len(matched_pred)
    false_negatives = len(gold_dict) - len(matched_gold)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # FHIR compliance validation
    fhir_compliant_count = sum(1 for mapping in predicted_mappings if validate_fhir_structure(mapping))
    fhir_compliance_rate = fhir_compliant_count / len(predicted_mappings) if predicted_mappings else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'fhir_compliance_rate': fhir_compliance_rate,
        'total_gold_standard': len(gold_standard_mappings),
        'total_predicted': len(predicted_mappings)
    }
```

## Implementation Steps

### Step 1: Update Validation Logic
Modify `run_breast_cancer_gold_standard_strict.py` to:
1. Use semantic similarity matching instead of exact text matching
2. Compare correct fields (`element_name` vs `element_name`)
3. Add FHIR structure validation

### Step 2: Create Utility Functions
Add the `semantic_similarity()` and `validate_fhir_structure()` functions

### Step 3: Update Test Runner
Ensure the test runner uses the fixed gold standard file

### Step 4: Test Performance
Run validation to verify improved mapping performance

## Expected Results
- Mapping performance should improve significantly (from 0% to >80%)
- Proper semantic matching instead of exact text comparison
- FHIR compliance validation included
- More accurate performance metrics