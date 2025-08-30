import json
from typing import List, Dict

def calculate_extraction_metrics(actual_entities: List[Dict], expected_entities: List[Dict]) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score for entity extraction.
    Uses strict text matching on the 'text' field in LLM format.
    """
    # Extract text values for comparison
    actual_texts = [entity['text'].lower().strip() for entity in actual_entities if 'text' in entity]
    expected_texts = [entity['text'].lower().strip() for entity in expected_entities if 'text' in entity]
    
    # Calculate true positives, false positives, false negatives
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Count true positives (entities found in both actual and expected)
    for actual_text in actual_texts:
        if actual_text in expected_texts:
            true_positives += 1
        else:
            false_positives += 1
    
    # Count false negatives (entities in expected but not in actual)
    for expected_text in expected_texts:
        if expected_text not in actual_texts:
            false_negatives += 1
    
    # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "actual_count": len(actual_texts),
        "expected_count": len(expected_texts)
    }

# Load the new gold standard
with open('examples/breast_cancer_data/breast_cancer_gold_standard_llm_format.json', 'r') as f:
    gold_data = json.load(f)

# Load the actual LLM output
with open('results/strict_pipeline_testing/strict_pipeline_test_results.json', 'r') as f:
    actual_data = json.load(f)

# Extract the test data
gold_entities = gold_data['gold_standard']['breast_cancer_her2_positive']['expected_extraction']['entities']
actual_entities = actual_data[0]['extraction_result']['entities']

# Test the metrics calculation
metrics = calculate_extraction_metrics(actual_entities, gold_entities)

print('Extraction Metrics:')
print(f'Precision: {metrics["precision"]}')
print(f'Recall: {metrics["recall"]}')
print(f'F1 Score: {metrics["f1"]}')
print(f'True Positives: {metrics["true_positives"]}')
print(f'False Positives: {metrics["false_positives"]}')
print(f'False Negatives: {metrics["false_negatives"]}')
print(f'Actual entities: {metrics["actual_count"]}')
print(f'Expected entities: {metrics["expected_count"]}')

# Show detailed comparison
print('\nDetailed comparison:')
print('Expected entities not found (false negatives):')
for entity in gold_entities:
    if entity['text'].lower().strip() not in [e['text'].lower().strip() for e in actual_entities]:
        print(f"  - {entity['text']}")

print('\nActual entities not expected (false positives):')
for entity in actual_entities:
    if entity['text'].lower().strip() not in [e['text'].lower().strip() for e in gold_entities]:
        print(f"  - {entity['text']}")