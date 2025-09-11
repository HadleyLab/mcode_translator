# mCODE Extraction Performance Comparison Report

Generated: 2025-09-09 18:17:50

## Summary

- **Total Comparisons**: 15
- **Overall Average Precision**: 0.107
- **Overall Average Recall**: 0.235
- **Overall Average F1-Score**: 0.144

## Model Performance

| Model | Avg Precision | Avg Recall | Avg F1-Score | Trials |
|-------|---------------|------------|--------------|--------|
| chat | 0.109 | 0.240 | 0.146 | 5 |
| coder | 0.104 | 0.240 | 0.144 | 5 |
| reasoner | 0.107 | 0.226 | 0.143 | 5 |

## Detailed Results

| Trial | Model | Precision | Recall | F1-Score | TP | FP | FN |
|-------|-------|-----------|--------|----------|----|----|----|
| NCT00109785 | chat | 0.231 | 0.333 | 0.273 | 3 | 10 | 6 |
| NCT00109785 | coder | 0.176 | 0.333 | 0.231 | 3 | 14 | 6 |
| NCT00109785 | reasoner | 0.167 | 0.222 | 0.190 | 2 | 10 | 7 |
| NCT00616135 | chat | 0.045 | 0.143 | 0.069 | 1 | 21 | 6 |
| NCT00616135 | coder | 0.059 | 0.143 | 0.083 | 1 | 16 | 6 |
| NCT00616135 | reasoner | 0.059 | 0.143 | 0.083 | 1 | 16 | 6 |
| NCT01026116 | chat | 0.100 | 0.250 | 0.143 | 2 | 18 | 6 |
| NCT01026116 | coder | 0.105 | 0.250 | 0.148 | 2 | 17 | 6 |
| NCT01026116 | reasoner | 0.158 | 0.375 | 0.222 | 3 | 16 | 5 |
| NCT01922921 | chat | 0.091 | 0.308 | 0.140 | 4 | 40 | 9 |
| NCT01922921 | coder | 0.098 | 0.308 | 0.148 | 4 | 37 | 9 |
| NCT01922921 | reasoner | 0.108 | 0.308 | 0.160 | 4 | 33 | 9 |
| NCT06650748 | chat | 0.077 | 0.167 | 0.105 | 2 | 24 | 10 |
| NCT06650748 | coder | 0.083 | 0.167 | 0.111 | 2 | 22 | 10 |
| NCT06650748 | reasoner | 0.045 | 0.083 | 0.059 | 1 | 21 | 11 |

## Recommendations

1. **Best Performing Model**: Based on F1-score
2. **Areas for Improvement**: Analysis of common false positives/negatives
3. **Next Steps**: Fine-tuning prompts for specific mCODE elements
