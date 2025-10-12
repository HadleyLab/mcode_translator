# Advanced Batch Processing Example

This example demonstrates advanced workflow features including batch processing, engine comparison, performance benchmarking, and result analysis.

## What You'll Learn

- Batch processing multiple clinical trials simultaneously
- Comparing RegexEngine vs LLMEngine performance
- Performance benchmarking and metrics collection
- Result aggregation and statistical analysis
- Error handling and recovery strategies

## Quick Start

```bash
cd examples/advanced_workflow
python advanced_batch_processing.py
```

## Expected Output

```
🚀 mCODE Translator - Advanced Batch Processing
============================================================

🎯 Processing 3 Clinical Trials
────────────────────────────────────────
   1. NCT02364999
   2. NCT02735178
   3. NCT03470922

🔧 Testing REGEX Engine
──────────────────────────────
   ⏱️  Processing Time: 0.45s
   ✅ Success: True

🔧 Testing LLM Engine
──────────────────────────────
   ⏱️  Processing Time: 12.30s
   ✅ Success: True

📊 ENGINE COMPARISON RESULTS
============================================================

Performance Metrics:
Engine      | Time (s) | Success | Trials Processed
------------|----------|---------|------------------
RegexEngine | 0.45     | ✅       | 3
LLMEngine    | 12.30    | ✅       | 3

🔍 REGEXENGINE Engine Detailed Results
────────────────────────────────────────
   • Total mCODE Elements: 15
   • Average Confidence: 94.2%
   • Elements per Trial: 5.0

🔍 LLMENGINE Engine Detailed Results
────────────────────────────────────────
   • Total mCODE Elements: 18
   • Average Confidence: 91.8%
   • Elements per Trial: 6.0

💡 RECOMMENDATIONS
============================================================
⚡ Regex engine is 27.3x faster for this workload
   → Use RegexEngine for speed-critical batch processing

🎯 Use Case Guidance:
   • Large datasets (100+ trials): RegexEngine
   • Complex eligibility criteria: LLMEngine
   • Research analysis: Both engines for comparison
   • Production pipelines: RegexEngine with periodic LLM validation

🎉 Advanced batch processing example completed!
```

## Configuration Options

The example processes these trials:
- **NCT02364999**: PALOMA-2 (Breast Cancer)
- **NCT02735178**: KEYNOTE-042 (Lung Cancer)
- **NCT03470922**: CheckMate 238 (Melanoma)

## Features Demonstrated

### Batch Processing
- Processing multiple trials in a single operation
- Efficient resource utilization
- Parallel processing capabilities

### Engine Comparison
- Side-by-side performance metrics
- Accuracy vs speed trade-offs
- Cost-benefit analysis

### Performance Analysis
- Execution time measurement
- Element extraction statistics
- Confidence score aggregation

### Error Handling
- Graceful failure recovery
- Partial result processing
- Comprehensive error reporting

## Files in This Example

- `advanced_batch_processing.py` - Main example script
- `README.md` - This documentation

## Customization Options

You can modify the example to:

1. **Change Trial Set**: Update the `trial_ids` list
2. **Adjust Engines**: Modify the `engines` list
3. **Batch Sizes**: Change processing batch sizes
4. **Enable Storage**: Set `store_results: True` for CORE Memory
5. **Add Metrics**: Include additional performance measurements

## Performance Insights

Typical performance characteristics:

| Metric | RegexEngine | LLMEngine | Notes |
|--------|-------------|-----------|-------|
| Speed | ~0.15s/trial | ~4s/trial | Network dependent |
| Accuracy | 94% | 92% | Structured vs complex text |
| Cost | $0.00 | $0.05-0.10 | API costs apply |
| Scalability | Excellent | Good | Memory usage |

## Next Steps

After running this example, explore:

1. **Larger Datasets**: Try with 10+ trials
2. **Different Conditions**: Process trials from various cancer types
3. **Custom Metrics**: Add domain-specific performance measurements
4. **Result Persistence**: Enable CORE Memory storage and querying
5. **Pipeline Integration**: Use results in downstream analysis workflows

## Troubleshooting

- **Memory Issues**: Reduce batch size for large trial sets
- **API Limits**: LLM engine may hit rate limits on large batches
- **Network Timeouts**: Increase timeout values for slow connections
- **Storage Errors**: Ensure CORE Memory API key is configured for storage features