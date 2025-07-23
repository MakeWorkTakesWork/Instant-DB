# Instant-DB Benchmarks

This directory contains benchmark scripts to evaluate Instant-DB's search quality and performance.

## Quick Start

Run the simple benchmark using the demo dataset:

```bash
cd /path/to/Instant-DB
python benchmarks/simple_benchmark.py
```

## What It Tests

The simple benchmark evaluates:

1. **Search Accuracy** - How well queries find relevant content
2. **Precision** - Percentage of relevant results in top results
3. **Performance** - Search response time
4. **Keyword Matching** - Presence of expected terms in results

## Scoring System

Each query is scored out of 100 points:
- **40 points**: Finding content from the expected file
- **30 points**: Keyword matches (10 points per keyword, max 30)
- **20 points**: Precision (percentage of relevant results)
- **10 points**: Similarity score quality

## Performance Ratings

- 🌟 **EXCELLENT**: 90-100 (Very accurate search results)
- ✅ **GOOD**: 75-89 (Reliable for most use cases)
- ⚠️  **FAIR**: 60-74 (Acceptable but could improve)
- ❌ **NEEDS IMPROVEMENT**: Below 60 (Significant issues)

## Test Queries

The benchmark tests common search scenarios:
- Finding pricing objection responses
- Locating security features
- Retrieving action items from meetings
- Finding product benefits
- Addressing migration concerns

## Output

Results are displayed in the console and saved to `benchmark_results.json` with:
- Individual query scores and metrics
- Overall performance summary
- Search timing information
- Result previews

## Extending the Benchmark

To add more test cases, edit `get_test_queries()` in `simple_benchmark.py`:

```python
{
    "query": "your test query",
    "expected_keywords": ["keyword1", "keyword2"],
    "expected_file": "filename.txt",
    "description": "What this test validates"
}
```

## Future Enhancements

- Add more complex multi-hop queries
- Test graph-enhanced search capabilities
- Benchmark different embedding models
- Compare vector database performance
- Add recall metrics
- Test with larger datasets