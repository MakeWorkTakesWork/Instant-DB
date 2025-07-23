# CLAUDE.md - Instant-DB Project Memory

This file contains important context and learnings for Claude when working on the Instant-DB project.

## Project Overview
Instant-DB is a tool that transforms document collections into searchable RAG databases with semantic search and graph memory capabilities. It's designed for sales teams, knowledge workers, and anyone needing AI-searchable documents.

## Recent Sessions

### 2025-01-23: Security, Benchmarks, Metadata Fix, and MCP Discovery
- Added comprehensive SECURITY.md documentation
- Created benchmarking system in `/benchmarks/`
- **CRITICAL FIX**: Resolved metadata nesting issue that was breaking all searches
  - ChromaDB requires flat metadata (no nested dicts, no None values)
  - Solution: Flatten metadata in `database.py` before passing to vector store
  - Always test with benchmark after changes!
- **MCP DISCOVERY**: Claude Code only supports specific Python-based MCP servers
  - Claude Desktop: All 17 MCP servers work
  - Claude Code: Only gemini-collab and multi-ai-collab work
  - Created template for wrapping MCP servers: `/Users/johnsweazey/claude_code-linkup-mcp/`

### Previous Session (from context-saves)
- Fixed CI/CD platform-specific dependencies (python-magic issue)
- Enhanced documentation with troubleshooting section
- Created demo dataset for quick testing
- Improved installation guide with OS-specific instructions

## Key Technical Details

### Architecture
- **Vector Stores**: ChromaDB (default), FAISS, SQLite
- **Embedding Providers**: sentence-transformers (local), OpenAI
- **Search Engine**: Combines vector search with graph memory
- **Chunking**: Smart text chunking with metadata preservation

### Important Code Patterns

1. **Adding Documents** (correct way):
```python
db = InstantDB(db_path)
result = db.add_document(
    content="text content",
    metadata={"title": "Doc Title", "type": "text"},  # Flat dict only!
    document_id="unique_id"
)
```

2. **Metadata Rules**:
- Must be flat dictionary (no nested objects)
- Only primitive types: str, int, float, bool
- No None values (filter them out)
- ChromaDB will reject nested metadata

3. **Search Results Structure**:
```python
{
    "id": "chunk_id",
    "content": "matching text",
    "similarity": 0.85,
    "document_id": "source_doc",  # Added at top level for convenience
    "metadata": {...}  # Original metadata
}
```

## Common Issues and Solutions

### 1. Metadata Type Errors
**Symptom**: "Expected metadata value to be str, int, float, bool, or None, got dict"
**Solution**: Flatten metadata structure in `database.py` (already fixed)

### 2. CLI Config Error
**Symptom**: "cli() got an unexpected keyword argument 'config'"
**Solution**: Use direct API instead of CLI for now, or fix Click decorator issue

### 3. Search Returns No Results
**Check**:
- Documents actually added (check return status)
- Metadata properly flattened
- Embeddings generated correctly

## Testing and Quality

### Running Benchmarks
```bash
# Quick test
python3 benchmarks/simple_benchmark_minimal.py

# Full benchmark with demo data
python3 benchmarks/simple_benchmark.py
```

### Expected Performance
- Search latency: ~13ms average
- Indexing: Should show "X chunks processed"
- Benchmark score: 60-80% is good for semantic search

## Development Workflow

1. **Before Making Changes**:
   - Check existing tests/benchmarks
   - Understand current architecture
   - Read this file for context

2. **After Making Changes**:
   - Run benchmarks to ensure search still works
   - Check for metadata issues
   - Update this file with new learnings

3. **Common Commands**:
```bash
# Install in dev mode
pip install -e .

# Run tests
pytest tests/

# Check search functionality
python3 benchmarks/simple_benchmark_minimal.py
```

## Future Improvements (Not Yet Implemented)
1. Full benchmarking system with more metrics
2. `instant-db update` command for incremental updates
3. Terminal recording GIF for README
4. Video walkthrough/tutorial
5. Fix CLI config argument issue

## Important Files
- `/instant_db/core/database.py` - Main database interface (has metadata flattening)
- `/instant_db/core/search.py` - Search engine and vector stores
- `/benchmarks/` - Benchmark scripts for testing
- `/demo_dataset/` - Sample data for testing
- `/context-saves/` - Session history and detailed notes

## MCP Server Notes (Claude Code vs Desktop)

### Claude Code Limitations
- Only supports Python-based MCP servers with specific JSON-RPC protocol
- Cannot access local file system, Docker containers, or localhost
- Working servers must follow pattern like `claude_code-*`
- Template for creating compatible servers: `/Users/johnsweazey/claude_code-linkup-mcp/`

### Available in Claude Code
- `mcp__gemini-collab__*` - Gemini AI collaboration
- `mcp__multi-ai-collab__*` - Multi-AI collaboration

### Not Available in Claude Code (Desktop only)
- Zen, Graphiti, filesystem, and other local/Docker-based MCPs
- npm-based MCP servers
- Anything requiring localhost connections

## Contact and Resources
- GitHub: https://github.com/MakeWorkTakesWork/Instant-DB
- Main dependencies: ChromaDB, FAISS, sentence-transformers
- Python 3.8-3.12 supported

---
*Last updated: 2025-01-23 after fixing critical metadata issue and discovering MCP limitations*