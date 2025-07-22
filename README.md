# 🚀 Instant-DB: Documents to Searchable RAG Database in Minutes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://github.com/MakeWorkTakesWork/Instant-DB)

Transform any collection of documents into a production-ready, searchable RAG database with semantic search capabilities. Perfect for sales teams, knowledge workers, and anyone who needs to make their documents AI-searchable.

## ⚡ What is Instant-DB?

Instant-DB takes your **existing document processing pipeline** and adds **intelligent graph memory** with advanced semantic search. Unlike simple vector databases, Instant-DB builds knowledge graphs that understand relationships, concepts, and context. In minutes, not days.

### **Input** → **Output**
```
📄 Documents (.pdf, .docx, .ppt)     →     🧠 Intelligent Knowledge Graph
📝 Parsed Text & Metadata           →     🔗 Entity Relationships & Concepts  
🔧 Manual Knowledge Management       →     ⚡ Graph-Enhanced Semantic Search
```

## 🎯 Perfect For

- **🏢 SaaS Sales Teams**: Search across pitch decks, objection handling, product docs
- **📚 Knowledge Workers**: Research papers, documentation, meeting notes
- **🎓 Students/Researchers**: Academic papers, study materials, personal notes
- **💼 Consultants**: Client materials, case studies, methodologies

## ✨ Key Features

### **📊 Multiple Vector Database Support**
- **ChromaDB** - Best for most users (default)
- **FAISS** - High performance for large datasets  
- **SQLite** - Simple, portable for small collections

### **🧠 Multiple Embedding Providers**
- **Sentence Transformers** - Free, runs locally
- **OpenAI Embeddings** - Premium quality, API-based

### **🔗 Ready-to-Use Integrations**
- **Custom GPT** - Upload and use immediately
- **Local LLMs** - Ollama, LM Studio integration
- **API Server** - REST endpoints for custom apps
- **Export Formats** - Ready for LangChain, LlamaIndex

### **🧠 Graph Memory Features**
- **Entity Extraction**: Automatically identifies people, products, metrics, concepts
- **Relationship Discovery**: Maps connections between entities across documents
- **Concept Formation**: Creates higher-level abstractions and memory clusters
- **Graph-Enhanced Search**: Finds answers using relationship reasoning
- **Context Awareness**: Understands how information relates to build insights

### **📈 Production Features**
- Smart chunking with section awareness
- Relevance scoring and ranking
- Document type classification
- Metadata preservation
- Incremental updates
- Batch processing

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install sentence-transformers chromadb numpy pandas
```

### 2. Process Your Documents
```bash
# Single document
python instant_db.py process document.pdf

# Entire directory
python instant_db.py process ./sales-materials --batch

# With OpenAI embeddings (higher quality)
python instant_db.py process ./docs --embedding-provider openai
```

### 3. Search Your Knowledge Base
```bash
# Command line search
python instant_db.py search "pricing objection handling"

# Interactive search
python instant_db.py search --interactive

# Graph-enhanced search (relationship reasoning)
python instant_db.py search "pricing objections" --graph --include-relationships
```

### 4. Export for Custom GPT
```bash
# Create file ready for ChatGPT upload
python instant_db.py export --format custom-gpt
```

## 📋 Complete Workflow Example

Starting from **raw documents** to **searchable database**:

```bash
# 1. Process documents (works with existing MegaParse outputs too!)
python instant_db.py process ./my-documents --batch

# 2. Search your knowledge base
python instant_db.py search "customer onboarding process"

# 3. Export for team use
python instant_db.py export --format custom-gpt
python instant_db.py export --format api-server

# 4. Get database stats
python instant_db.py stats
```

## 🏗️ Architecture

```
Documents → Text Extraction → Smart Chunking → Graph Memory Engine → Intelligent Database
    ↓            ↓               ↓                    ↓                      ↓
Raw Files    Clean Text      Section-Aware      Entity Extraction      Knowledge Graph
(.pdf/.docx)  + Metadata     Chunks + Overlap   Relationships          + Vector Search
                                               Concept Formation       + Reasoning
```

## 📊 Performance Benchmarks

| Dataset Size | Processing Time | Search Time | Memory Usage |
|--------------|----------------|-------------|--------------|
| 10 documents | 2-5 minutes | <100ms | 100MB |
| 100 documents | 15-30 minutes | <200ms | 500MB |
| 1000+ documents | 1-3 hours | <300ms | 2GB |

## 🔍 Search Quality Examples

**Query**: "How to handle pricing objections"

**Traditional keyword search** finds:
- ❌ Documents containing "pricing" AND "objections"

**Instant-DB graph-enhanced search** finds:
- ✅ "Addressing cost concerns with ROI data" (+ connected to ROI metrics)
- ✅ "When prospects have budget constraints" (+ related to budget processes)
- ✅ "Value-based selling techniques" (+ linked to product benefits)
- ✅ "Overcoming price resistance" (+ connected to competitive pricing)
- 🧠 **Plus relationship context**: Shows how pricing connects to products, competitors, and outcomes

## 🛠️ Integration Options

### **🤖 Custom GPT (Easiest)**
```bash
python instant_db.py export --format custom-gpt
# Upload to https://chat.openai.com/gpts/editor
```

### **🏠 Local LLM (Most Private)**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2

# Use with your database
python instant_db.py serve --llm ollama
```

### **🌐 OpenAI API (Best Quality)**
```python
from instant_db import InstantDB

db = InstantDB("./my_database")
context = db.search("pricing objections", top_k=3)

# Send context + question to GPT-4
response = openai.chat.completions.create(...)
```

### **🔌 API Server (For Apps)**
```bash
python instant_db.py serve --api
# REST API available at http://localhost:5000
```

## 📁 Repository Structure

```
instant-db/
├── instant_db/
│   ├── __init__.py
│   ├── core/
│   │   ├── database.py          # Vector database management
│   │   ├── embeddings.py        # Embedding providers
│   │   ├── chunking.py          # Smart text chunking
│   │   └── search.py            # Semantic search engine
│   ├── integrations/
│   │   ├── custom_gpt.py        # Custom GPT export
│   │   ├── ollama.py            # Local LLM integration
│   │   ├── openai_api.py        # OpenAI API integration
│   │   └── api_server.py        # REST API server
│   ├── processors/
│   │   ├── document.py          # Document processing
│   │   ├── megaparse.py         # MegaParse integration
│   │   └── batch.py             # Batch processing
│   └── utils/
│       ├── config.py            # Configuration management
│       ├── logging.py           # Logging utilities
│       └── stats.py             # Database statistics
├── examples/
│   ├── sales_team_setup.py     # Sales team example
│   ├── research_workflow.py    # Research workflow
│   └── custom_integration.py   # Custom integration example
├── tests/
├── docs/
├── requirements.txt
├── setup.py
├── instant_db.py              # Main CLI interface
└── README.md
```

## 🎯 Use Cases

### **Sales Team Knowledge Base**
```bash
# Process all sales materials
python instant_db.py process ./sales-collateral --batch

# Search for specific scenarios
python instant_db.py search "competitive objections"
python instant_db.py search "ROI calculations" 
python instant_db.py search "pricing strategies"

# Export for team Custom GPT
python instant_db.py export --format custom-gpt --name "Sales Assistant"
```

### **Research Paper Collection**
```bash
# Process academic papers
python instant_db.py process ./research-papers --batch --embedding-provider openai

# Advanced search with filtering
python instant_db.py search "machine learning optimization" --document-type "Research Paper"
```

### **Company Knowledge Management**
```bash
# Process internal documentation
python instant_db.py process ./company-docs --batch

# Create API for internal tools
python instant_db.py serve --api --host 0.0.0.0
```

## 🔄 Works With Existing Workflows

**Already using MegaParse?** Perfect! Instant-DB works with your existing outputs:
```bash
# Process existing MegaParse outputs
python instant_db.py process-megaparse ./processed-documents
```

**Have raw documents?** We'll handle the processing:
```bash
# Full pipeline from raw documents
python instant_db.py process ./raw-documents --include-parsing
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install from source (recommended for now)
git clone https://github.com/MakeWorkTakesWork/Instant-DB.git
cd Instant-DB
pip install -e .

# Or with all optional dependencies
pip install -e ".[all]"
```

### 2. Initialize Your Project

```bash
# Create configuration and sample directories
instant-db init
```

### 3. Process Your Documents

```bash
# Single document
instant-db process document.pdf

# Entire directory 
instant-db process ./documents --batch

# With custom settings
instant-db process ./documents --batch --chunk-size 800 --workers 4
```

### 4. Search Your Knowledge Base

```bash
# Simple search
instant-db search "machine learning concepts"

# Interactive search mode
instant-db search --interactive

# Export for Custom GPT
instant-db export --format markdown --split-by-type
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

We welcome contributions! See CONTRIBUTING.md for guidelines.

---

**Transform your documents into an intelligent, searchable knowledge base in minutes, not days.** 