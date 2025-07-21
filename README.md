# 🚀 Instant-DB: Documents to Intelligent Knowledge Graph in Minutes

Transform any collection of documents into a production-ready, searchable knowledge graph with graph-enhanced semantic search. Unlike simple vector databases, Instant-DB builds intelligent knowledge graphs that understand relationships, concepts, and context.

## ⚡ What is Instant-DB?

Instant-DB takes your **existing document processing pipeline** and adds **intelligent graph memory** with advanced semantic search. Unlike simple vector databases, Instant-DB builds knowledge graphs that understand relationships, concepts, and context. In minutes, not days.

### **Input** → **Output**
```
📄 Documents (.pdf, .docx, .ppt)     →     🧠 Intelligent Knowledge Graph
📝 Parsed Text & Metadata           →     🔗 Entity Relationships & Concepts  
�� Manual Knowledge Management       →     ⚡ Graph-Enhanced Semantic Search
```

## 🎯 Perfect For

- **🏢 SaaS Sales Teams**: Search across pitch decks, objection handling, product docs
- **📚 Knowledge Workers**: Research papers, documentation, meeting notes
- **🎓 Students/Researchers**: Academic papers, study materials, personal notes
- **💼 Consultants**: Client materials, case studies, methodologies

## ✨ Key Features

### **🧠 Graph Memory Features**
- **Entity Extraction**: Automatically identifies people, products, metrics, concepts
- **Relationship Discovery**: Maps connections between entities across documents
- **Concept Formation**: Creates higher-level abstractions and memory clusters
- **Graph-Enhanced Search**: Finds answers using relationship reasoning
- **Context Awareness**: Understands how information relates to build insights

### **📊 Multiple Vector Database Support**
- **ChromaDB** - Best for most users (default)
- **FAISS** - High performance for large datasets  
- **SQLite** - Simple, portable for small collections

### **🧠 Multiple Embedding Providers**
- **Sentence Transformers** - Free, runs locally
- **OpenAI Embeddings** - Premium quality, API-based

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install sentence-transformers chromadb numpy pandas networkx
```

### 2. Process Your Documents
```bash
# Single document
python instant_db.py process document.pdf

# Entire directory with graph memory
python instant_db.py process ./sales-materials --batch

# Graph-enhanced search (relationship reasoning)
python instant_db.py search "pricing objections" --graph --include-relationships
```

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

## 🏗️ Architecture

```
Documents → Text Extraction → Smart Chunking → Graph Memory Engine → Intelligent Database
    ↓            ↓               ↓                    ↓                      ↓
Raw Files    Clean Text      Section-Aware      Entity Extraction      Knowledge Graph
(.pdf/.docx)  + Metadata     Chunks + Overlap   Relationships          + Vector Search
                                               Concept Formation       + Reasoning
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

We welcome contributions! This is the future of knowledge management.

---

**Transform your documents into an intelligent, searchable knowledge base in minutes, not days.**
