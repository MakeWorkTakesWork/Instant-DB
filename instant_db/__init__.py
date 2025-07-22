"""
Instant-DB: Documents to Searchable RAG Database in Minutes

Transform any collection of documents into a production-ready, 
searchable RAG database with semantic search capabilities.
"""

__version__ = "1.0.0"
__author__ = "Instant-DB Contributors"
__license__ = "MIT"

from .core.database import InstantDB
from .core.search import SearchEngine
from .core.embeddings import EmbeddingProvider
from .processors.document import DocumentProcessor
from .integrations.custom_gpt import CustomGPTExporter

__all__ = [
    'InstantDB',
    'SearchEngine', 
    'EmbeddingProvider',
    'DocumentProcessor',
    'CustomGPTExporter',
    '__version__'
] 