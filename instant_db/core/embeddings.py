"""
Embedding provider abstraction for Instant-DB
Supports multiple embedding providers with unified interface
"""

import os
import hashlib
from typing import List, Optional, Any, Dict
from abc import ABC, abstractmethod

import numpy as np
from pathlib import Path


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Sentence Transformers embedding provider (local, free)"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._dimension = None
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Sentence Transformers"""
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self._dimension is None:
            # Test with a dummy text to get dimension
            test_embedding = self.encode(["test"])
            self._dimension = test_embedding.shape[1] if len(test_embedding) > 0 else 384
        return self._dimension


class OpenAIProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider (API-based, premium)"""
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai not installed. "
                "Install with: pip install openai"
            )
        
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter"
            )
        
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Model dimensions
        self._model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using OpenAI API"""
        if not texts:
            return np.array([])
        
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            
            embeddings = []
            for item in response.data:
                embeddings.append(item.embedding)
            
            return np.array(embeddings)
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self._model_dimensions.get(self.model_name, 1536)


class EmbeddingProvider:
    """
    Main embedding provider factory and interface
    Manages different embedding providers with caching
    """
    
    def __init__(self, provider: str = "sentence-transformers", model: str = None, **kwargs):
        """
        Initialize embedding provider
        
        Args:
            provider: 'sentence-transformers' or 'openai'
            model: Model name for the provider
            **kwargs: Additional provider-specific arguments
        """
        self.provider_name = provider
        self.model_name = model
        self.kwargs = kwargs
        
        # Initialize the actual provider
        if provider == "sentence-transformers":
            model = model or "all-MiniLM-L6-v2"
            self._provider = SentenceTransformerProvider(model)
        elif provider == "openai":
            model = model or "text-embedding-3-small"
            self._provider = OpenAIProvider(model, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.model_name = model
        
        # Embedding cache
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def encode(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings with optional caching
        
        Args:
            texts: List of texts to encode
            use_cache: Whether to use embedding cache
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        if not use_cache:
            return self._provider.encode(texts)
        
        # Check cache for each text
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            
            if text_hash in self._cache:
                cached_embeddings.append((i, self._cache[text_hash]))
                self._cache_hits += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                self._cache_misses += 1
        
        # Encode uncached texts
        new_embeddings = []
        if uncached_texts:
            new_embeddings = self._provider.encode(uncached_texts)
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                text_hash = self._get_text_hash(text)
                self._cache[text_hash] = embedding
        
        # Combine cached and new embeddings in correct order
        all_embeddings = [None] * len(texts)
        
        # Place cached embeddings
        for idx, embedding in cached_embeddings:
            all_embeddings[idx] = embedding
        
        # Place new embeddings
        for i, embedding in enumerate(new_embeddings):
            original_idx = uncached_indices[i]
            all_embeddings[original_idx] = embedding
        
        return np.array(all_embeddings)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self._provider.get_dimension()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }
    
    def clear_cache(self):
        """Clear embedding cache"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching"""
        # Include provider and model in hash for cache key uniqueness
        content = f"{self.provider_name}:{self.model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def __repr__(self):
        return f"EmbeddingProvider(provider={self.provider_name}, model={self.model_name})" 