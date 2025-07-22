"""
Unit tests for embeddings module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from instant_db.core.embeddings import (
    EmbeddingProvider,
    SentenceTransformerProvider,
    OpenAIProvider
)


class TestEmbeddingProvider:
    """Test the main EmbeddingProvider class"""
    
    def test_sentence_transformer_provider_initialization(self):
        """Test SentenceTransformer provider initialization"""
        provider = EmbeddingProvider(
            provider="sentence-transformers",
            model="all-MiniLM-L6-v2"
        )
        
        assert provider.provider_name == "sentence-transformers"
        assert provider.model_name == "all-MiniLM-L6-v2"
        assert isinstance(provider._provider, SentenceTransformerProvider)
    
    def test_openai_provider_initialization(self):
        """Test OpenAI provider initialization"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = EmbeddingProvider(
                provider="openai",
                model="text-embedding-3-small"
            )
            
            assert provider.provider_name == "openai"
            assert provider.model_name == "text-embedding-3-small"
            assert isinstance(provider._provider, OpenAIProvider)
    
    def test_unsupported_provider(self):
        """Test unsupported provider raises error"""
        with pytest.raises(ValueError, match="Unsupported provider"):
            EmbeddingProvider(provider="unsupported")
    
    @patch('instant_db.core.embeddings.SentenceTransformer')
    def test_encode_with_cache(self, mock_st):
        """Test encoding with caching"""
        # Mock the SentenceTransformer
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_st.return_value = mock_model
        
        provider = EmbeddingProvider(provider="sentence-transformers")
        
        # First call should hit the model
        texts = ["hello world", "goodbye world"]
        embeddings1 = provider.encode(texts)
        
        assert embeddings1.shape == (2, 3)
        assert mock_model.encode.call_count == 1
        
        # Second call with same texts should use cache
        embeddings2 = provider.encode(texts)
        
        assert np.array_equal(embeddings1, embeddings2)
        assert mock_model.encode.call_count == 1  # Should not increase
        
        # Check cache stats
        stats = provider.get_cache_stats()
        assert stats['cache_hits'] == 2  # Both texts from second call
        assert stats['cache_misses'] == 2  # Both texts from first call
        assert stats['hit_rate'] == 0.5
    
    @patch('instant_db.core.embeddings.SentenceTransformer')
    def test_encode_without_cache(self, mock_st):
        """Test encoding without cache"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st.return_value = mock_model
        
        provider = EmbeddingProvider(provider="sentence-transformers")
        
        texts = ["hello world"]
        embeddings = provider.encode(texts, use_cache=False)
        
        assert embeddings.shape == (1, 3)
        assert mock_model.encode.call_count == 1
        
        # Second call should hit model again
        provider.encode(texts, use_cache=False)
        assert mock_model.encode.call_count == 2
    
    @patch('instant_db.core.embeddings.SentenceTransformer')
    def test_get_dimension(self, mock_st):
        """Test getting embedding dimension"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
        mock_st.return_value = mock_model
        
        provider = EmbeddingProvider(provider="sentence-transformers")
        
        dimension = provider.get_dimension()
        assert dimension == 4
    
    def test_clear_cache(self):
        """Test clearing cache"""
        with patch('instant_db.core.embeddings.SentenceTransformer'):
            provider = EmbeddingProvider(provider="sentence-transformers")
            
            # Add some items to cache
            provider._cache["test1"] = np.array([1, 2, 3])
            provider._cache["test2"] = np.array([4, 5, 6])
            provider._cache_hits = 5
            provider._cache_misses = 3
            
            provider.clear_cache()
            
            assert len(provider._cache) == 0
            assert provider._cache_hits == 0
            assert provider._cache_misses == 0
    
    def test_empty_text_list(self):
        """Test encoding empty text list"""
        with patch('instant_db.core.embeddings.SentenceTransformer'):
            provider = EmbeddingProvider(provider="sentence-transformers")
            
            result = provider.encode([])
            assert isinstance(result, np.ndarray)
            assert result.size == 0


class TestSentenceTransformerProvider:
    """Test SentenceTransformer provider directly"""
    
    @patch('instant_db.core.embeddings.SentenceTransformer')
    def test_initialization(self, mock_st):
        """Test provider initialization"""
        mock_model = Mock()
        mock_st.return_value = mock_model
        
        provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        
        assert provider.model_name == "all-MiniLM-L6-v2"
        assert provider.model == mock_model
        mock_st.assert_called_once_with("all-MiniLM-L6-v2")
    
    def test_missing_sentence_transformers(self):
        """Test missing sentence-transformers library"""
        with patch('instant_db.core.embeddings.SentenceTransformer', side_effect=ImportError):
            with pytest.raises(ImportError, match="sentence-transformers not installed"):
                SentenceTransformerProvider()
    
    @patch('instant_db.core.embeddings.SentenceTransformer')
    def test_encode_empty_list(self, mock_st):
        """Test encoding empty list"""
        mock_model = Mock()
        mock_st.return_value = mock_model
        
        provider = SentenceTransformerProvider()
        result = provider.encode([])
        
        assert isinstance(result, np.ndarray)
        assert result.size == 0
        mock_model.encode.assert_not_called()


class TestOpenAIProvider:
    """Test OpenAI provider directly"""
    
    @patch('instant_db.core.embeddings.openai.OpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_initialization(self, mock_openai):
        """Test OpenAI provider initialization"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        provider = OpenAIProvider()
        
        assert provider.model_name == "text-embedding-3-small"
        assert provider.api_key == "test-key"
        assert provider.client == mock_client
    
    def test_missing_api_key(self):
        """Test missing API key"""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key required"):
                OpenAIProvider()
    
    def test_missing_openai_library(self):
        """Test missing openai library"""
        with patch('instant_db.core.embeddings.openai', None):
            with patch.dict('instant_db.core.embeddings.__dict__', {'openai': None}):
                with pytest.raises(ImportError, match="openai not installed"):
                    OpenAIProvider(api_key="test")
    
    @patch('instant_db.core.embeddings.openai.OpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_encode(self, mock_openai):
        """Test OpenAI encoding"""
        # Mock the API response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        provider = OpenAIProvider()
        
        texts = ["hello", "world"]
        embeddings = provider.encode(texts)
        
        assert embeddings.shape == (2, 3)
        assert np.allclose(embeddings[0], [0.1, 0.2, 0.3])
        assert np.allclose(embeddings[1], [0.4, 0.5, 0.6])
        
        mock_client.embeddings.create.assert_called_once_with(
            input=texts,
            model="text-embedding-3-small"
        )
    
    @patch('instant_db.core.embeddings.openai.OpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_encode_api_error(self, mock_openai):
        """Test OpenAI API error handling"""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        provider = OpenAIProvider()
        
        with pytest.raises(RuntimeError, match="OpenAI API error"):
            provider.encode(["test"])
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_get_dimension(self):
        """Test getting dimension for different models"""
        with patch('instant_db.core.embeddings.openai.OpenAI'):
            provider = OpenAIProvider(model_name="text-embedding-3-small")
            assert provider.get_dimension() == 1536
            
            provider = OpenAIProvider(model_name="text-embedding-3-large")
            assert provider.get_dimension() == 3072
            
            provider = OpenAIProvider(model_name="unknown-model")
            assert provider.get_dimension() == 1536  # Default 