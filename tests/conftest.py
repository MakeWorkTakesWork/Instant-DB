"""
Pytest configuration and shared fixtures for Instant-DB tests
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any

from instant_db.core.database import InstantDB
from instant_db.core.embeddings import EmbeddingProvider
from instant_db.core.chunking import ChunkingEngine
from instant_db.processors.document import DocumentProcessor
from instant_db.utils.config import Config


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing"""
    return """
    # Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence (AI) that focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.

    ## Types of Machine Learning

    ### Supervised Learning
    Supervised learning uses labeled datasets to train algorithms to classify data or predict outcomes accurately.

    ### Unsupervised Learning  
    Unsupervised learning uses machine learning algorithms to analyze and cluster unlabeled datasets.

    ### Reinforcement Learning
    Reinforcement learning is a machine learning training method based on rewarding desired behaviors and/or punishing undesired ones.

    ## Applications

    Machine learning has applications in many fields including:
    - Healthcare and medical diagnosis
    - Financial services and fraud detection  
    - Transportation and autonomous vehicles
    - Natural language processing
    - Computer vision and image recognition

    ## Conclusion

    As data availability continues to grow exponentially, machine learning will become increasingly important for businesses and society.
    """


@pytest.fixture
def sample_documents() -> Dict[str, str]:
    """Sample documents for testing"""
    return {
        "ml_basics.txt": """
        Machine learning is a method of data analysis that automates analytical model building. 
        It is a branch of artificial intelligence based on the idea that systems can learn from data, 
        identify patterns and make decisions with minimal human intervention.
        """,
        "ai_overview.md": """
        # Artificial Intelligence Overview
        
        Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural 
        intelligence displayed by humans. AI research has been highly successful in developing effective 
        techniques for solving a wide range of problems.
        
        ## Key Areas
        - Machine Learning
        - Natural Language Processing
        - Computer Vision
        - Robotics
        """,
        "data_science.txt": """
        Data science is an inter-disciplinary field that uses scientific methods, processes, algorithms 
        and systems to extract knowledge and insights from many structural and unstructured data.
        
        Data science combines domain expertise, programming skills, and knowledge of mathematics and 
        statistics to extract meaningful insights from data.
        """
    }


@pytest.fixture
def embedding_provider() -> EmbeddingProvider:
    """Create a test embedding provider"""
    return EmbeddingProvider(
        provider="sentence-transformers",
        model="all-MiniLM-L6-v2"
    )


@pytest.fixture
def chunking_engine() -> ChunkingEngine:
    """Create a test chunking engine"""
    return ChunkingEngine(
        chunk_size=500,
        chunk_overlap=100,
        min_chunk_size=50
    )


@pytest.fixture
def test_database(temp_dir: Path) -> InstantDB:
    """Create a test database instance"""
    return InstantDB(
        db_path=str(temp_dir / "test_db"),
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        vector_db="sqlite"  # Use SQLite for faster tests
    )


@pytest.fixture
def document_processor(temp_dir: Path) -> DocumentProcessor:
    """Create a test document processor"""
    return DocumentProcessor(
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        vector_db="sqlite",
        chunk_size=500,
        chunk_overlap=100
    )


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    """Create a test configuration"""
    config = Config()
    config.database.path = str(temp_dir / "test_db")
    config.database.vector_db = "sqlite"
    config.processing.max_workers = 2
    config.processing.skip_errors = True
    return config


@pytest.fixture
def create_sample_files(temp_dir: Path, sample_documents: Dict[str, str]) -> Path:
    """Create sample files for testing"""
    docs_dir = temp_dir / "sample_docs"
    docs_dir.mkdir()
    
    for filename, content in sample_documents.items():
        file_path = docs_dir / filename
        file_path.write_text(content.strip())
    
    return docs_dir


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration between tests"""
    import logging
    
    # Clear all handlers from the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Reset level
    root_logger.setLevel(logging.WARNING)
    
    # Clear instant_db loggers
    instant_db_logger = logging.getLogger('instant_db')
    for handler in instant_db_logger.handlers[:]:
        instant_db_logger.removeHandler(handler)
    
    yield
    
    # Cleanup after test
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    for handler in instant_db_logger.handlers[:]:
        instant_db_logger.removeHandler(handler)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    import warnings
    
    # Filter out common warnings during testing
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers for slow tests
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit) 