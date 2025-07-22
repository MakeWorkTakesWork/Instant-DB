# Contributing to Instant-DB

🎉 Thank you for considering contributing to Instant-DB! We welcome contributions from everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Guidelines](#coding-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Standards

- **Be inclusive**: Welcome newcomers and encourage diverse perspectives
- **Be respectful**: Treat everyone with respect and professionalism
- **Be collaborative**: Work together to improve the project
- **Be helpful**: Share knowledge and assist others
- **Be constructive**: Provide actionable feedback

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of vector databases and semantic search

### First Time Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/Instant-DB.git
   cd Instant-DB
   ```

3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/MakeWorkTakesWork/Instant-DB.git
   ```

4. **Set up development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev,document-processing,performance]"
   ```

5. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

## Development Setup

### Environment Setup

Create a `.env` file for development:
```bash
# Optional: OpenAI API key for testing
OPENAI_API_KEY=your_api_key_here

# Development settings
INSTANT_DB_DEBUG=true
INSTANT_DB_LOG_LEVEL=DEBUG
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=instant_db

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run specific test file
pytest tests/unit/test_embeddings.py

# Run with verbose output
pytest -v
```

### Code Quality Tools

```bash
# Format code with black
black instant_db tests

# Check code style
flake8 instant_db tests

# Type checking
mypy instant_db

# Run all quality checks
make lint  # If Makefile exists
```

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

- 🐛 **Bug Reports**: Help us identify and fix issues
- 🚀 **Feature Requests**: Suggest new features or improvements
- 📝 **Documentation**: Improve docs, add examples, write tutorials
- 🔧 **Code Contributions**: Fix bugs, add features, improve performance
- 🧪 **Testing**: Add tests, improve test coverage
- 🎨 **Design**: UI/UX improvements for CLI or documentation

### Before You Start

1. **Check existing issues** to avoid duplicates
2. **Open an issue** to discuss major changes
3. **Keep contributions focused** - one feature/fix per PR
4. **Follow the coding guidelines** below

## Coding Guidelines

### Python Code Style

- **Follow PEP 8** with these modifications:
  - Line length: 100 characters (configured in `pyproject.toml`)
  - Use `black` for automatic formatting
  - Use type hints for all functions and methods

### Code Organization

```python
# Import order (use isort)
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from instant_db.core.database import InstantDB
from instant_db.utils.logging import get_logger

# Use descriptive variable names
embedding_provider = "sentence-transformers"
chunk_size = 1000

# Add docstrings to all public functions
def process_document(file_path: Path, chunk_size: int = 1000) -> Dict[str, Any]:
    """
    Process a document and return results.
    
    Args:
        file_path: Path to the document file
        chunk_size: Size of text chunks in characters
        
    Returns:
        Dictionary with processing results
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If chunk_size is invalid
    """
    pass
```

### Error Handling

```python
# Use specific exceptions
raise ValueError(f"Invalid chunk size: {chunk_size}")

# Log errors appropriately
logger = get_logger(__name__)
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise
```

### Configuration

- Use the `Config` class for settings
- Support environment variables
- Provide sensible defaults
- Document all configuration options

## Testing

### Test Categories

1. **Unit Tests** (`tests/unit/`): Test individual components
2. **Integration Tests** (`tests/integration/`): Test component interactions
3. **End-to-End Tests**: Test full workflows (if applicable)

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

def test_function_behavior():
    """Test that function behaves correctly under normal conditions."""
    # Arrange
    input_data = "test input"
    expected_output = "expected result"
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_output

def test_function_error_handling():
    """Test that function handles errors appropriately."""
    with pytest.raises(ValueError, match="Invalid input"):
        function_under_test(invalid_input)

@patch('instant_db.core.embeddings.SentenceTransformer')
def test_with_mocking(mock_transformer):
    """Test using mocks for external dependencies."""
    mock_transformer.return_value.encode.return_value = np.array([[1, 2, 3]])
    # Test code here
```

### Test Guidelines

- **Test one thing per test**: Keep tests focused
- **Use descriptive names**: Test name should describe what it tests
- **Follow AAA pattern**: Arrange, Act, Assert
- **Mock external dependencies**: Don't depend on external services
- **Test edge cases**: Empty inputs, large inputs, error conditions
- **Maintain test independence**: Tests should not depend on each other

## Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings, type hints, comments
2. **API Documentation**: Function/class documentation
3. **User Documentation**: README, tutorials, examples
4. **Developer Documentation**: Architecture, contributing guides

### Documentation Standards

- **Write clear docstrings** for all public functions/classes
- **Use type hints** consistently
- **Keep README up-to-date** with new features
- **Add examples** for new functionality
- **Document breaking changes** in CHANGELOG.md

### Adding Examples

Create examples in the `examples/` directory:

```python
# examples/basic_usage.py
"""
Basic usage example for Instant-DB
"""

from instant_db import InstantDB

# Initialize database
db = InstantDB(
    embedding_provider="sentence-transformers",
    vector_db="chroma"
)

# Process documents
result = db.process_document("path/to/document.txt")
print(f"Processed {result['chunks_processed']} chunks")

# Search
results = db.search("machine learning", top_k=5)
for result in results:
    print(f"Found: {result['content'][:100]}...")
```

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**: `pytest`
2. **Check code quality**: `flake8`, `mypy`, `black`
3. **Update documentation** if needed
4. **Add changelog entry** for user-facing changes
5. **Rebase on latest main**: `git rebase upstream/main`

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated  
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changelog updated (if user-facing)
- [ ] Tests pass locally
```

### Review Process

1. **Automated checks** must pass
2. **At least one maintainer** review required
3. **Address feedback** promptly and professionally
4. **Squash commits** if requested
5. **Maintainer will merge** when approved

## Issue Guidelines

### Bug Reports

Use the bug report template and include:

- **Environment details**: OS, Python version, package versions
- **Steps to reproduce**: Minimal example that shows the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Additional context**: Screenshots, logs, error messages

### Feature Requests

Use the feature request template and include:

- **Problem description**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you've thought about
- **Additional context**: Use cases, examples

### Questions

For questions:
- Check existing documentation first
- Search existing issues
- Use clear, specific titles
- Provide context about what you're trying to achieve

## Development Workflow

### Branch Naming

- `feature/description`: New features
- `fix/description`: Bug fixes
- `docs/description`: Documentation updates
- `test/description`: Test improvements

### Commit Messages

Follow conventional commits format:

```
type(scope): description

Longer explanation if needed

- List of changes
- Another change
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`

Examples:
```
feat(cli): add interactive search mode
fix(embeddings): handle empty text input
docs(readme): add installation instructions
test(chunking): add edge case tests
```

## Getting Help

- 📖 **Documentation**: Check the README and docs
- 💬 **Discussions**: Use GitHub Discussions for questions
- 🐛 **Issues**: Use GitHub Issues for bugs and feature requests
- 📧 **Email**: Contact maintainers for sensitive issues

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Project documentation

Thank you for contributing to Instant-DB! 🚀 