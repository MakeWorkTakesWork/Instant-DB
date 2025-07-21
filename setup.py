#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="instant-db",
    version="1.0.0",
    author="Instant-DB Team",
    description="Transform documents into intelligent knowledge graphs with graph-enhanced search",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MakeWorkTakesWork/Instant-DB",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "sentence-transformers>=2.2.2",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "chromadb>=0.4.15",
        "faiss-cpu>=1.7.4",
        "networkx>=3.0",
        "openai>=1.0.0",
        "flask>=2.3.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
    ],
    keywords=["rag", "vector-database", "knowledge-graph", "semantic-search", "ai", "nlp"],
)
