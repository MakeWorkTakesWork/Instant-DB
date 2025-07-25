#!/usr/bin/env python3
"""
Simple validation test for specific Instant-DB enhancements
Tests individual components without heavy dependencies
"""

import sys
import os
import tempfile
import sqlite3
from pathlib import Path

# Add the instant_db module to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_faiss_class_structure():
    """Test FAISS class has the new methods"""
    print("🔍 Testing FAISS class structure...")
    try:
        # Import just the search module
        from instant_db.core.search import FAISSVectorStore
        
        # Check if the class has the new methods without instantiating
        required_methods = [
            '_save_embeddings',
            '_load_embeddings',
            'delete_document'
        ]
        
        for method in required_methods:
            if not hasattr(FAISSVectorStore, method):
                print(f"   ❌ Missing method: {method}")
                return False
        
        print("   ✅ FAISS class structure validated")
        return True
    except Exception as e:
        print(f"   ❌ FAISS structure test failed: {e}")
        return False

def test_openai_class_structure():
    """Test OpenAI class has the new methods"""
    print("🔍 Testing OpenAI class structure...")
    try:
        from instant_db.core.embeddings import OpenAIProvider
        
        required_methods = [
            '_get_cache_key',
            '_load_cache', 
            '_save_cache',
            '_encode_single'
        ]
        
        for method in required_methods:
            if not hasattr(OpenAIProvider, method):
                print(f"   ❌ Missing method: {method}")
                return False
        
        print("   ✅ OpenAI class structure validated")
        return True
    except Exception as e:
        print(f"   ❌ OpenAI structure test failed: {e}")
        return False

def test_discovery_class_structure():
    """Test DocumentDiscovery class has the new methods"""
    print("🔍 Testing DocumentDiscovery class structure...")
    try:
        from instant_db.core.discovery import DocumentDiscovery
        
        required_methods = [
            'scan_directory_for_documents_parallel',
            '_scan_directory_recursive',
            'discover_documents',
            '_is_supported_file',
            '_passes_filters'
        ]
        
        for method in required_methods:
            if not hasattr(DocumentDiscovery, method):
                print(f"   ❌ Missing method: {method}")
                return False
        
        print("   ✅ DocumentDiscovery class structure validated")
        return True
    except Exception as e:
        print(f"   ❌ DocumentDiscovery structure test failed: {e}")
        return False

def test_metadata_filter_structure():
    """Test MetadataFilterEngine has the required methods"""
    print("🔍 Testing MetadataFilterEngine structure...")
    try:
        from instant_db.core.metadata_filter import MetadataFilterEngine
        
        # Check if _in method exists
        if not hasattr(MetadataFilterEngine, '_in'):
            print("   ❌ Missing method: _in")
            return False
        
        print("   ✅ MetadataFilterEngine structure validated")
        return True
    except Exception as e:
        print(f"   ❌ MetadataFilterEngine structure test failed: {e}")
        return False

def test_cli_help_text():
    """Test CLI help text has been updated"""
    print("🔍 Testing CLI help text...")
    try:
        # Read the CLI file and check for the manifest format documentation
        cli_file = Path(__file__).parent / "instant_db" / "cli.py"
        with open(cli_file, 'r') as f:
            content = f.read()
        
        if 'Manifest format: {"files": [{"path": "file.txt", "metadata": {...}}]}' in content:
            print("   ✅ CLI help text updated")
            return True
        else:
            print("   ❌ CLI help text not updated")
            return False
    except Exception as e:
        print(f"   ❌ CLI help text test failed: {e}")
        return False

def test_database_class_structure():
    """Test database class has new methods without importing graph_memory"""
    print("🔍 Testing database class structure...")
    try:
        # Read the database file directly to check for methods
        db_file = Path(__file__).parent / "instant_db" / "core" / "database.py"
        with open(db_file, 'r') as f:
            content = f.read()
        
        required_methods = [
            'def transaction(',
            'def _transaction_context(',
            'def add_documents_batch(',
            'def _compare_chunks('
        ]
        
        for method in required_methods:
            if method not in content:
                print(f"   ❌ Missing method definition: {method}")
                return False
        
        # Check for the bug fix
        if 'file_hash is not None and' in content:
            print("   ✅ Database class structure and bug fix validated")
            return True
        else:
            print("   ❌ Bug fix not found")
            return False
    except Exception as e:
        print(f"   ❌ Database structure test failed: {e}")
        return False

def test_search_file_structure():
    """Test search file has FAISS improvements"""
    print("🔍 Testing search file structure...")
    try:
        search_file = Path(__file__).parent / "instant_db" / "core" / "search.py"
        with open(search_file, 'r') as f:
            content = f.read()
        
        required_elements = [
            'self.embeddings_store = {}',
            'def _save_embeddings(',
            'def _load_embeddings(',
            'import pickle'
        ]
        
        for element in required_elements:
            if element not in content:
                print(f"   ❌ Missing element: {element}")
                return False
        
        print("   ✅ Search file structure validated")
        return True
    except Exception as e:
        print(f"   ❌ Search file structure test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🚀 STARTING SIMPLE INSTANT-DB VALIDATION")
    print("=" * 60)
    
    tests = [
        test_faiss_class_structure,
        test_openai_class_structure,
        test_discovery_class_structure,
        test_metadata_filter_structure,
        test_cli_help_text,
        test_database_class_structure,
        test_search_file_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ❌ Test {test.__name__} crashed: {e}")
    
    print("=" * 60)
    print(f"📊 VALIDATION SUMMARY")
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    print(f"📈 Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All validation tests passed!")
        return 0
    else:
        print("⚠️ Some validation tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

