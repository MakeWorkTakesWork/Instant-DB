#!/usr/bin/env python3
"""
Simple validation test for Instant-DB enhancements
Tests the core functionality without heavy dependencies
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add the instant_db module to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    try:
        from instant_db.core.database import InstantDB
        from instant_db.core.search import FAISSVectorStore, BaseVectorStore
        from instant_db.core.embeddings import OpenAIProvider, BaseEmbeddingProvider
        from instant_db.core.discovery import DocumentDiscovery
        from instant_db.core.metadata_filter import MetadataFilterEngine
        print("   ✅ All core modules imported successfully")
        return True
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_faiss_embeddings_storage():
    """Test FAISS embeddings storage functionality"""
    print("🔍 Testing FAISS embeddings storage...")
    try:
        from instant_db.core.search import FAISSVectorStore
        import numpy as np
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create FAISS store
            store = FAISSVectorStore(temp_path, dimension=384)
            
            # Check if embeddings storage is initialized
            assert hasattr(store, 'embeddings_store'), "embeddings_store not initialized"
            assert hasattr(store, '_save_embeddings'), "_save_embeddings method missing"
            assert hasattr(store, '_load_embeddings'), "_load_embeddings method missing"
            
            print("   ✅ FAISS embeddings storage structure validated")
            return True
    except Exception as e:
        print(f"   ❌ FAISS test failed: {e}")
        return False

def test_database_transactions():
    """Test database transaction functionality"""
    print("🔍 Testing database transactions...")
    try:
        from instant_db.core.database import InstantDB
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize database
            db = InstantDB(db_path=temp_dir, vector_db="sqlite")
            
            # Check transaction methods exist
            assert hasattr(db, 'transaction'), "transaction method missing"
            assert hasattr(db, '_transaction_context'), "_transaction_context method missing"
            assert hasattr(db, 'add_documents_batch'), "add_documents_batch method missing"
            assert hasattr(db, '_compare_chunks'), "_compare_chunks method missing"
            
            print("   ✅ Database transaction methods validated")
            return True
    except Exception as e:
        print(f"   ❌ Database transaction test failed: {e}")
        return False

def test_parallel_discovery():
    """Test parallel document discovery functionality"""
    print("🔍 Testing parallel document discovery...")
    try:
        from instant_db.core.discovery import DocumentDiscovery
        
        discovery = DocumentDiscovery()
        
        # Check parallel methods exist
        assert hasattr(discovery, 'scan_directory_for_documents_parallel'), "parallel scan method missing"
        assert hasattr(discovery, '_scan_directory_recursive'), "recursive scan method missing"
        assert hasattr(discovery, 'discover_documents'), "discover_documents method missing"
        assert hasattr(discovery, '_is_supported_file'), "_is_supported_file method missing"
        assert hasattr(discovery, '_passes_filters'), "_passes_filters method missing"
        
        print("   ✅ Parallel discovery methods validated")
        return True
    except Exception as e:
        print(f"   ❌ Parallel discovery test failed: {e}")
        return False

def test_openai_batch_processing():
    """Test OpenAI batch processing functionality"""
    print("🔍 Testing OpenAI batch processing...")
    try:
        from instant_db.core.embeddings import OpenAIProvider
        
        # Check if enhanced methods exist (without actually calling OpenAI)
        provider_class = OpenAIProvider
        
        # Check if the class has the new methods
        assert hasattr(provider_class, '_get_cache_key'), "_get_cache_key method missing"
        assert hasattr(provider_class, '_load_cache'), "_load_cache method missing"
        assert hasattr(provider_class, '_save_cache'), "_save_cache method missing"
        assert hasattr(provider_class, '_encode_single'), "_encode_single method missing"
        
        print("   ✅ OpenAI batch processing methods validated")
        return True
    except Exception as e:
        print(f"   ❌ OpenAI batch processing test failed: {e}")
        return False

def test_bug_fixes():
    """Test that bug fixes are in place"""
    print("🔍 Testing bug fixes...")
    try:
        from instant_db.core.database import InstantDB
        from instant_db.core.metadata_filter import MetadataFilterEngine
        
        # Test _document_exists fix (should handle None file_hash)
        with tempfile.TemporaryDirectory() as temp_dir:
            db = InstantDB(db_path=temp_dir, vector_db="sqlite")
            
            # This should not crash with None file_hash
            result = db._document_exists("test_doc", None)
            assert result == False, "_document_exists should return False for None file_hash"
        
        # Test metadata filter _in operator
        filter_engine = MetadataFilterEngine()
        assert hasattr(filter_engine, '_in'), "_in method missing"
        
        print("   ✅ Bug fixes validated")
        return True
    except Exception as e:
        print(f"   ❌ Bug fixes test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🚀 STARTING INSTANT-DB VALIDATION TESTS")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_faiss_embeddings_storage,
        test_database_transactions,
        test_parallel_discovery,
        test_openai_batch_processing,
        test_bug_fixes
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

