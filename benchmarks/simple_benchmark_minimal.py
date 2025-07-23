#!/usr/bin/env python3
"""
Minimal benchmark script for Instant-DB search quality evaluation.

This simplified version tests search functionality with predefined test data.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instant_db import InstantDB


def run_minimal_benchmark():
    """Run a minimal benchmark test."""
    
    print("🚀 Instant-DB Simple Benchmark")
    print("=" * 60)
    
    # Test data
    test_documents = [
        {
            "id": "pricing_doc",
            "content": """CloudSync Pro Pricing Objection Handling Guide
            
            Common Price Objections and Responses:
            
            1. "It's too expensive for our budget"
            Response: Let's look at the ROI. CloudSync Pro typically saves companies 20-30 hours per month 
            in manual data synchronization. At an average hourly rate of $50, that's $1000-1500 in savings 
            monthly, while our Pro plan is only $299/month.
            
            2. "We can't justify the cost to management"
            Response: I can provide you with a detailed cost-benefit analysis template that shows the breakeven 
            point is typically reached within 2 months. We also offer a 30-day money-back guarantee.
            
            3. "Your competitor is cheaper"
            Response: While some alternatives may have a lower sticker price, CloudSync Pro includes enterprise 
            security features, 24/7 support, and unlimited integrations that would cost extra with other solutions.
            """,
            "metadata": {
                "title": "Pricing Objection Handling",
                "type": "guide",
                "category": "sales"
            }
        },
        {
            "id": "security_doc", 
            "content": """CloudSync Pro Security Features
            
            Enterprise-Grade Security:
            - End-to-end encryption using AES-256
            - SOC 2 Type II compliance
            - GDPR and CCPA compliant data handling
            - Multi-factor authentication (MFA)
            - Role-based access control (RBAC)
            - Audit logs with tamper protection
            - Data residency options (US, EU, APAC)
            - Zero-knowledge architecture for sensitive data
            - Regular third-party security audits
            - 99.99% uptime SLA with redundancy
            """,
            "metadata": {
                "title": "Security Features",
                "type": "documentation", 
                "category": "product"
            }
        },
        {
            "id": "meeting_doc",
            "content": """Q4 Planning Meeting Notes - CloudSync Pro
            
            Date: October 15, 2024
            Attendees: Product, Sales, Engineering
            
            Action Items:
            1. Launch enterprise tier by Nov 15 (Owner: Product Team)
            2. Complete SOC 2 audit by Dec 1 (Owner: Security Team)
            3. Hire 5 new sales reps for Q4 push (Owner: Sales Director)
            4. Implement Salesforce integration by Nov 30 (Owner: Engineering)
            5. Create new pricing calculator tool (Owner: Marketing)
            
            Q4 Revenue Target: $2.5M ARR
            Current Pipeline: $1.8M
            Gap to close: $700K
            """,
            "metadata": {
                "title": "Q4 Planning Meeting",
                "type": "meeting_notes",
                "category": "planning"
            }
        }
    ]
    
    # Initialize database
    db_path = "./benchmark_test_db"
    
    # Clean up any existing database
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
    
    print("📦 Initializing database...")
    db = InstantDB(
        db_path=db_path,
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    # Add documents
    print("📄 Adding test documents...")
    for doc in test_documents:
        result = db.add_document(
            content=doc["content"],
            metadata=doc["metadata"],
            document_id=doc["id"]
        )
        print(f"  - {doc['id']}: {result.get('status', 'unknown')}")
        if result.get('status') == 'error':
            print(f"    Error: {result.get('error', 'unknown error')}")
    
    # Test queries
    test_queries = [
        {
            "query": "pricing objections expensive budget",
            "expected": "pricing_doc",
            "description": "Find pricing objection handling"
        },
        {
            "query": "security encryption compliance",
            "expected": "security_doc", 
            "description": "Find security features"
        },
        {
            "query": "Q4 action items deadline",
            "expected": "meeting_doc",
            "description": "Find Q4 planning items"
        }
    ]
    
    print("\n🔍 Running search tests...")
    print("=" * 60)
    
    results_summary = []
    
    for test in test_queries:
        print(f"\nQuery: '{test['query']}'")
        print(f"Expected: {test['expected']}")
        
        # Perform search
        start_time = time.time()
        results = db.search(test["query"], top_k=3)
        search_time = time.time() - start_time
        
        # Check results
        found = False
        if results:
            for i, result in enumerate(results):
                doc_id = result.get("document_id", "unknown")
                score = result.get("similarity", 0)
                print(f"  Result {i+1}: {doc_id} (score: {score:.3f})")
                if i == 0:  # Debug first result
                    metadata = result.get("metadata", {})
                    print(f"    Metadata: {metadata}")
                
                if doc_id == test["expected"]:
                    found = True
        else:
            print("  No results found!")
        
        success = "✅ PASS" if found else "❌ FAIL"
        print(f"  Status: {success} (search time: {search_time:.3f}s)")
        
        results_summary.append({
            "query": test["query"],
            "expected": test["expected"],
            "found": found,
            "search_time": search_time
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results_summary if r["found"])
    total = len(results_summary)
    avg_time = sum(r["search_time"] for r in results_summary) / total
    
    print(f"Tests Passed: {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"Average Search Time: {avg_time:.3f}s")
    
    # Cleanup
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
        print("\n🧹 Cleaned up test database")
    
    return passed == total


if __name__ == "__main__":
    success = run_minimal_benchmark()
    sys.exit(0 if success else 1)