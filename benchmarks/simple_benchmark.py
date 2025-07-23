#!/usr/bin/env python3
"""
Simple benchmark script for Instant-DB search quality evaluation.

This script tests search accuracy using predefined queries and expected results.
It processes the demo dataset and evaluates how well the search finds relevant content.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instant_db import InstantDB


class SearchBenchmark:
    """Simple benchmark for evaluating search quality."""
    
    def __init__(self, db_path: str = "./benchmark_db"):
        self.db_path = db_path
        self.db = None
        self.results = []
        
    def setup(self, demo_dataset_path: str = "./demo_dataset"):
        """Initialize database with demo dataset."""
        print("🔧 Setting up benchmark database...")
        
        # Clean up any existing benchmark database
        if os.path.exists(self.db_path):
            import shutil
            shutil.rmtree(self.db_path)
            
        # Initialize database
        self.db = InstantDB(
            db_path=self.db_path,
            embedding_provider="sentence-transformers",
            embedding_model="all-MiniLM-L6-v2"
        )
        
        # Process demo files
        demo_path = Path(demo_dataset_path)
        if not demo_path.exists():
            raise FileNotFoundError(f"Demo dataset not found at {demo_dataset_path}")
            
        files = list(demo_path.glob("*.txt"))
        print(f"📄 Processing {len(files)} demo files...")
        
        for file_path in files:
            print(f"  - {file_path.name}")
            
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create metadata
            metadata = {
                "title": file_path.stem.replace('_', ' ').title(),
                "source": str(file_path),
                "type": "text",
                "file_name": file_path.name
            }
            
            # Add to database
            result = self.db.add_document(
                content=content,
                metadata=metadata,
                document_id=file_path.stem
            )
            if result.get('status') == 'error':
                print(f"    Error: {result.get('error')}")
            else:
                print(f"    Success: {result.get('chunks_processed', 0)} chunks")
            
        print("✅ Setup complete!")
        
    def get_test_queries(self) -> List[Dict]:
        """Define test queries with expected content."""
        return [
            {
                "query": "pricing objections",
                "expected_keywords": ["cost", "price", "expensive", "budget", "ROI"],
                "expected_file": "objection_handling.txt",
                "description": "Should find pricing objection handling content"
            },
            {
                "query": "security features",
                "expected_keywords": ["encryption", "security", "protection", "compliance"],
                "expected_file": "product_features.txt",
                "description": "Should find security-related product features"
            },
            {
                "query": "action items from Q4 planning",
                "expected_keywords": ["action", "Q4", "deadline", "responsible"],
                "expected_file": "meeting_notes.txt",
                "description": "Should find Q4 planning action items"
            },
            {
                "query": "CloudSync Pro benefits",
                "expected_keywords": ["CloudSync", "benefit", "advantage", "solution"],
                "expected_file": "sample_pitch_deck.txt",
                "description": "Should find CloudSync Pro benefits"
            },
            {
                "query": "data migration concerns",
                "expected_keywords": ["migration", "data", "transfer", "existing"],
                "expected_file": "objection_handling.txt",
                "description": "Should find data migration objection handling"
            }
        ]
        
    def evaluate_query(self, test_case: Dict) -> Dict:
        """Evaluate a single query and return results."""
        query = test_case["query"]
        print(f"\n🔍 Testing: {query}")
        print(f"   Description: {test_case['description']}")
        
        # Time the search
        start_time = time.time()
        results = self.db.search(query, top_k=5)
        search_time = time.time() - start_time
        
        # Debug: print first result if any
        if results:
            print(f"   Found {len(results)} results")
        else:
            print(f"   WARNING: No results found for query: {query}")
        
        # Evaluate results
        evaluation = {
            "query": query,
            "search_time": search_time,
            "num_results": len(results),
            "top_result_score": results[0]["similarity"] if results else 0,
            "found_expected_file": False,
            "keyword_matches": 0,
            "relevant_results": 0,
            "top_result_preview": "No results found"
        }
        
        # Check each result
        for i, result in enumerate(results):
            content_lower = result["content"].lower()
            
            # Check if from expected file (check both document_id and metadata)
            doc_id = result.get("document_id", "")
            if not doc_id and "metadata" in result:
                doc_id = result["metadata"].get("document_id", "")
            
            if test_case["expected_file"].replace(".txt", "") in doc_id:
                evaluation["found_expected_file"] = True
                
            # Count keyword matches
            keyword_matches = sum(1 for kw in test_case["expected_keywords"] 
                                if kw.lower() in content_lower)
            
            # Consider result relevant if it has 2+ keyword matches
            if keyword_matches >= 2:
                evaluation["relevant_results"] += 1
                
            if i == 0:  # Top result
                evaluation["keyword_matches"] = keyword_matches
                evaluation["top_result_preview"] = result["content"][:150] + "..."
                
        # Calculate precision
        evaluation["precision"] = evaluation["relevant_results"] / len(results) if results else 0
        
        return evaluation
        
    def run_benchmark(self):
        """Run all benchmark tests."""
        print("\n🚀 Running search quality benchmark...")
        print("=" * 60)
        
        test_queries = self.get_test_queries()
        total_score = 0
        
        for test_case in test_queries:
            result = self.evaluate_query(test_case)
            self.results.append(result)
            
            # Calculate score for this query (0-100)
            score = 0
            score += 40 if result["found_expected_file"] else 0
            score += min(30, result["keyword_matches"] * 10)
            score += min(20, result["precision"] * 20)
            score += 10 if result["top_result_score"] > 0.7 else 5 if result["top_result_score"] > 0.5 else 0
            
            result["score"] = score
            total_score += score
            
            # Print results
            print(f"\n✅ Score: {score}/100")
            print(f"   - Found expected file: {'Yes' if result['found_expected_file'] else 'No'}")
            print(f"   - Keyword matches: {result['keyword_matches']}")
            print(f"   - Precision: {result['precision']:.2f}")
            print(f"   - Top result score: {result['top_result_score']:.3f}")
            print(f"   - Search time: {result['search_time']:.3f}s")
            print(f"   - Preview: {result['top_result_preview']}")
            
        # Overall summary
        avg_score = total_score / len(test_queries)
        avg_search_time = sum(r["search_time"] for r in self.results) / len(self.results)
        
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Overall Score: {avg_score:.1f}/100")
        print(f"Average Search Time: {avg_search_time:.3f}s")
        print(f"Total Queries: {len(test_queries)}")
        
        # Performance rating
        if avg_score >= 90:
            rating = "🌟 EXCELLENT"
        elif avg_score >= 75:
            rating = "✅ GOOD"
        elif avg_score >= 60:
            rating = "⚠️  FAIR"
        else:
            rating = "❌ NEEDS IMPROVEMENT"
            
        print(f"Performance Rating: {rating}")
        
        # Save results
        self.save_results()
        
    def save_results(self):
        """Save benchmark results to JSON file."""
        output_file = Path(self.db_path).parent / "benchmark_results.json"
        
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "db_path": self.db_path,
            "embedding_provider": "sentence-transformers",
            "embedding_model": "all-MiniLM-L6-v2",
            "results": self.results,
            "summary": {
                "total_queries": len(self.results),
                "average_score": sum(r["score"] for r in self.results) / len(self.results),
                "average_search_time": sum(r["search_time"] for r in self.results) / len(self.results),
                "average_precision": sum(r["precision"] for r in self.results) / len(self.results)
            }
        }
        
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n💾 Results saved to: {output_file}")
        
    def cleanup(self):
        """Clean up benchmark database."""
        if os.path.exists(self.db_path):
            import shutil
            shutil.rmtree(self.db_path)
            print("🧹 Cleaned up benchmark database")


def main():
    """Run the benchmark."""
    benchmark = SearchBenchmark()
    
    try:
        # Setup with demo dataset
        benchmark.setup("./demo_dataset")
        
        # Run benchmark
        benchmark.run_benchmark()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
        
    finally:
        # Cleanup
        benchmark.cleanup()


if __name__ == "__main__":
    main()