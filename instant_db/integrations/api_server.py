"""
REST API server for Instant-DB
Provides HTTP endpoints for document processing and search
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

try:
    from flask import Flask, request, jsonify, send_file
    from flask_cors import CORS
except ImportError:
    Flask = None
    CORS = None

from ..core.database import InstantDB
from ..processors.document import DocumentProcessor
from ..processors.batch import BatchProcessor


class APIServer:
    """
    REST API server for Instant-DB operations
    Provides endpoints for document processing, search, and database management
    """
    
    def __init__(self, 
                 db: Optional[InstantDB] = None,
                 host: str = "localhost",
                 port: int = 5000,
                 debug: bool = False):
        """
        Initialize API server
        
        Args:
            db: InstantDB instance (will create default if None)
            host: Server host
            port: Server port
            debug: Enable debug mode
        """
        if Flask is None:
            raise ImportError(
                "Flask is required for API server. "
                "Install with: pip install flask flask-cors"
            )
        
        self.db = db or InstantDB()
        self.host = host
        self.port = port
        self.debug = debug
        
        # Initialize Flask app
        self.app = Flask(__name__)
        if CORS:
            CORS(self.app)  # Enable CORS for all routes
        
        # Initialize processors
        self.document_processor = DocumentProcessor()
        self.batch_processor = BatchProcessor(self.document_processor)
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register all API routes"""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            })
        
        @self.app.route('/api/stats', methods=['GET'])
        def get_stats():
            """Get database statistics"""
            try:
                stats = self.db.get_stats()
                return jsonify({
                    "status": "success",
                    "stats": stats
                })
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/search', methods=['POST'])
        def search():
            """Search documents"""
            try:
                data = request.get_json() or {}
                
                query = data.get('query', '').strip()
                if not query:
                    return jsonify({
                        "status": "error",
                        "error": "Query parameter is required"
                    }), 400
                
                top_k = data.get('top_k', 5)
                filters = data.get('filters', {})
                
                # Validate top_k
                if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
                    top_k = 5
                
                results = self.db.search(
                    query=query,
                    top_k=top_k,
                    filters=filters
                )
                
                return jsonify({
                    "status": "success",
                    "query": query,
                    "results": results,
                    "count": len(results)
                })
                
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/documents', methods=['POST'])
        def add_documents():
            """Add documents to database"""
            try:
                data = request.get_json() or {}
                
                documents = data.get('documents', [])
                if not documents:
                    return jsonify({
                        "status": "error",
                        "error": "Documents array is required"
                    }), 400
                
                # Validate document structure
                for i, doc in enumerate(documents):
                    if not isinstance(doc, dict):
                        return jsonify({
                            "status": "error",
                            "error": f"Document {i} must be an object"
                        }), 400
                    
                    if 'content' not in doc:
                        return jsonify({
                            "status": "error",
                            "error": f"Document {i} missing 'content' field"
                        }), 400
                
                result = self.db.add_documents(documents)
                
                return jsonify({
                    "status": "success",
                    "result": result
                })
                
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/process/file', methods=['POST'])
        def process_file():
            """Process a single file upload"""
            try:
                # Check if file was uploaded
                if 'file' not in request.files:
                    return jsonify({
                        "status": "error",
                        "error": "No file uploaded"
                    }), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({
                        "status": "error",
                        "error": "No file selected"
                    }), 400
                
                # Get additional parameters
                output_dir = request.form.get('output_dir')
                metadata = {}
                
                # Parse metadata if provided
                metadata_json = request.form.get('metadata')
                if metadata_json:
                    try:
                        metadata = json.loads(metadata_json)
                    except json.JSONDecodeError:
                        return jsonify({
                            "status": "error",
                            "error": "Invalid metadata JSON"
                        }), 400
                
                # Save uploaded file temporarily
                temp_path = Path(f"/tmp/{file.filename}")  # Simple temp storage
                file.save(temp_path)
                
                try:
                    # Process the file
                    result = self.document_processor.process_document(
                        file_path=temp_path,
                        output_dir=output_dir,
                        metadata=metadata
                    )
                    
                    return jsonify(result)
                
                finally:
                    # Clean up temp file
                    if temp_path.exists():
                        temp_path.unlink()
                
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/process/directory', methods=['POST'])
        def process_directory():
            """Process all files in a directory"""
            try:
                data = request.get_json() or {}
                
                input_dir = data.get('input_dir')
                if not input_dir:
                    return jsonify({
                        "status": "error",
                        "error": "input_dir parameter is required"
                    }), 400
                
                output_dir = data.get('output_dir')
                file_extensions = data.get('file_extensions')
                recursive = data.get('recursive', True)
                exclude_patterns = data.get('exclude_patterns')
                
                result = self.batch_processor.process_directory(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    file_extensions=file_extensions,
                    recursive=recursive,
                    exclude_patterns=exclude_patterns
                )
                
                return jsonify(result)
                
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/export/custom-gpt', methods=['POST'])
        def export_custom_gpt():
            """Export database for Custom GPT"""
            try:
                from ..integrations.custom_gpt import CustomGPTExporter
                
                data = request.get_json() or {}
                
                output_path = data.get('output_path', './custom_gpt_export')
                format_type = data.get('format', 'markdown')
                max_file_size_mb = data.get('max_file_size_mb', 25)
                include_metadata = data.get('include_metadata', True)
                group_by_document = data.get('group_by_document', True)
                
                exporter = CustomGPTExporter(self.db)
                result = exporter.export_knowledge_file(
                    output_path=output_path,
                    format_type=format_type,
                    max_file_size_mb=max_file_size_mb,
                    include_metadata=include_metadata,
                    group_by_document=group_by_document
                )
                
                return jsonify(result)
                
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/config', methods=['GET'])
        def get_config():
            """Get server configuration"""
            return jsonify({
                "status": "success",
                "config": {
                    "embedding_provider": self.db.embedding_provider_name,
                    "embedding_model": self.db.embedding_model_name,
                    "vector_db": self.db.vector_db_name,
                    "supported_formats": self.document_processor.get_supported_formats()
                }
            })
        
        @self.app.route('/api/config', methods=['POST'])
        def update_config():
            """Update server configuration"""
            try:
                data = request.get_json() or {}
                
                # This would require reinitializing components
                # For now, return current config
                return jsonify({
                    "status": "success",
                    "message": "Configuration update not implemented yet",
                    "current_config": {
                        "embedding_provider": self.db.embedding_provider_name,
                        "embedding_model": self.db.embedding_model_name,
                        "vector_db": self.db.vector_db_name
                    }
                })
                
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "error": str(e)
                }), 500
        
        # Error handlers
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                "status": "error",
                "error": "Endpoint not found",
                "available_endpoints": [
                    "/api/health",
                    "/api/stats", 
                    "/api/search",
                    "/api/documents",
                    "/api/process/file",
                    "/api/process/directory",
                    "/api/export/custom-gpt",
                    "/api/config"
                ]
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({
                "status": "error",
                "error": "Internal server error"
            }), 500
    
    def run(self):
        """Start the API server"""
        print(f"🚀 Starting Instant-DB API Server on {self.host}:{self.port}")
        print(f"📖 API Documentation available at http://{self.host}:{self.port}/api/health")
        
        self.app.run(
            host=self.host,
            port=self.port,
            debug=self.debug,
            threaded=True
        )
    
    def get_app(self):
        """Get Flask app instance (for testing or custom deployment)"""
        return self.app


def create_app(db_path: str = "./instant_db_database",
               embedding_provider: str = "sentence-transformers",
               embedding_model: str = "all-MiniLM-L6-v2",
               vector_db: str = "chroma") -> Flask:
    """
    Factory function to create Flask app with InstantDB
    
    Args:
        db_path: Path to database
        embedding_provider: Embedding provider
        embedding_model: Embedding model
        vector_db: Vector database type
        
    Returns:
        Flask app instance
    """
    db = InstantDB(
        db_path=db_path,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        vector_db=vector_db
    )
    
    server = APIServer(db)
    return server.get_app()


# CLI integration
if __name__ == "__main__":
    import sys
    
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    
    server = APIServer(host=host, port=port)
    server.run() 