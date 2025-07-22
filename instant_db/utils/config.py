"""
Configuration management for Instant-DB
Handles loading and saving configuration settings
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    path: str = "./instant_db_database"
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_db: str = "chroma"
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class ProcessingConfig:
    """Document processing configuration"""
    batch_size: int = 10
    max_workers: int = 4
    skip_errors: bool = True
    supported_extensions: list = None
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = ['.txt', '.md', '.pdf', '.docx', '.html']


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "localhost"
    port: int = 5000
    debug: bool = False
    cors_enabled: bool = True
    max_file_size_mb: int = 100


@dataclass
class ExportConfig:
    """Export configuration"""
    default_format: str = "markdown"
    max_file_size_mb: int = 25
    include_metadata: bool = True
    group_by_document: bool = True


class Config:
    """
    Main configuration manager for Instant-DB
    Supports YAML, JSON, and environment variable configuration
    """
    
    DEFAULT_CONFIG_FILES = [
        ".instant_db.yaml",
        ".instant_db.yml", 
        ".instant_db.json",
        "instant_db_config.yaml",
        "instant_db_config.yml",
        "instant_db_config.json"
    ]
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Optional path to config file
        """
        self.config_path = Path(config_path) if config_path else None
        
        # Initialize with defaults
        self.database = DatabaseConfig()
        self.processing = ProcessingConfig()
        self.api = APIConfig()
        self.export = ExportConfig()
        
        # Load configuration
        self._load_config()
        self._load_environment_variables()
    
    def _load_config(self):
        """Load configuration from file"""
        config_file = self._find_config_file()
        
        if not config_file:
            return  # Use defaults
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    return
            
            self._update_from_dict(config_data)
            print(f"📝 Loaded configuration from {config_file}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load config file {config_file}: {e}")
    
    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file"""
        # Check specified path first
        if self.config_path and self.config_path.exists():
            return self.config_path
        
        # Check current directory and home directory
        search_paths = [
            Path.cwd(),
            Path.home()
        ]
        
        for base_path in search_paths:
            for config_file in self.DEFAULT_CONFIG_FILES:
                config_path = base_path / config_file
                if config_path.exists():
                    return config_path
        
        return None
    
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary"""
        # Database configuration
        if 'database' in config_data:
            db_config = config_data['database']
            for key, value in db_config.items():
                if hasattr(self.database, key):
                    setattr(self.database, key, value)
        
        # Processing configuration
        if 'processing' in config_data:
            proc_config = config_data['processing']
            for key, value in proc_config.items():
                if hasattr(self.processing, key):
                    setattr(self.processing, key, value)
        
        # API configuration
        if 'api' in config_data:
            api_config = config_data['api']
            for key, value in api_config.items():
                if hasattr(self.api, key):
                    setattr(self.api, key, value)
        
        # Export configuration
        if 'export' in config_data:
            export_config = config_data['export']
            for key, value in export_config.items():
                if hasattr(self.export, key):
                    setattr(self.export, key, value)
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        env_mappings = {
            # Database settings
            'INSTANT_DB_PATH': ('database', 'path'),
            'INSTANT_DB_EMBEDDING_PROVIDER': ('database', 'embedding_provider'),
            'INSTANT_DB_EMBEDDING_MODEL': ('database', 'embedding_model'),
            'INSTANT_DB_VECTOR_DB': ('database', 'vector_db'),
            'INSTANT_DB_CHUNK_SIZE': ('database', 'chunk_size'),
            'INSTANT_DB_CHUNK_OVERLAP': ('database', 'chunk_overlap'),
            
            # Processing settings
            'INSTANT_DB_BATCH_SIZE': ('processing', 'batch_size'),
            'INSTANT_DB_MAX_WORKERS': ('processing', 'max_workers'),
            'INSTANT_DB_SKIP_ERRORS': ('processing', 'skip_errors'),
            
            # API settings
            'INSTANT_DB_HOST': ('api', 'host'),
            'INSTANT_DB_PORT': ('api', 'port'),
            'INSTANT_DB_DEBUG': ('api', 'debug'),
            'INSTANT_DB_CORS_ENABLED': ('api', 'cors_enabled'),
            
            # OpenAI API key
            'OPENAI_API_KEY': ('database', '_openai_api_key')  # Special handling
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Type conversion
                if key in ['chunk_size', 'chunk_overlap', 'batch_size', 'max_workers', 'port']:
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                elif key in ['skip_errors', 'debug', 'cors_enabled']:
                    value = value.lower() in ['true', '1', 'yes', 'on']
                
                # Set the value
                if key == '_openai_api_key':
                    self.database._openai_api_key = value
                else:
                    config_obj = getattr(self, section)
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def save_config(self, config_path: Optional[Union[str, Path]] = None, 
                   format_type: str = "yaml") -> Path:
        """
        Save current configuration to file
        
        Args:
            config_path: Path to save config file
            format_type: Format to save ('yaml' or 'json')
            
        Returns:
            Path to saved config file
        """
        if config_path is None:
            config_path = f".instant_db_config.{format_type}"
        
        config_path = Path(config_path)
        
        # Convert to dictionary
        config_dict = {
            'database': asdict(self.database),
            'processing': asdict(self.processing), 
            'api': asdict(self.api),
            'export': asdict(self.export)
        }
        
        # Remove private keys
        for section in config_dict.values():
            keys_to_remove = [k for k in section.keys() if k.startswith('_')]
            for key in keys_to_remove:
                del section[key]
        
        # Save file
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if format_type.lower() == 'yaml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format_type.lower() == 'json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format_type}")
            
            print(f"💾 Configuration saved to {config_path}")
            return config_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to save config: {e}")
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration as dictionary"""
        config = asdict(self.database)
        # Add OpenAI API key if available
        if hasattr(self.database, '_openai_api_key'):
            config['openai_api_key'] = self.database._openai_api_key
        return config
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration as dictionary"""
        return asdict(self.processing)
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration as dictionary"""
        return asdict(self.api)
    
    def get_export_config(self) -> Dict[str, Any]:
        """Get export configuration as dictionary"""
        return asdict(self.export)
    
    def create_interactive_config(self) -> 'Config':
        """Create configuration through interactive prompts"""
        print("🔧 Interactive Instant-DB Configuration")
        print("=" * 50)
        
        # Database configuration
        print("\n📁 Database Configuration:")
        self.database.path = input(f"Database path [{self.database.path}]: ").strip() or self.database.path
        
        print("\nEmbedding Provider:")
        print("1. sentence-transformers (free, local)")
        print("2. openai (premium, API-based)")
        provider_choice = input(f"Choose provider (1-2) [{1 if self.database.embedding_provider == 'sentence-transformers' else 2}]: ").strip()
        
        if provider_choice == "2":
            self.database.embedding_provider = "openai"
            self.database.embedding_model = "text-embedding-3-small"
            
            api_key = input("OpenAI API Key: ").strip()
            if api_key:
                self.database._openai_api_key = api_key
        else:
            self.database.embedding_provider = "sentence-transformers"
            self.database.embedding_model = input(f"Model [{self.database.embedding_model}]: ").strip() or self.database.embedding_model
        
        print("\nVector Database:")
        print("1. chroma (recommended)")
        print("2. faiss (high performance)")
        print("3. sqlite (simple)")
        db_choice = input(f"Choose database (1-3) [1]: ").strip()
        
        db_map = {"2": "faiss", "3": "sqlite"}
        self.database.vector_db = db_map.get(db_choice, "chroma")
        
        # Processing configuration
        print("\n⚙️ Processing Configuration:")
        try:
            chunk_size = input(f"Chunk size [{self.processing.chunk_size}]: ").strip()
            if chunk_size:
                self.processing.chunk_size = int(chunk_size)
        except ValueError:
            pass
        
        try:
            max_workers = input(f"Max workers [{self.processing.max_workers}]: ").strip()
            if max_workers:
                self.processing.max_workers = int(max_workers)
        except ValueError:
            pass
        
        # API configuration
        print("\n🌐 API Server Configuration:")
        self.api.host = input(f"Host [{self.api.host}]: ").strip() or self.api.host
        
        try:
            port = input(f"Port [{self.api.port}]: ").strip()
            if port:
                self.api.port = int(port)
        except ValueError:
            pass
        
        # Save configuration
        save_config = input("\nSave configuration? (y/n) [y]: ").strip().lower()
        if save_config != 'n':
            format_choice = input("Format (yaml/json) [yaml]: ").strip().lower() or "yaml"
            config_path = self.save_config(format_type=format_choice)
            print(f"✅ Configuration saved to {config_path}")
        
        return self
    
    def __repr__(self):
        return f"Config(database={self.database}, processing={self.processing}, api={self.api}, export={self.export})" 