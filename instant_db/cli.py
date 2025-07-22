"""
Command Line Interface for Instant-DB
Enhanced CLI using Click framework with better UX
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

import click
from tqdm import tqdm

from .core.database import InstantDB
from .processors.document import DocumentProcessor
from .processors.batch import BatchProcessor
from .integrations.custom_gpt import CustomGPTExporter
from .integrations.api_server import APIServer
from .utils.config import Config
from .utils.logging import setup_logging, get_logger

# Version information
__version__ = "1.0.0"

# Global configuration
config = Config()


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-error output')
@click.option('--config', '-c', type=click.Path(), help='Path to config file')
@click.option('--db-path', help='Database path')
@click.option('--embedding-provider', type=click.Choice(['sentence-transformers', 'openai']), 
              help='Embedding provider')
@click.option('--vector-db', type=click.Choice(['chroma', 'faiss', 'sqlite']), 
              help='Vector database type')
@click.version_option(version=__version__, prog_name='Instant-DB')
@click.pass_context
def cli(ctx, verbose, quiet, config_path, db_path, embedding_provider, vector_db):
    """🚀 Instant-DB: Transform documents into searchable knowledge bases in minutes"""
    
    # Ensure we have a context object
    ctx.ensure_object(dict)
    
    # Setup logging
    if quiet:
        log_level = 'ERROR'
    elif verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'
    
    setup_logging(level=log_level, console_output=not quiet)
    logger = get_logger()
    
    # Load configuration
    global config
    if config_path:
        config = Config(config_path)
    
    # Override config with CLI options
    if db_path:
        config.database.path = db_path
    if embedding_provider:
        config.database.embedding_provider = embedding_provider
    if vector_db:
        config.database.vector_db = vector_db
    
    # Store in context
    ctx.obj['config'] = config
    ctx.obj['logger'] = logger


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-o', '--output', help='Output directory for database')
@click.option('--batch', is_flag=True, help='Batch process directory')
@click.option('--extensions', multiple=True, 
              help='File extensions to process (e.g., --extensions .pdf --extensions .txt)')
@click.option('--recursive/--no-recursive', default=True, 
              help='Search subdirectories recursively')
@click.option('--exclude', multiple=True, 
              help='Patterns to exclude from processing')
@click.option('--workers', type=int, 
              help='Number of worker threads for parallel processing')
@click.pass_context
def process(ctx, input_path, output, batch, extensions, recursive, exclude, workers):
    """Process documents and add them to the database"""
    
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    input_path = Path(input_path)
    
    # Initialize processor with config
    processor = DocumentProcessor(
        embedding_provider=config.database.embedding_provider,
        embedding_model=config.database.embedding_model,
        vector_db=config.database.vector_db,
        chunk_size=config.database.chunk_size,
        chunk_overlap=config.database.chunk_overlap
    )
    
    output_dir = output or config.database.path
    
    if batch or input_path.is_dir():
        # Batch processing
        click.echo(f"🚀 Processing directory: {input_path}")
        
        batch_processor = BatchProcessor(
            document_processor=processor,
            max_workers=workers or config.processing.max_workers,
            skip_errors=config.processing.skip_errors
        )
        
        # Convert extensions to list
        file_extensions = list(extensions) if extensions else None
        exclude_patterns = list(exclude) if exclude else None
        
        with click.progressbar(length=0, label='Processing files') as bar:
            def update_progress():
                progress = batch_processor.get_progress()
                bar.length = progress['total_files']
                bar.current = progress['processed'] + progress['errors']
                bar.render_progress()
            
            # Start processing
            result = batch_processor.process_directory(
                input_dir=input_path,
                output_dir=output_dir,
                file_extensions=file_extensions,
                recursive=recursive,
                exclude_patterns=exclude_patterns
            )
        
        # Display results
        if result['status'] == 'completed':
            click.echo(f"\n✅ Processing completed!")
            click.echo(f"   📁 Files processed: {result['files_processed']}")
            click.echo(f"   ❌ Files with errors: {result['files_with_errors']}")
            click.echo(f"   📊 Total chunks: {result['total_chunks']}")
            click.echo(f"   ⏱️  Time: {result['processing_time_seconds']:.1f}s")
            click.echo(f"   💾 Database: {output_dir}")
            
            if result['files_with_errors'] > 0:
                click.echo(f"\n⚠️  {result['files_with_errors']} files had errors. Use --verbose for details.")
        else:
            click.echo(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    else:
        # Single file processing
        click.echo(f"🚀 Processing file: {input_path}")
        
        with click.progressbar(length=1, label='Processing') as bar:
            result = processor.process_document(
                file_path=input_path,
                output_dir=output_dir
            )
            bar.update(1)
        
        if result['status'] == 'success':
            click.echo(f"\n✅ Processing completed!")
            click.echo(f"   📊 Chunks created: {result['chunks_processed']}")
            click.echo(f"   📝 Total words: {result['total_words']:,}")
            click.echo(f"   💾 Database: {result['database_path']}")
        else:
            click.echo(f"❌ Processing failed: {result['error']}")
            sys.exit(1)


@cli.command()
@click.argument('query')
@click.option('-k', '--top-k', type=int, default=5, help='Number of results to return')
@click.option('--interactive', '-i', is_flag=True, help='Start interactive search mode')
@click.option('--format', 'output_format', type=click.Choice(['text', 'json']), default='text',
              help='Output format')
@click.option('--document-type', help='Filter by document type')
@click.option('--include-metadata', is_flag=True, help='Include document metadata in results')
@click.pass_context  
def search(ctx, query, top_k, interactive, output_format, document_type, include_metadata):
    """Search the document database"""
    
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    # Initialize database
    db = InstantDB(
        db_path=config.database.path,
        embedding_provider=config.database.embedding_provider,
        embedding_model=config.database.embedding_model,
        vector_db=config.database.vector_db
    )
    
    def perform_search(search_query: str) -> None:
        """Perform a single search"""
        filters = {}
        if document_type:
            filters['document_type'] = document_type
        
        with click.progressbar(length=1, label='Searching') as bar:
            results = db.search(
                query=search_query,
                top_k=top_k,
                filters=filters
            )
            bar.update(1)
        
        if not results:
            click.echo("❌ No results found.")
            return
        
        if output_format == 'json':
            click.echo(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            click.echo(f"\n🔍 Found {len(results)} results for: '{search_query}'\n")
            
            for i, result in enumerate(results, 1):
                click.echo(f"--- Result {i} ---")
                click.echo(f"📄 Document: {result.get('document_id', 'Unknown')}")
                
                if result.get('section'):
                    click.echo(f"📂 Section: {result['section']}")
                
                similarity = result.get('similarity', 0)
                click.echo(f"📊 Relevance: {similarity:.1%}")
                
                content = result.get('content', '')
                if len(content) > 300:
                    content = content[:300] + "..."
                click.echo(f"📝 Content: {content}")
                
                if include_metadata and result.get('metadata'):
                    click.echo(f"🏷️  Metadata: {json.dumps(result['metadata'], indent=2)}")
                
                click.echo()
    
    if interactive:
        click.echo("🔍 Interactive Search Mode (type 'quit' to exit)")
        while True:
            try:
                search_query = click.prompt("\nEnter search query", type=str)
                if search_query.lower() in ['quit', 'exit', 'q']:
                    break
                perform_search(search_query)
            except (KeyboardInterrupt, EOFError):
                click.echo("\n👋 Goodbye!")
                break
    else:
        if not query:
            click.echo("❌ Query is required for non-interactive search")
            sys.exit(1)
        perform_search(query)


@cli.command()
@click.option('--format', 'export_format', type=click.Choice(['markdown', 'json', 'txt']), 
              default='markdown', help='Export format')
@click.option('-o', '--output', help='Output directory')
@click.option('--max-size', type=int, default=25, help='Maximum file size in MB')
@click.option('--split-by-type', is_flag=True, help='Split export by document type')
@click.option('--include-metadata/--no-metadata', default=True, help='Include metadata')
@click.option('--create-index', is_flag=True, default=True, help='Create index file')
@click.pass_context
def export(ctx, export_format, output, max_size, split_by_type, include_metadata, create_index):
    """Export database for Custom GPT or other AI assistants"""
    
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    # Initialize database
    db = InstantDB(
        db_path=config.database.path,
        embedding_provider=config.database.embedding_provider,
        embedding_model=config.database.embedding_model,
        vector_db=config.database.vector_db
    )
    
    # Initialize exporter
    exporter = CustomGPTExporter(db)
    
    output_dir = output or f"./instant_db_export_{export_format}"
    
    click.echo(f"📤 Exporting database to {export_format} format...")
    
    with click.progressbar(length=1, label='Exporting') as bar:
        if split_by_type:
            result = exporter.export_structured_knowledge(
                output_dir=output_dir,
                create_index=create_index,
                split_by_type=True
            )
        else:
            result = exporter.export_knowledge_file(
                output_path=Path(output_dir) / f"knowledge_base.{export_format}",
                format_type=export_format,
                max_file_size_mb=max_size,
                include_metadata=include_metadata
            )
        bar.update(1)
    
    if result['status'] == 'success':
        click.echo(f"\n✅ Export completed!")
        
        if 'files_created' in result:
            click.echo(f"   📁 Files created: {len(result['files_created'])}")
            click.echo(f"   📊 Total documents: {result.get('total_documents', 'N/A')}")
            click.echo(f"   📝 Total chunks: {result.get('total_chunks', 'N/A')}")
            click.echo(f"   💾 Output directory: {result.get('output_directory', output_dir)}")
            
            # Show first few files
            for file_info in result['files_created'][:5]:
                size_mb = file_info.get('file_size', 0) / (1024 * 1024)
                click.echo(f"   📄 {file_info['filename']} ({size_mb:.1f} MB)")
        
        # Generate instructions for Custom GPT
        if export_format == 'markdown' and not split_by_type:
            instructions = exporter.create_gpt_instructions(
                knowledge_files=[f"knowledge_base.{export_format}"],
                assistant_name="Knowledge Assistant"
            )
            
            instructions_file = Path(output_dir).parent / "custom_gpt_instructions.txt"
            with open(instructions_file, 'w') as f:
                f.write(instructions)
            
            click.echo(f"   📋 GPT Instructions: {instructions_file}")
    
    else:
        click.echo(f"❌ Export failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


@cli.command()
@click.option('--host', default='localhost', help='Server host')
@click.option('--port', type=int, default=5000, help='Server port')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.pass_context
def serve(ctx, host, port, debug):
    """Start the API server"""
    
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    # Override with CLI options
    if host != 'localhost':
        config.api.host = host
    if port != 5000:
        config.api.port = port
    if debug:
        config.api.debug = debug
    
    # Initialize database
    db = InstantDB(
        db_path=config.database.path,
        embedding_provider=config.database.embedding_provider,
        embedding_model=config.database.embedding_model,
        vector_db=config.database.vector_db
    )
    
    # Start server
    server = APIServer(
        db=db,
        host=config.api.host,
        port=config.api.port,
        debug=config.api.debug
    )
    
    click.echo(f"🚀 Starting Instant-DB API server...")
    click.echo(f"📍 URL: http://{config.api.host}:{config.api.port}")
    click.echo(f"📖 Health check: http://{config.api.host}:{config.api.port}/api/health")
    click.echo(f"🛑 Press Ctrl+C to stop")
    
    try:
        server.run()
    except KeyboardInterrupt:
        click.echo("\n👋 Server stopped.")


@cli.command()
@click.pass_context
def stats(ctx):
    """Show database statistics"""
    
    config = ctx.obj['config']
    
    # Initialize database
    db = InstantDB(
        db_path=config.database.path,
        embedding_provider=config.database.embedding_provider,
        embedding_model=config.database.embedding_model,
        vector_db=config.database.vector_db
    )
    
    try:
        stats = db.get_stats()
        
        click.echo("📊 Database Statistics")
        click.echo("=" * 40)
        click.echo(f"📍 Database Path: {config.database.path}")
        click.echo(f"🔧 Vector DB: {stats['vector_db']}")
        click.echo(f"🤖 Embedding Provider: {stats['embedding_provider']}")
        click.echo(f"📦 Embedding Model: {stats['embedding_model']}")
        click.echo(f"📄 Total Documents: {stats['document_count']:,}")
        click.echo(f"🔢 Embedding Dimension: {stats['embedding_dimension']}")
        
        # Cache stats
        cache_stats = stats.get('cache_stats', {})
        if cache_stats:
            click.echo(f"\n💾 Cache Statistics")
            click.echo(f"   📦 Cache Size: {cache_stats.get('cache_size', 0)}")
            click.echo(f"   ✅ Cache Hits: {cache_stats.get('cache_hits', 0)}")
            click.echo(f"   ❌ Cache Misses: {cache_stats.get('cache_misses', 0)}")
            if cache_stats.get('total_requests', 0) > 0:
                hit_rate = cache_stats.get('hit_rate', 0) * 100
                click.echo(f"   📈 Hit Rate: {hit_rate:.1f}%")
    
    except Exception as e:
        click.echo(f"❌ Error getting statistics: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def config(ctx):
    """Create or update configuration interactively"""
    
    click.echo("🔧 Instant-DB Configuration Wizard")
    click.echo("=" * 50)
    
    # Create interactive config
    new_config = Config()
    new_config.create_interactive_config()
    
    # Update global config
    ctx.obj['config'] = new_config


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize a new Instant-DB project"""
    
    click.echo("🚀 Initialize Instant-DB Project")
    click.echo("=" * 40)
    
    # Check if already initialized
    if Path('.instant_db.yaml').exists():
        if not click.confirm("Project already initialized. Reinitialize?"):
            return
    
    # Create configuration
    config = Config()
    config.create_interactive_config()
    
    # Create sample directories
    sample_dirs = ['documents', 'exports']
    for dir_name in sample_dirs:
        path = Path(dir_name)
        if not path.exists():
            path.mkdir(exist_ok=True)
            click.echo(f"📁 Created directory: {dir_name}/")
    
    # Create sample .gitignore
    gitignore_content = """# Instant-DB
instant_db_database/
*.db
*.index
*.log

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    
    if not Path('.gitignore').exists():
        Path('.gitignore').write_text(gitignore_content)
        click.echo("📄 Created .gitignore")
    
    click.echo("\n✅ Project initialized successfully!")
    click.echo("🎯 Next steps:")
    click.echo("   1. Add documents to the 'documents/' directory")
    click.echo("   2. Run: instant-db process documents/ --batch")
    click.echo("   3. Search: instant-db search 'your query'")


def main():
    """Main entry point for the CLI"""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n👋 Operation cancelled.")
        sys.exit(130)
    except Exception as e:
        logger = get_logger()
        logger.exception(f"Unexpected error: {e}")
        click.echo(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 