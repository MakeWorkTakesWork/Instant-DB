"""
Custom GPT exporter for Instant-DB
Export database contents in formats suitable for Custom GPT upload
"""

import json
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from ..core.database import InstantDB


class CustomGPTExporter:
    """
    Export Instant-DB contents for Custom GPT integration
    Supports various export formats optimized for AI assistants
    """
    
    def __init__(self, db: InstantDB):
        """
        Initialize exporter with InstantDB instance
        
        Args:
            db: InstantDB instance to export from
        """
        self.db = db
    
    def export_knowledge_file(self, 
                             output_path: Union[str, Path],
                             format_type: str = "markdown",
                             max_file_size_mb: int = 25,
                             include_metadata: bool = True,
                             group_by_document: bool = True) -> Dict[str, Any]:
        """
        Export database as knowledge files for Custom GPT
        
        Args:
            output_path: Output file or directory path
            format_type: Export format ('markdown', 'txt', 'json')
            max_file_size_mb: Maximum file size in MB (Custom GPT limit)
            include_metadata: Whether to include document metadata
            group_by_document: Whether to group chunks by document
            
        Returns:
            Export results dictionary
        """
        output_path = Path(output_path)
        
        # Get all documents from database
        all_docs = self._get_all_documents()
        
        if not all_docs:
            return {
                "status": "error",
                "error": "No documents found in database",
                "document_count": 0
            }
        
        # Organize documents
        if group_by_document:
            organized_docs = self._group_by_document(all_docs)
        else:
            organized_docs = {"all_content": all_docs}
        
        # Export based on format
        if format_type == "markdown":
            result = self._export_markdown(organized_docs, output_path, max_file_size_mb, include_metadata)
        elif format_type == "txt":
            result = self._export_text(organized_docs, output_path, max_file_size_mb, include_metadata)
        elif format_type == "json":
            result = self._export_json(organized_docs, output_path, max_file_size_mb, include_metadata)
        else:
            return {
                "status": "error",
                "error": f"Unsupported format: {format_type}",
                "supported_formats": ["markdown", "txt", "json"]
            }
        
        return result
    
    def export_structured_knowledge(self,
                                  output_dir: Union[str, Path],
                                  create_index: bool = True,
                                  split_by_type: bool = True) -> Dict[str, Any]:
        """
        Export database as structured knowledge files
        Creates multiple files organized by document type and sections
        
        Args:
            output_dir: Output directory path
            create_index: Whether to create an index file
            split_by_type: Whether to split by document type
            
        Returns:
            Export results dictionary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Get all documents
        all_docs = self._get_all_documents()
        
        if not all_docs:
            return {
                "status": "error",
                "error": "No documents found in database"
            }
        
        files_created = []
        
        # Organize by document type if requested
        if split_by_type:
            by_type = {}
            for doc in all_docs:
                doc_type = doc.get("document_type", "Unknown")
                if doc_type not in by_type:
                    by_type[doc_type] = []
                by_type[doc_type].append(doc)
            
            # Export each type
            for doc_type, docs in by_type.items():
                filename = f"{doc_type.lower().replace(' ', '_')}_knowledge.md"
                file_path = output_dir / filename
                
                content = self._create_markdown_content(docs, include_metadata=True)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                files_created.append({
                    "filename": filename,
                    "path": str(file_path),
                    "document_type": doc_type,
                    "document_count": len(set(doc["document_id"] for doc in docs)),
                    "chunk_count": len(docs),
                    "file_size": file_path.stat().st_size
                })
        else:
            # Single file export
            content = self._create_markdown_content(all_docs, include_metadata=True)
            file_path = output_dir / "knowledge_base.md"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            files_created.append({
                "filename": "knowledge_base.md",
                "path": str(file_path),
                "document_count": len(set(doc["document_id"] for doc in all_docs)),
                "chunk_count": len(all_docs),
                "file_size": file_path.stat().st_size
            })
        
        # Create index file if requested
        if create_index:
            index_content = self._create_index_content(files_created, all_docs)
            index_path = output_dir / "README.md"
            
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(index_content)
            
            files_created.insert(0, {
                "filename": "README.md",
                "path": str(index_path),
                "is_index": True,
                "file_size": index_path.stat().st_size
            })
        
        return {
            "status": "success",
            "output_directory": str(output_dir),
            "files_created": files_created,
            "total_documents": len(set(doc["document_id"] for doc in all_docs)),
            "total_chunks": len(all_docs),
            "export_timestamp": datetime.now().isoformat()
        }
    
    def create_gpt_instructions(self,
                              knowledge_files: List[str],
                              assistant_name: str = "Knowledge Assistant",
                              domain_focus: Optional[str] = None) -> str:
        """
        Generate instructions for Custom GPT based on exported knowledge
        
        Args:
            knowledge_files: List of knowledge file names
            assistant_name: Name for the Custom GPT
            domain_focus: Optional domain focus (e.g., "sales", "research")
            
        Returns:
            GPT instructions text
        """
        instructions = f"""# {assistant_name}

You are a knowledgeable assistant with access to a curated knowledge base. Your primary role is to help users find information, answer questions, and provide insights based on the uploaded knowledge files.

## Knowledge Base Contents

The following knowledge files have been uploaded:
"""
        
        for filename in knowledge_files:
            instructions += f"- {filename}\n"
        
        if domain_focus:
            instructions += f"\n## Domain Focus\n\nYour expertise is focused on: {domain_focus}\n"
        
        instructions += """
## Instructions

1. **Primary Source**: Always base your answers on the information in the uploaded knowledge files
2. **Citation**: When possible, mention which document or section your information comes from
3. **Accuracy**: If you cannot find information in the knowledge base, clearly state this
4. **Context**: Provide relevant context and explain connections between different pieces of information
5. **Clarity**: Present information in a clear, organized manner
6. **Follow-up**: Suggest related topics or questions when appropriate

## Response Style

- Be conversational but professional
- Use bullet points and formatting to improve readability
- Provide specific examples when available in the knowledge base
- If asked about topics not covered in the knowledge base, acknowledge the limitation

## Special Capabilities

You have access to comprehensive information about:
"""
        
        # Get document types and subjects from database
        all_docs = self._get_all_documents()
        doc_types = set(doc.get("document_type", "Unknown") for doc in all_docs)
        
        for doc_type in sorted(doc_types):
            instructions += f"- {doc_type} documents\n"
        
        instructions += """
Always strive to provide the most helpful and accurate information based on your knowledge base.
"""
        
        return instructions
    
    def _get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from database"""
        # This would normally use the database search functionality
        # For now, we'll use a placeholder that would be implemented
        # when the database has a "get all" method
        
        # Placeholder - in actual implementation, this would query all docs
        search_results = self.db.search("", top_k=10000)  # Get many results
        
        documents = []
        for result in search_results:
            doc = {
                "id": result.get("id", ""),
                "content": result.get("content", ""),
                "document_id": result.get("document_id", ""),
                "section": result.get("section", ""),
                "subsection": result.get("subsection", ""),
                "document_type": result.get("document_type", "Unknown"),
                "filename": result.get("filename", ""),
                "source_file": result.get("source_file", ""),
                **result.get("metadata", {})
            }
            documents.append(doc)
        
        return documents
    
    def _group_by_document(self, docs: List[Dict]) -> Dict[str, List[Dict]]:
        """Group documents by document_id"""
        grouped = {}
        for doc in docs:
            doc_id = doc.get("document_id", "unknown")
            if doc_id not in grouped:
                grouped[doc_id] = []
            grouped[doc_id].append(doc)
        
        return grouped
    
    def _export_markdown(self, organized_docs: Dict, output_path: Path,
                        max_size_mb: int, include_metadata: bool) -> Dict[str, Any]:
        """Export as Markdown files"""
        max_size_bytes = max_size_mb * 1024 * 1024
        files_created = []
        
        for doc_group, docs in organized_docs.items():
            content = self._create_markdown_content(docs, include_metadata)
            
            # Split if too large
            if len(content.encode('utf-8')) > max_size_bytes:
                parts = self._split_content(content, max_size_bytes)
                for i, part in enumerate(parts):
                    filename = f"{doc_group}_part{i+1}.md"
                    file_path = output_path.parent / filename
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(part)
                    
                    files_created.append({
                        "filename": filename,
                        "path": str(file_path),
                        "size": len(part.encode('utf-8'))
                    })
            else:
                filename = f"{doc_group}.md"
                file_path = output_path.parent / filename
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                files_created.append({
                    "filename": filename,
                    "path": str(file_path),
                    "size": len(content.encode('utf-8'))
                })
        
        return {
            "status": "success",
            "format": "markdown",
            "files_created": files_created,
            "total_size": sum(f["size"] for f in files_created)
        }
    
    def _export_text(self, organized_docs: Dict, output_path: Path,
                    max_size_mb: int, include_metadata: bool) -> Dict[str, Any]:
        """Export as plain text files"""
        max_size_bytes = max_size_mb * 1024 * 1024
        files_created = []
        
        for doc_group, docs in organized_docs.items():
            content = self._create_text_content(docs, include_metadata)
            
            # Split if too large
            if len(content.encode('utf-8')) > max_size_bytes:
                parts = self._split_content(content, max_size_bytes)
                for i, part in enumerate(parts):
                    filename = f"{doc_group}_part{i+1}.txt"
                    file_path = output_path.parent / filename
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(part)
                    
                    files_created.append({
                        "filename": filename,
                        "path": str(file_path),
                        "size": len(part.encode('utf-8'))
                    })
            else:
                filename = f"{doc_group}.txt"
                file_path = output_path.parent / filename
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                files_created.append({
                    "filename": filename,
                    "path": str(file_path),
                    "size": len(content.encode('utf-8'))
                })
        
        return {
            "status": "success",
            "format": "text",
            "files_created": files_created,
            "total_size": sum(f["size"] for f in files_created)
        }
    
    def _export_json(self, organized_docs: Dict, output_path: Path,
                    max_size_mb: int, include_metadata: bool) -> Dict[str, Any]:
        """Export as JSON files"""
        max_size_bytes = max_size_mb * 1024 * 1024
        files_created = []
        
        for doc_group, docs in organized_docs.items():
            # Create structured JSON
            json_data = {
                "export_info": {
                    "timestamp": datetime.now().isoformat(),
                    "group": doc_group,
                    "document_count": len(set(doc.get("document_id", "") for doc in docs)),
                    "chunk_count": len(docs)
                },
                "documents": docs
            }
            
            content = json.dumps(json_data, indent=2, ensure_ascii=False)
            
            # Split if too large
            if len(content.encode('utf-8')) > max_size_bytes:
                # For JSON, we need to split the documents array
                chunk_size = max_size_bytes // 2  # Conservative estimate
                doc_chunks = self._chunk_list(docs, chunk_size)
                
                for i, doc_chunk in enumerate(doc_chunks):
                    part_data = {
                        "export_info": {
                            "timestamp": datetime.now().isoformat(),
                            "group": f"{doc_group}_part{i+1}",
                            "part": i+1,
                            "total_parts": len(doc_chunks),
                            "document_count": len(set(doc.get("document_id", "") for doc in doc_chunk)),
                            "chunk_count": len(doc_chunk)
                        },
                        "documents": doc_chunk
                    }
                    
                    part_content = json.dumps(part_data, indent=2, ensure_ascii=False)
                    filename = f"{doc_group}_part{i+1}.json"
                    file_path = output_path.parent / filename
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(part_content)
                    
                    files_created.append({
                        "filename": filename,
                        "path": str(file_path),
                        "size": len(part_content.encode('utf-8'))
                    })
            else:
                filename = f"{doc_group}.json"
                file_path = output_path.parent / filename
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                files_created.append({
                    "filename": filename,
                    "path": str(file_path),
                    "size": len(content.encode('utf-8'))
                })
        
        return {
            "status": "success",
            "format": "json",
            "files_created": files_created,
            "total_size": sum(f["size"] for f in files_created)
        }
    
    def _create_markdown_content(self, docs: List[Dict], include_metadata: bool) -> str:
        """Create markdown content from documents"""
        content = []
        current_doc_id = None
        current_section = None
        
        # Sort by document_id and section
        sorted_docs = sorted(docs, key=lambda x: (
            x.get("document_id", ""),
            x.get("section", "") or "",
            x.get("chunk_index", 0)
        ))
        
        for doc in sorted_docs:
            doc_id = doc.get("document_id", "")
            section = doc.get("section", "")
            subsection = doc.get("subsection", "")
            
            # New document
            if doc_id != current_doc_id:
                current_doc_id = doc_id
                current_section = None
                
                filename = doc.get("filename", f"Document {doc_id}")
                content.append(f"\n# {filename}\n")
                
                if include_metadata:
                    content.append("## Document Information\n")
                    content.append(f"- **Document ID**: {doc_id}\n")
                    if doc.get("document_type"):
                        content.append(f"- **Type**: {doc.get('document_type')}\n")
                    if doc.get("source_file"):
                        content.append(f"- **Source**: {doc.get('source_file')}\n")
                    content.append("\n---\n")
            
            # New section
            if section and section != current_section:
                current_section = section
                content.append(f"\n## {section}\n")
            
            # Subsection
            if subsection:
                content.append(f"\n### {subsection}\n")
            
            # Content
            content.append(f"{doc.get('content', '')}\n")
        
        return "\n".join(content)
    
    def _create_text_content(self, docs: List[Dict], include_metadata: bool) -> str:
        """Create plain text content from documents"""
        content = []
        current_doc_id = None
        
        # Sort by document_id and section
        sorted_docs = sorted(docs, key=lambda x: (
            x.get("document_id", ""),
            x.get("section", "") or "",
            x.get("chunk_index", 0)
        ))
        
        for doc in sorted_docs:
            doc_id = doc.get("document_id", "")
            
            # New document
            if doc_id != current_doc_id:
                current_doc_id = doc_id
                
                filename = doc.get("filename", f"Document {doc_id}")
                content.append(f"\n{'='*60}\n{filename}\n{'='*60}\n")
                
                if include_metadata and doc.get("document_type"):
                    content.append(f"Document Type: {doc.get('document_type')}\n")
                
                content.append("")
            
            # Section info
            if doc.get("section"):
                content.append(f"[{doc.get('section')}]")
                if doc.get("subsection"):
                    content.append(f"  [{doc.get('subsection')}]")
                content.append("")
            
            # Content
            content.append(doc.get('content', ''))
            content.append("")
        
        return "\n".join(content)
    
    def _create_index_content(self, files_created: List[Dict], all_docs: List[Dict]) -> str:
        """Create index/README content"""
        content = [
            "# Knowledge Base Export",
            "",
            f"**Export Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Documents**: {len(set(doc['document_id'] for doc in all_docs))}",
            f"**Total Content Chunks**: {len(all_docs)}",
            "",
            "## Files in this Export",
            ""
        ]
        
        for file_info in files_created:
            if file_info.get("is_index"):
                continue
                
            content.append(f"- **{file_info['filename']}**")
            if "document_type" in file_info:
                content.append(f"  - Type: {file_info['document_type']}")
            if "document_count" in file_info:
                content.append(f"  - Documents: {file_info['document_count']}")
            if "chunk_count" in file_info:
                content.append(f"  - Chunks: {file_info['chunk_count']}")
            content.append(f"  - Size: {file_info['file_size']:,} bytes")
            content.append("")
        
        # Document types summary
        doc_types = {}
        for doc in all_docs:
            doc_type = doc.get("document_type", "Unknown")
            if doc_type not in doc_types:
                doc_types[doc_type] = set()
            doc_types[doc_type].add(doc.get("document_id", ""))
        
        content.extend([
            "## Document Types",
            ""
        ])
        
        for doc_type, doc_ids in doc_types.items():
            content.append(f"- **{doc_type}**: {len(doc_ids)} documents")
        
        content.extend([
            "",
            "## Usage",
            "",
            "These files contain the complete knowledge base exported from Instant-DB.",
            "You can upload these files to Custom GPT or other AI assistants that support",
            "knowledge file uploads.",
            "",
            "For best results:",
            "1. Upload all files to maintain complete context",
            "2. Use the generated instructions when configuring your AI assistant",
            "3. Test with sample queries to ensure proper knowledge access"
        ])
        
        return "\n".join(content)
    
    def _split_content(self, content: str, max_bytes: int) -> List[str]:
        """Split content into parts that fit within size limit"""
        content_bytes = content.encode('utf-8')
        if len(content_bytes) <= max_bytes:
            return [content]
        
        # Split by paragraphs to maintain readability
        paragraphs = content.split('\n\n')
        parts = []
        current_part = []
        current_size = 0
        
        for paragraph in paragraphs:
            para_bytes = len(paragraph.encode('utf-8')) + 2  # +2 for \n\n
            
            if current_size + para_bytes > max_bytes and current_part:
                # Save current part and start new one
                parts.append('\n\n'.join(current_part))
                current_part = [paragraph]
                current_size = para_bytes
            else:
                current_part.append(paragraph)
                current_size += para_bytes
        
        # Add final part
        if current_part:
            parts.append('\n\n'.join(current_part))
        
        return parts
    
    def _chunk_list(self, items: List, max_size_estimate: int) -> List[List]:
        """Split list into chunks based on estimated size"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for item in items:
            # Rough size estimate
            item_size = len(json.dumps(item, ensure_ascii=False))
            
            if current_size + item_size > max_size_estimate and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [item]
                current_size = item_size
            else:
                current_chunk.append(item)
                current_size += item_size
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks 