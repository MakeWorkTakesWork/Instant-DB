"""
Intelligent text chunking engine for Instant-DB
Smart chunking with section awareness and overlap management
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    id: str
    content: str
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    section: Optional[str] = None
    subsection: Optional[str] = None
    chunk_type: str = "content"  # content, header, table, list, etc.
    word_count: int = 0
    char_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived fields"""
        if not self.word_count:
            self.word_count = len(self.content.split())
        if not self.char_count:
            self.char_count = len(self.content)
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for chunk"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.document_id}_chunk_{self.chunk_index}_{content_hash}"


class ChunkingEngine:
    """
    Intelligent text chunking engine with section awareness
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100,
                 respect_sentence_boundaries: bool = True,
                 respect_paragraph_boundaries: bool = True):
        """
        Initialize chunking engine
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size in characters
            respect_sentence_boundaries: Whether to avoid splitting sentences
            respect_paragraph_boundaries: Whether to avoid splitting paragraphs
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_sentence_boundaries = respect_sentence_boundaries
        self.respect_paragraph_boundaries = respect_paragraph_boundaries
        
        # Section detection patterns
        self.section_patterns = [
            # Headers with #, ##, ###, etc.
            r'^(#{1,6})\s+(.+)$',
            
            # Numbered sections
            r'^(\d+\.?\d*\.?)\s+(.+)$',
            
            # Roman numerals
            r'^([IVXLCDM]+\.?)\s+(.+)$',
            
            # Letters
            r'^([A-Z]\.)\s+(.+)$',
            
            # All caps sections
            r'^([A-Z\s]{3,}):?\s*$'
        ]
        
        # Sentence boundary detection
        self.sentence_endings = r'[.!?]+(?:\s+|$)'
        
        # Paragraph detection
        self.paragraph_sep = r'\n\s*\n'
    
    def chunk_text(self, text: str, document_id: str, metadata: Optional[Dict] = None) -> List[TextChunk]:
        """
        Chunk text into intelligent segments
        
        Args:
            text: Input text to chunk
            document_id: Unique document identifier
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        
        # Detect document structure
        structure = self._analyze_document_structure(text)
        
        # Create chunks based on structure
        chunks = self._create_structured_chunks(text, document_id, structure, metadata)
        
        # Post-process chunks
        chunks = self._post_process_chunks(chunks)
        
        return chunks
    
    def _analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure to identify sections, paragraphs, etc."""
        lines = text.split('\n')
        structure = {
            'sections': [],
            'paragraphs': [],
            'lists': [],
            'tables': [],
            'special_blocks': []
        }
        
        current_section = None
        current_subsection = None
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                continue
            
            # Check for section headers
            section_match = self._detect_section_header(line_stripped)
            if section_match:
                level, title = section_match
                section_info = {
                    'level': level,
                    'title': title,
                    'line_number': i,
                    'start_char': sum(len(lines[j]) + 1 for j in range(i)),
                }
                
                if level <= 2:
                    current_section = title
                    current_subsection = None
                else:
                    current_subsection = title
                
                structure['sections'].append(section_info)
                continue
            
            # Check for lists
            if self._is_list_item(line_stripped):
                structure['lists'].append({
                    'line_number': i,
                    'content': line_stripped,
                    'section': current_section,
                    'subsection': current_subsection
                })
            
            # Check for tables (simple detection)
            if '|' in line_stripped and line_stripped.count('|') >= 2:
                structure['tables'].append({
                    'line_number': i,
                    'content': line_stripped,
                    'section': current_section,
                    'subsection': current_subsection
                })
        
        return structure
    
    def _detect_section_header(self, line: str) -> Optional[Tuple[int, str]]:
        """Detect section headers and return (level, title)"""
        for pattern in self.section_patterns:
            match = re.match(pattern, line, re.MULTILINE)
            if match:
                marker = match.group(1)
                title = match.group(2).strip()
                
                # Determine level based on marker
                if marker.startswith('#'):
                    level = len(marker)
                elif marker.isdigit() or '.' in marker:
                    level = marker.count('.') + 1
                elif marker in ['I', 'II', 'III', 'IV', 'V']:
                    level = 2
                elif len(marker) == 2 and marker.endswith('.'):
                    level = 3
                else:
                    level = 1
                
                return (level, title)
        
        return None
    
    def _is_list_item(self, line: str) -> bool:
        """Check if line is a list item"""
        list_patterns = [
            r'^\s*[-*+]\s+',  # Bullet lists
            r'^\s*\d+\.?\s+',  # Numbered lists
            r'^\s*[a-zA-Z]\.?\s+',  # Letter lists
            r'^\s*[ivxlcdm]+\.?\s+',  # Roman numeral lists
        ]
        
        return any(re.match(pattern, line) for pattern in list_patterns)
    
    def _create_structured_chunks(self, text: str, document_id: str, 
                                structure: Dict, metadata: Dict) -> List[TextChunk]:
        """Create chunks based on document structure"""
        chunks = []
        
        # Split into paragraphs first
        paragraphs = re.split(self.paragraph_sep, text)
        
        current_chunk_text = ""
        current_section = None
        current_subsection = None
        start_char = 0
        chunk_index = 0
        
        for para in paragraphs:
            if not para.strip():
                continue
            
            # Check if this paragraph starts a new section
            first_line = para.split('\n')[0].strip()
            section_match = self._detect_section_header(first_line)
            
            if section_match:
                # Save current chunk if it has content
                if current_chunk_text.strip():
                    chunk = self._create_chunk(
                        current_chunk_text.strip(),
                        document_id,
                        chunk_index,
                        start_char,
                        start_char + len(current_chunk_text),
                        current_section,
                        current_subsection,
                        metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    start_char += len(current_chunk_text)
                    current_chunk_text = ""
                
                # Update section info
                level, title = section_match
                if level <= 2:
                    current_section = title
                    current_subsection = None
                else:
                    current_subsection = title
            
            # Check if adding this paragraph would exceed chunk size
            potential_text = current_chunk_text + "\n\n" + para if current_chunk_text else para
            
            if len(potential_text) > self.chunk_size and current_chunk_text:
                # Create chunk with current content
                chunk = self._create_chunk(
                    current_chunk_text.strip(),
                    document_id,
                    chunk_index,
                    start_char,
                    start_char + len(current_chunk_text),
                    current_section,
                    current_subsection,
                    metadata
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk_text)
                start_char += len(current_chunk_text) - len(overlap_text)
                current_chunk_text = overlap_text + "\n\n" + para
            else:
                # Add paragraph to current chunk
                if current_chunk_text:
                    current_chunk_text += "\n\n" + para
                else:
                    current_chunk_text = para
        
        # Add final chunk
        if current_chunk_text.strip():
            chunk = self._create_chunk(
                current_chunk_text.strip(),
                document_id,
                chunk_index,
                start_char,
                start_char + len(current_chunk_text),
                current_section,
                current_subsection,
                metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, content: str, document_id: str, chunk_index: int,
                     start_char: int, end_char: int, section: Optional[str],
                     subsection: Optional[str], metadata: Dict) -> TextChunk:
        """Create a TextChunk object"""
        return TextChunk(
            id="",  # Will be generated in __post_init__
            content=content,
            document_id=document_id,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            section=section,
            subsection=subsection,
            metadata=metadata.copy()
        )
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        if len(text) <= self.chunk_overlap:
            return text
        
        overlap_start = len(text) - self.chunk_overlap
        
        # Try to respect sentence boundaries
        if self.respect_sentence_boundaries:
            sentences = re.split(self.sentence_endings, text[overlap_start:])
            if len(sentences) > 1:
                # Keep complete sentences in overlap
                return sentences[0] + text[overlap_start + len(sentences[0]):]
        
        # Try to respect word boundaries
        overlap_text = text[overlap_start:]
        if ' ' in overlap_text:
            words = overlap_text.split()
            if len(words) > 1:
                # Start from first complete word
                first_space = overlap_text.index(' ')
                return overlap_text[first_space + 1:]
        
        return overlap_text
    
    def _post_process_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Post-process chunks to ensure quality"""
        processed_chunks = []
        
        for chunk in chunks:
            # Skip chunks that are too small
            if chunk.char_count < self.min_chunk_size:
                # Try to merge with previous chunk if possible
                if processed_chunks and processed_chunks[-1].section == chunk.section:
                    last_chunk = processed_chunks[-1]
                    if last_chunk.char_count + chunk.char_count <= self.chunk_size * 1.5:
                        # Merge chunks
                        merged_content = last_chunk.content + "\n\n" + chunk.content
                        merged_chunk = TextChunk(
                            id="",
                            content=merged_content,
                            document_id=chunk.document_id,
                            chunk_index=last_chunk.chunk_index,
                            start_char=last_chunk.start_char,
                            end_char=chunk.end_char,
                            section=last_chunk.section,
                            subsection=last_chunk.subsection,
                            metadata=last_chunk.metadata
                        )
                        processed_chunks[-1] = merged_chunk
                        continue
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def get_chunking_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about the chunking results"""
        if not chunks:
            return {}
        
        char_counts = [chunk.char_count for chunk in chunks]
        word_counts = [chunk.word_count for chunk in chunks]
        sections = set(chunk.section for chunk in chunks if chunk.section)
        
        return {
            "total_chunks": len(chunks),
            "avg_chars_per_chunk": sum(char_counts) / len(char_counts),
            "avg_words_per_chunk": sum(word_counts) / len(word_counts),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
            "total_chars": sum(char_counts),
            "total_words": sum(word_counts),
            "sections_detected": len(sections),
            "sections": list(sections)
        } 