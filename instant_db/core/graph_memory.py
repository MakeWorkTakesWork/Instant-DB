"""
Graph Memory Component for Instant-DB
Builds and maintains intelligent knowledge graphs from document content
"""

import json
import sqlite3
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer


class GraphMemoryEngine:
    """
    Creates and manages knowledge graphs from document content
    
    Features:
    - Entity extraction and relationship mapping
    - Concept clustering and hierarchy building
    - Context-aware memory formation
    - Graph-based reasoning for enhanced search
    """
    
    def __init__(self, db_path: Path, embedding_provider=None):
        """
        Initialize graph memory engine
        
        Args:
            db_path: Path to store graph database
            embedding_provider: Embedding model for entity vectorization
        """
        self.db_path = db_path
        self.graph_db_path = db_path / "graph_memory.db"
        
        # Initialize knowledge graph
        self.knowledge_graph = nx.MultiDiGraph()
        
        # Initialize embedding model for entities
        self.embedding_provider = embedding_provider or SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize graph database
        self._init_graph_db()
        
        # Load existing graph
        self._load_graph()
    
    def _init_graph_db(self):
        """Initialize SQLite database for persistent graph storage"""
        conn = sqlite3.connect(self.graph_db_path)
        
        # Entities table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                entity_text TEXT UNIQUE,
                entity_type TEXT,
                frequency INTEGER DEFAULT 1,
                embedding BLOB,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # Relationships table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity TEXT,
                target_entity TEXT,
                relationship_type TEXT,
                strength REAL DEFAULT 1.0,
                context TEXT,
                document_source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_entity) REFERENCES entities (entity_id),
                FOREIGN KEY (target_entity) REFERENCES entities (entity_id)
            )
        ''')
        
        # Concepts table (higher-level abstractions)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS concepts (
                concept_id TEXT PRIMARY KEY,
                concept_name TEXT,
                concept_type TEXT,
                related_entities TEXT,
                embedding BLOB,
                importance_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # Memory clusters (groups of related information)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS memory_clusters (
                cluster_id TEXT PRIMARY KEY,
                cluster_name TEXT,
                cluster_type TEXT,
                member_entities TEXT,
                cluster_summary TEXT,
                centroid_embedding BLOB,
                coherence_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def process_document_for_graph(self, document_id: str, content: str, 
                                  chunks: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """
        Process a document to extract entities, relationships, and concepts
        
        Args:
            document_id: Unique document identifier
            content: Full document content
            chunks: Document chunks from chunking engine
            metadata: Document metadata
            
        Returns:
            Dict with graph processing results
        """
        results = {
            'entities_extracted': 0,
            'relationships_found': 0,
            'concepts_created': 0,
            'memory_clusters': 0
        }
        
        try:
            # Step 1: Extract entities from content
            entities = self._extract_entities(content, document_id)
            results['entities_extracted'] = len(entities)
            
            # Step 2: Find relationships between entities
            relationships = self._extract_relationships(content, entities, document_id)
            results['relationships_found'] = len(relationships)
            
            # Step 3: Create higher-level concepts
            concepts = self._create_concepts(entities, relationships, content)
            results['concepts_created'] = len(concepts)
            
            # Step 4: Form memory clusters
            clusters = self._form_memory_clusters(entities, relationships, concepts)
            results['memory_clusters'] = len(clusters)
            
            # Step 5: Update knowledge graph
            self._update_knowledge_graph(entities, relationships, concepts, clusters)
            
            # Step 6: Persist to database
            self._persist_graph_data(entities, relationships, concepts, clusters)
            
            return {
                'status': 'success',
                'document_id': document_id,
                **results
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'document_id': document_id,
                'error': str(e)
            }
    
    def _extract_entities(self, content: str, document_id: str) -> List[Dict]:
        """
        Extract named entities, key concepts, and important terms
        
        This is a simplified version - in production, you'd use:
        - spaCy NER models
        - Custom domain-specific extractors
        - LLM-based entity extraction
        """
        entities = []
        
        # Simple extraction for demo (replace with sophisticated NER)
        import re
        
        # Extract potential entities using patterns
        patterns = {
            'MONEY': r'\$[\d,]+(?:\.\d{2})?',
            'PERCENT': r'\d+(?:\.\d+)?%',
            'PRODUCT': r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b',  # Title case phrases
            'PROCESS': r'\b\w+ing\b',  # Processes (ending in -ing)
            'METRIC': r'\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|years?|hours?)\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, content)
            for match in set(matches):  # Remove duplicates
                if len(match.strip()) > 2:  # Filter short matches
                    entity_id = hashlib.md5(f"{match}_{entity_type}".encode()).hexdigest()[:16]
                    
                    # Create entity embedding
                    embedding = self.embedding_provider.encode([match])[0]
                    
                    entities.append({
                        'entity_id': entity_id,
                        'text': match,
                        'type': entity_type,
                        'embedding': embedding,
                        'document_source': document_id,
                        'frequency': content.count(match)
                    })
        
        return entities
    
    def _extract_relationships(self, content: str, entities: List[Dict], 
                              document_id: str) -> List[Dict]:
        """
        Extract relationships between entities based on co-occurrence and context
        """
        relationships = []
        
        # Simple co-occurrence based relationships
        sentences = content.split('.')
        
        for sentence in sentences:
            sentence_entities = []
            for entity in entities:
                if entity['text'].lower() in sentence.lower():
                    sentence_entities.append(entity)
            
            # Create relationships between co-occurring entities
            for i, entity1 in enumerate(sentence_entities):
                for entity2 in sentence_entities[i+1:]:
                    
                    # Determine relationship type based on context
                    rel_type = self._determine_relationship_type(
                        entity1, entity2, sentence
                    )
                    
                    if rel_type:
                        relationships.append({
                            'source': entity1['entity_id'],
                            'target': entity2['entity_id'],
                            'type': rel_type,
                            'context': sentence.strip(),
                            'strength': self._calculate_relationship_strength(
                                entity1, entity2, sentence
                            ),
                            'document_source': document_id
                        })
        
        return relationships
    
    def _determine_relationship_type(self, entity1: Dict, entity2: Dict, 
                                   context: str) -> Optional[str]:
        """Determine the type of relationship between two entities"""
        
        # Simple rule-based relationship detection
        context_lower = context.lower()
        
        if any(word in context_lower for word in ['costs', 'price', 'budget']):
            if entity1['type'] == 'MONEY' or entity2['type'] == 'MONEY':
                return 'COSTS'
        
        if any(word in context_lower for word in ['improves', 'increases', 'reduces']):
            return 'AFFECTS'
        
        if any(word in context_lower for word in ['includes', 'contains', 'has']):
            return 'CONTAINS'
        
        if any(word in context_lower for word in ['requires', 'needs', 'depends']):
            return 'REQUIRES'
        
        if entity1['type'] == entity2['type']:
            return 'SIMILAR_TYPE'
        
        return 'RELATED'
    
    def _calculate_relationship_strength(self, entity1: Dict, entity2: Dict, 
                                       context: str) -> float:
        """Calculate the strength of a relationship based on various factors"""
        
        # Factors affecting relationship strength:
        # 1. Proximity in text
        # 2. Frequency of co-occurrence
        # 3. Semantic similarity
        # 4. Contextual indicators
        
        base_strength = 1.0
        
        # Adjust based on context length (closer = stronger)
        if len(context) < 100:
            base_strength *= 1.5
        
        # Adjust based on entity types
        if entity1['type'] == entity2['type']:
            base_strength *= 1.2
        
        # Adjust based on frequency
        avg_frequency = (entity1.get('frequency', 1) + entity2.get('frequency', 1)) / 2
        if avg_frequency > 3:
            base_strength *= 1.3
        
        return min(base_strength, 3.0)  # Cap at 3.0
    
    def _create_concepts(self, entities: List[Dict], relationships: List[Dict], 
                        content: str) -> List[Dict]:
        """
        Create higher-level concepts by clustering related entities
        """
        concepts = []
        
        # Group entities by type to form concepts
        entity_types = {}
        for entity in entities:
            entity_type = entity['type']
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity)
        
        # Create concept for each entity type cluster
        for concept_type, type_entities in entity_types.items():
            if len(type_entities) > 1:  # Only create concepts for multiple entities
                
                concept_id = hashlib.md5(f"{concept_type}_concept".encode()).hexdigest()[:16]
                
                # Calculate concept embedding as centroid of entity embeddings
                embeddings = np.array([e['embedding'] for e in type_entities])
                concept_embedding = np.mean(embeddings, axis=0)
                
                # Calculate importance score
                total_frequency = sum(e.get('frequency', 1) for e in type_entities)
                importance_score = min(total_frequency / 10.0, 1.0)
                
                concepts.append({
                    'concept_id': concept_id,
                    'name': f"{concept_type.title()} Concept",
                    'type': concept_type,
                    'entities': [e['entity_id'] for e in type_entities],
                    'embedding': concept_embedding,
                    'importance_score': importance_score,
                    'description': f"Collection of {concept_type.lower()} entities"
                })
        
        return concepts
    
    def _form_memory_clusters(self, entities: List[Dict], relationships: List[Dict], 
                             concepts: List[Dict]) -> List[Dict]:
        """
        Form memory clusters - coherent groups of related information
        """
        clusters = []
        
        # Create clusters based on highly connected entity groups
        if not relationships:
            return clusters
        
        # Build temporary graph for clustering
        temp_graph = nx.Graph()
        
        # Add entities as nodes
        for entity in entities:
            temp_graph.add_node(entity['entity_id'], **entity)
        
        # Add relationships as edges
        for rel in relationships:
            if rel['strength'] > 1.5:  # Only strong relationships
                temp_graph.add_edge(
                    rel['source'], 
                    rel['target'], 
                    weight=rel['strength']
                )
        
        # Find connected components as clusters
        try:
            connected_components = list(nx.connected_components(temp_graph))
            
            for i, component in enumerate(connected_components):
                if len(component) > 2:  # Minimum cluster size
                    
                    cluster_id = hashlib.md5(f"cluster_{i}".encode()).hexdigest()[:16]
                    
                    # Get entities in cluster
                    cluster_entities = [
                        entity for entity in entities 
                        if entity['entity_id'] in component
                    ]
                    
                    # Calculate cluster centroid embedding
                    embeddings = np.array([e['embedding'] for e in cluster_entities])
                    centroid_embedding = np.mean(embeddings, axis=0)
                    
                    # Calculate coherence score
                    coherence_score = self._calculate_cluster_coherence(
                        cluster_entities, relationships
                    )
                    
                    # Generate cluster summary
                    cluster_summary = self._generate_cluster_summary(cluster_entities)
                    
                    clusters.append({
                        'cluster_id': cluster_id,
                        'name': f"Memory Cluster {i+1}",
                        'type': 'SEMANTIC_CLUSTER',
                        'entities': list(component),
                        'summary': cluster_summary,
                        'centroid_embedding': centroid_embedding,
                        'coherence_score': coherence_score
                    })
        
        except Exception as e:
            print(f"Error in clustering: {e}")
        
        return clusters
    
    def _calculate_cluster_coherence(self, entities: List[Dict], 
                                   relationships: List[Dict]) -> float:
        """Calculate how coherent/related the entities in a cluster are"""
        
        if len(entities) < 2:
            return 0.0
        
        entity_ids = {e['entity_id'] for e in entities}
        
        # Count internal relationships
        internal_rels = [
            r for r in relationships 
            if r['source'] in entity_ids and r['target'] in entity_ids
        ]
        
        # Calculate coherence based on relationship density
        max_possible_rels = len(entities) * (len(entities) - 1) / 2
        if max_possible_rels == 0:
            return 0.0
        
        coherence = len(internal_rels) / max_possible_rels
        return min(coherence, 1.0)
    
    def _generate_cluster_summary(self, entities: List[Dict]) -> str:
        """Generate a human-readable summary of what a cluster represents"""
        
        if not entities:
            return "Empty cluster"
        
        # Group by entity type
        type_counts = {}
        for entity in entities:
            entity_type = entity['type']
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        # Create summary
        summary_parts = []
        for entity_type, count in type_counts.items():
            if count == 1:
                summary_parts.append(f"1 {entity_type.lower()}")
            else:
                summary_parts.append(f"{count} {entity_type.lower()}s")
        
        return f"Cluster containing {', '.join(summary_parts)}"
    
    def graph_enhanced_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform graph-enhanced search that considers relationships and concepts
        """
        results = []
        
        try:
            # Step 1: Find directly relevant entities
            query_embedding = self.embedding_provider.encode([query])[0]
            
            # Get all entities and calculate similarity
            conn = sqlite3.connect(self.graph_db_path)
            cursor = conn.execute('SELECT * FROM entities')
            
            entity_similarities = []
            for row in cursor.fetchall():
                entity_id, text, entity_type, freq, embedding_blob, _, _, metadata = row
                
                if embedding_blob:
                    entity_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    similarity = np.dot(query_embedding, entity_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(entity_embedding)
                    )
                    
                    entity_similarities.append({
                        'entity_id': entity_id,
                        'text': text,
                        'type': entity_type,
                        'similarity': similarity,
                        'frequency': freq
                    })
            
            # Sort by similarity
            entity_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Step 2: Expand search using graph relationships
            expanded_entities = set()
            for entity in entity_similarities[:5]:  # Top 5 most similar
                expanded_entities.add(entity['entity_id'])
                
                # Find related entities
                cursor = conn.execute('''
                    SELECT target_entity, relationship_type, strength 
                    FROM relationships 
                    WHERE source_entity = ? AND strength > 1.0
                    ORDER BY strength DESC
                    LIMIT 3
                ''', (entity['entity_id'],))
                
                for target, rel_type, strength in cursor.fetchall():
                    expanded_entities.add(target)
            
            # Step 3: Rank expanded results
            for entity_id in expanded_entities:
                cursor = conn.execute(
                    'SELECT entity_text, entity_type, frequency FROM entities WHERE entity_id = ?',
                    (entity_id,)
                )
                result = cursor.fetchone()
                if result:
                    text, entity_type, frequency = result
                    
                    # Calculate composite score
                    base_similarity = next(
                        (e['similarity'] for e in entity_similarities if e['entity_id'] == entity_id),
                        0.0
                    )
                    
                    # Boost score based on graph centrality and frequency
                    centrality_boost = min(frequency / 10.0, 0.3)
                    composite_score = base_similarity + centrality_boost
                    
                    results.append({
                        'entity_id': entity_id,
                        'text': text,
                        'type': entity_type,
                        'similarity': base_similarity,
                        'composite_score': composite_score,
                        'frequency': frequency,
                        'search_type': 'graph_enhanced'
                    })
            
            conn.close()
            
            # Sort by composite score and return top results
            results.sort(key=lambda x: x['composite_score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            print(f"Error in graph-enhanced search: {e}")
            return []
    
    def get_entity_relationships(self, entity_id: str) -> Dict[str, Any]:
        """Get all relationships for a specific entity"""
        
        conn = sqlite3.connect(self.graph_db_path)
        
        # Get entity info
        cursor = conn.execute(
            'SELECT entity_text, entity_type, frequency FROM entities WHERE entity_id = ?',
            (entity_id,)
        )
        entity_info = cursor.fetchone()
        
        if not entity_info:
            return {'error': 'Entity not found'}
        
        # Get outgoing relationships
        cursor = conn.execute('''
            SELECT r.target_entity, r.relationship_type, r.strength, r.context, e.entity_text
            FROM relationships r
            JOIN entities e ON r.target_entity = e.entity_id
            WHERE r.source_entity = ?
            ORDER BY r.strength DESC
        ''', (entity_id,))
        
        outgoing = []
        for target_id, rel_type, strength, context, target_text in cursor.fetchall():
            outgoing.append({
                'target_entity': target_id,
                'target_text': target_text,
                'relationship_type': rel_type,
                'strength': strength,
                'context': context
            })
        
        # Get incoming relationships
        cursor = conn.execute('''
            SELECT r.source_entity, r.relationship_type, r.strength, r.context, e.entity_text
            FROM relationships r
            JOIN entities e ON r.source_entity = e.entity_id
            WHERE r.target_entity = ?
            ORDER BY r.strength DESC
        ''', (entity_id,))
        
        incoming = []
        for source_id, rel_type, strength, context, source_text in cursor.fetchall():
            incoming.append({
                'source_entity': source_id,
                'source_text': source_text,
                'relationship_type': rel_type,
                'strength': strength,
                'context': context
            })
        
        conn.close()
        
        return {
            'entity_id': entity_id,
            'entity_text': entity_info[0],
            'entity_type': entity_info[1],
            'frequency': entity_info[2],
            'outgoing_relationships': outgoing,
            'incoming_relationships': incoming,
            'total_connections': len(outgoing) + len(incoming)
        }
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph"""
        
        conn = sqlite3.connect(self.graph_db_path)
        
        # Basic counts
        entity_count = conn.execute('SELECT COUNT(*) FROM entities').fetchone()[0]
        relationship_count = conn.execute('SELECT COUNT(*) FROM relationships').fetchone()[0]
        concept_count = conn.execute('SELECT COUNT(*) FROM concepts').fetchone()[0]
        cluster_count = conn.execute('SELECT COUNT(*) FROM memory_clusters').fetchone()[0]
        
        # Entity types distribution
        cursor = conn.execute('''
            SELECT entity_type, COUNT(*), AVG(frequency)
            FROM entities 
            GROUP BY entity_type 
            ORDER BY COUNT(*) DESC
        ''')
        entity_types = []
        for entity_type, count, avg_freq in cursor.fetchall():
            entity_types.append({
                'type': entity_type,
                'count': count,
                'avg_frequency': round(avg_freq, 2)
            })
        
        # Relationship types
        cursor = conn.execute('''
            SELECT relationship_type, COUNT(*), AVG(strength)
            FROM relationships 
            GROUP BY relationship_type 
            ORDER BY COUNT(*) DESC
        ''')
        relationship_types = []
        for rel_type, count, avg_strength in cursor.fetchall():
            relationship_types.append({
                'type': rel_type,
                'count': count,
                'avg_strength': round(avg_strength, 2)
            })
        
        conn.close()
        
        return {
            'total_entities': entity_count,
            'total_relationships': relationship_count,
            'total_concepts': concept_count,
            'total_clusters': cluster_count,
            'entity_types': entity_types,
            'relationship_types': relationship_types,
            'graph_density': relationship_count / max(entity_count * (entity_count - 1) / 2, 1)
        }
    
    def _update_knowledge_graph(self, entities: List[Dict], relationships: List[Dict],
                               concepts: List[Dict], clusters: List[Dict]):
        """Update the in-memory NetworkX graph"""
        
        # Add entities as nodes
        for entity in entities:
            self.knowledge_graph.add_node(
                entity['entity_id'],
                text=entity['text'],
                entity_type=entity['type'],
                frequency=entity.get('frequency', 1)
            )
        
        # Add relationships as edges
        for rel in relationships:
            self.knowledge_graph.add_edge(
                rel['source'],
                rel['target'],
                relationship_type=rel['type'],
                strength=rel['strength'],
                context=rel.get('context', '')
            )
    
    def _persist_graph_data(self, entities: List[Dict], relationships: List[Dict],
                           concepts: List[Dict], clusters: List[Dict]):
        """Persist graph data to SQLite database"""
        
        conn = sqlite3.connect(self.graph_db_path)
        
        # Insert/update entities
        for entity in entities:
            embedding_blob = entity['embedding'].tobytes()
            
            conn.execute('''
                INSERT OR REPLACE INTO entities 
                (entity_id, entity_text, entity_type, frequency, embedding, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                entity['entity_id'],
                entity['text'],
                entity['type'],
                entity.get('frequency', 1),
                embedding_blob,
                datetime.now()
            ))
        
        # Insert relationships
        for rel in relationships:
            conn.execute('''
                INSERT INTO relationships 
                (source_entity, target_entity, relationship_type, strength, context, document_source)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                rel['source'],
                rel['target'],
                rel['type'],
                rel['strength'],
                rel.get('context', ''),
                rel.get('document_source', '')
            ))
        
        # Insert concepts
        for concept in concepts:
            embedding_blob = concept['embedding'].tobytes()
            entities_json = json.dumps(concept['entities'])
            
            conn.execute('''
                INSERT OR REPLACE INTO concepts
                (concept_id, concept_name, concept_type, related_entities, 
                 embedding, importance_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                concept['concept_id'],
                concept['name'],
                concept['type'],
                entities_json,
                embedding_blob,
                concept['importance_score'],
                json.dumps(concept)
            ))
        
        # Insert clusters
        for cluster in clusters:
            embedding_blob = cluster['centroid_embedding'].tobytes()
            entities_json = json.dumps(cluster['entities'])
            
            conn.execute('''
                INSERT OR REPLACE INTO memory_clusters
                (cluster_id, cluster_name, cluster_type, member_entities,
                 cluster_summary, centroid_embedding, coherence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                cluster['cluster_id'],
                cluster['name'],
                cluster['type'],
                entities_json,
                cluster['summary'],
                embedding_blob,
                cluster['coherence_score']
            ))
        
        conn.commit()
        conn.close()
    
    def _load_graph(self):
        """Load existing graph from database"""
        
        if not self.graph_db_path.exists():
            return
        
        conn = sqlite3.connect(self.graph_db_path)
        
        # Load entities
        cursor = conn.execute('SELECT entity_id, entity_text, entity_type, frequency FROM entities')
        for entity_id, text, entity_type, frequency in cursor.fetchall():
            self.knowledge_graph.add_node(
                entity_id,
                text=text,
                entity_type=entity_type,
                frequency=frequency
            )
        
        # Load relationships
        cursor = conn.execute('''
            SELECT source_entity, target_entity, relationship_type, strength, context
            FROM relationships
        ''')
        for source, target, rel_type, strength, context in cursor.fetchall():
            if self.knowledge_graph.has_node(source) and self.knowledge_graph.has_node(target):
                self.knowledge_graph.add_edge(
                    source, target,
                    relationship_type=rel_type,
                    strength=strength,
                    context=context
                )
        
        conn.close() 