"""
Neo4j Knowledge Graph Writer

This module implements the Neo4j knowledge graph writer. It provides a class for managing the Neo4j connection and adding triplets to the graph database.
"""

import sys
from pathlib import Path
from typing import Iterable, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# from Neo4j_instance import Neo4jKnowledgeGraph  # noqa: E402


TripleTuple = Tuple[str, str, str]

import os
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


class Neo4jKnowledgeGraph:
    """
    Neo4j Knowledge Graph manager for storing and querying triplets.
    """
    
    def __init__(self, uri: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (default: from .env NEO4J_URI)
            username: Neo4j username (default: from .env NEO4J_USERNAME)
            password: Neo4j password (default: from .env NEO4J_PASSWORD)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "neo4j+s://e61fcb7a.databases.neo4j.io")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "AT7hulWiD_lP5b_bIvf_nnIbhP4J5VugtPaVm49nqYk")
        
        if not self.password:
            raise ValueError("Neo4j password not provided. Set NEO4J_PASSWORD in .env file.")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test connection
            self.driver.verify_connectivity()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")
        print(f"[OK] Connected to Neo4j at {self.uri}")
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def add_triplet(self, subject: str, relation: str, obj: str) -> bool:
        """
        Add a single triplet to the knowledge graph.
        
        Args:
            subject: Subject entity
            relation: Relationship type
            obj: Object entity
            
        Returns:
            True if successful, False otherwise
        """
        return self.add_triplets([(subject, relation, obj)])
    
    def add_triplets(self, triplets: List[Tuple[str, str, str]], batch_size: int = 100) -> bool:
        """
        Add multiple triplets to the knowledge graph in batches.
        
        Args:
            triplets: List of (subject, relation, object) tuples
            batch_size: Number of triplets to process in each batch
            
        Returns:
            True if successful, False otherwise
        """
        if not triplets:
            print("No triplets to add")
            return True
        
        try:
            total_batches = (len(triplets) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(triplets))
                batch = triplets[start_idx:end_idx]
                
                with self.driver.session() as session:
                    session.execute_write(self._create_triplets_batch, batch)
                
                print(f"✅ Processed batch {batch_idx + 1}/{total_batches} ({len(batch)} triplets)")
            
            print(f"🎉 Successfully added {len(triplets)} triplets to Neo4j")
            return True
            
        except Exception as e:
            print(f"❌ Error adding triplets: {e}")
            return False
    
    @staticmethod
    def _create_triplets_batch(tx, batch: List[Tuple[str, str, str]]):
        """
        Transaction function to create a batch of triplets.
        Uses MERGE to avoid duplicates and CREATE for relationships.
        Handles both 3-tuples (subject, relation, object) and 
        4-tuples (subject, relation, object, evidence).
        """
        if not batch:
            return
        
        # Check if batch contains evidence (4-tuples) or not (3-tuples)
        first_item_length = len(batch[0])
        
        if first_item_length == 4:
            # Batch contains evidence
            query = """
            UNWIND $triplets AS triplet
            MERGE (s:Entity {name: triplet.subject})
            MERGE (o:Entity {name: triplet.object})
            CREATE (s)-[r:RELATION {type: triplet.relation, evidence: triplet.evidence}]->(o)
            """
            triplets_data = [
                {
                    "subject": item[0],
                    "relation": item[1],
                    "object": item[2],
                    "evidence": item[3] if item[3] else None
                }
                for item in batch
            ]
        else:
            # Batch without evidence
            query = """
            UNWIND $triplets AS triplet
            MERGE (s:Entity {name: triplet.subject})
            MERGE (o:Entity {name: triplet.object})
            CREATE (s)-[r:RELATION {type: triplet.relation}]->(o)
            """
            triplets_data = [
                {
                    "subject": item[0],
                    "relation": item[1],
                    "object": item[2]
                }
                for item in batch
            ]
        
        tx.run(query, triplets=triplets_data)
    
    def clear_graph(self) -> bool:
        """
        Clear all nodes and relationships from the graph.
        ⚠️ USE WITH CAUTION - This deletes all data!
        """
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            print("✅ Graph cleared successfully")
            return True
        except Exception as e:
            print(f"❌ Error clearing graph: {e}")
            return False
    
    def get_stats(self) -> dict:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary with node count, relationship count, etc.
        """
        try:
            with self.driver.session() as session:
                # Count nodes
                node_result = session.run("MATCH (n:Entity) RETURN count(n) as count")
                node_count = node_result.single()["count"]
                
                # Count relationships
                rel_result = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) as count")
                rel_count = rel_result.single()["count"]
                
                # Get unique relation types
                types_result = session.run(
                    "MATCH ()-[r:RELATION]->() RETURN DISTINCT r.type as type"
                )
                relation_types = [record["type"] for record in types_result]
                
                stats = {
                    "nodes": node_count,
                    "relationships": rel_count,
                    "unique_relations": len(relation_types),
                    "relation_types": relation_types
                }
                
                return stats
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {}
    
    def print_stats(self):
        """Print knowledge graph statistics in a nice format."""
        stats = self.get_stats()
        if stats:
            print("\n" + "=" * 60)
            print("📊 KNOWLEDGE GRAPH STATISTICS")
            print("=" * 60)
            print(f"Entities (nodes):      {stats['nodes']}")
            print(f"Relationships:         {stats['relationships']}")
            print(f"Unique relation types: {stats['unique_relations']}")
            print("=" * 60)
    
    def query(self, cypher_query: str, parameters: dict = None):
        """
        Execute a custom Cypher query.
        
        Args:
            cypher_query: Cypher query string
            parameters: Query parameters (optional)
            
        Returns:
            Query results as a list of records
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, parameters or {})
                return list(result)
        except Exception as e:
            print(f"❌ Query error: {e}")
            return []


# Convenience function for simple usage
def add_triplets_to_neo4j(triplets: List[Tuple[str, str, str]], 
                          uri: Optional[str] = None,
                          username: Optional[str] = None,
                          password: Optional[str] = None) -> bool:
    """
    Convenience function to add triplets to Neo4j in one call.
    
    Args:
        triplets: List of (subject, relation, object) tuples
        uri: Neo4j URI (optional, from .env if not provided)
        username: Neo4j username (optional, from .env if not provided)
        password: Neo4j password (optional, from .env if not provided)
    
    Returns:
        True if successful, False otherwise
    
    Example:
        triplets = [("Apple", "founded_by", "Steve Jobs")]
        add_triplets_to_neo4j(triplets)
    """
    try:
        with Neo4jKnowledgeGraph(uri, username, password) as kg:
            return kg.add_triplets(triplets)
    except Exception as e:
        print(f"❌ Failed to add triplets: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    print("Testing Neo4j Knowledge Graph Integration...")
    
    # Sample triplets
    sample_triplets = [
        ("Apple Inc.", "founded_by", "Steve Jobs"),
        ("Steve Jobs", "born_in", "California"),
        ("Apple Inc.", "headquartered_in", "Cupertino"),
        ("iPhone", "manufactured_by", "Apple Inc."),
        ("Apple Inc.", "revolutionized", "personal computing"),
        ("Apple Inc.", "revolutionized", "mobile technology"),
    ]
    
    try:
        # Option 1: Using context manager (recommended)
        with Neo4jKnowledgeGraph() as kg:
            print("\n📝 Adding sample triplets...")
            kg.add_triplets(sample_triplets)
            
            print("\n📊 Knowledge Graph Statistics:")
            kg.print_stats()
        
        # Option 2: Using convenience function
        # add_triplets_to_neo4j(sample_triplets)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure to set the following in your .env file:")
        print("   NEO4J_URI=bolt://localhost:7687")
        print("   NEO4J_USERNAME=neo4j")
        print("   NEO4J_PASSWORD=your_password")


def write_triples_to_neo4j(triples: Iterable[TripleTuple]) -> bool:
    """
    Persist triples to Neo4j via the shared Neo4jKnowledgeGraph helper.
    """
    triples_list = [(s.strip(), r.strip(), o.strip()) for s, r, o in triples if s and r and o]
    if not triples_list:
        print("⚠️  No triples were provided for Neo4j ingestion.")
        return False

    with Neo4jKnowledgeGraph() as kg:
        kg.clear_graph()
        kg.add_triplets(triples_list)
        return kg.get_stats()


