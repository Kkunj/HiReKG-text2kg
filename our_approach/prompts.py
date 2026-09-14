"""
Knowledge Graph Creation Pipeline Prompts
"""


from textwrap import dedent
from typing import Iterable


LOCAL_ENTITY_SYSTEM_PROMPT = dedent(
    """
    You are an expert entity extraction agent. Identify concise entities (<=3 words)
    that are explicitly present in the provided text chunk. Favor high recall and
    include people, organizations, locations, artifacts, and key concepts.
    Return unique entities for the chunk without inventing unsupported items.
    
    Return valid JSON matching this exact format:
    {
      "entities": ["entity one", "entity two", "entity three"]
    }
    
    Ensure all strings are properly quoted and no trailing commas exist.
    """
).strip()


def build_local_entity_user_prompt(chunk_text: str) -> str:
    return dedent(
        f"""
        Extract entities from this chunk.
        Chunk:
        ---
        {chunk_text.strip()}
        ---
        """
    ).strip()


ENTITY_REFINEMENT_SYSTEM_PROMPT = dedent(
    """
    You are a global entity canonization assistant. Given the summary
    and the entity list, produce a refined entity list that can be used as a subject in the knowledge graph.

    Your tasks:
      * Merge ONLY entities that are highly semantically similar.
      * Remove exact duplicates or trivial variations (e.g., "disease" vs "disease condition")
      * Keep entities that represent distinct concepts, even if they share words
      * Keep only entities that are meaningful for knowledge graph construction
      * Be lenient: when in doubt, keep the entity rather than merge it

    Guidelines:
      - Merge entities only when one is a clear expansion or descriptor of the other (e.g., "disease" and "disease type")
      - Preserve entities with different specificity levels (e.g., "market" and "emerging market" represent distinct concepts)
      - Do not merge unless absolutely certain they refer to the same concept
      - Example: DO merge "effects of climate change" into "climate change" (redundant expansion)
      - Example: DO NOT merge "cancer" and "breast cancer" (preserve specificity distinctions)

    Return valid JSON matching this exact format:
    {
      "entities_refined": ["refined entity 1", "refined entity 2", "refined entity 3"]
    }

    Ensure all strings are properly quoted and no trailing commas exist.
    """
).strip()


def build_entity_refinement_user_prompt(raw_text: str, raw_entities: Iterable[str]) -> str:
    pretty_entities = "\n".join(f"- {entity}" for entity in raw_entities if entity)
    return dedent(
        f"""
        Summary: 
        ---
        {raw_text.strip()}
        ---

        High-recall entities:
        {pretty_entities}

        Produce the refined list per the instructions.
        """
    ).strip()


SUMMARY_SYSTEM_PROMPT = dedent(
    """
    You are a technical summarizer. Produce a cohesive 5-6 sentence summary that
    preserves the key actors, facts, and relationships from the input text.
    
    Return valid JSON matching this exact format:
    {
      "summary": "Your cohesive 5-6 sentence summary here"
    }
    
    Ensure all strings are properly quoted and no trailing commas exist.
    """
).strip()


def build_summary_user_prompt(raw_text: str) -> str:
    return dedent(
        f"""
        Summarize the following text in 5-6 sentences:
        ---
        {raw_text.strip()}
        ---
        """
    ).strip()


TRIPLE_EXTRACTION_SYSTEM_PROMPT = dedent(
    """
    You are a precise relation extraction model. You are given a summary of the text to provide you with the overall context, and you are also provided the specific chunk from which to extract the triples. Extract comprehensive subject-relation-object triples that capture all important and necessary information about the subjects in the chunk.
    
    IMPORTANT CONSTRAINTS:
    - The SUBJECT must come from the provided refined entity list
    - The OBJECT can be ANY meaningful information from the text - it does NOT need to be 
      from the entity list, but it MUST be present in the text (chunk text) and provide important context
    - Extract every important relationship and property related to each subject entity
    - Focus on completeness: capture all key facts, attributes, and relationships for each subject
    
    Relation names should be verbs or verb phrases in lowercase.
    Evidence is a short quote or sentence fragment proving the triple.
    
    Return valid JSON matching this exact format:
    {
      "triples": [
        {
          "subject": "entity one",
          "relation": "relation verb",
          "object": "any meaningful info from text",
          "evidence": "supporting quote from text"
        }
      ]
    }
    
    Ensure all strings are properly quoted and no trailing commas exist.
    """
).strip()


def build_triple_extraction_user_prompt(chunk_text: str, summary: str, entities: Iterable[str]) -> str:
    pretty_entities = ", ".join(sorted(set(e for e in entities if e)))
    return dedent(
        f"""
        Global summary:
        {summary.strip()}

        Refined entities:
        {pretty_entities}

        Chunk text:
        ---
        {chunk_text.strip()}
        ---
        """
    ).strip()


TRIPLE_VERIFICATION_SYSTEM_PROMPT = dedent(
    """
    You are a knowledge graph triple validator and rewriter. You are given an INVALID triple 
    where the relation contains either the subject or object string within it.
    
    This is problematic because:
    - "target singer" -> "has singing voice" -> "singing voice" is redundant 
      (relation "has singing voice" contains object "singing voice")
    - The relation should be a PURE verb/verb phrase that connects subject to object
    
    Your task: Rewrite the triple with a clean, concise relation that:
    1. Does NOT contain the subject string
    2. Does NOT contain the object string  
    3. Is a clear verb or verb phrase (e.g., "has", "produces", "improves", "uses")
    4. Preserves the semantic meaning of the original relationship
    
    Examples of corrections:
    - "has singing voice" with object "singing voice" → "has"
    - "receives improvement in speech quality" with object "speech quality" → "improves"
    - "is a type of machine learning" with object "machine learning" → "is type of"
    
    Return valid JSON matching this exact format:
    {
      "subject": "original subject",
      "relation": "clean verb phrase",
      "object": "original object"
    }
    
    Ensure all strings are properly quoted and no trailing commas exist.
    """
).strip()


def build_triple_verification_user_prompt(
    subject: str, 
    relation: str, 
    object_val: str,
    evidence: str = ""
) -> str:
    evidence_section = f"\nEvidence: {evidence.strip()}" if evidence else ""
    return dedent(
        f"""
        The following triple is INVALID because the relation contains the subject or object:
        
        Subject: {subject}
        Relation: {relation}
        Object: {object_val}{evidence_section}
        
        Please rewrite this triple with a clean relation that does not contain 
        the subject or object strings.
        """
    ).strip()


