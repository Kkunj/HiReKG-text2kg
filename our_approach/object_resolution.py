"""
Object Resolution for Knowledge Graph Triple Repair

Resolves triples where objects contain refined entities as substrings,
restructuring them so that entities become proper nodes in the knowledge graph.
"""

import json
import logging
import os
import re
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

import nltk
from nltk.stem import WordNetLemmatizer
import openai
from dotenv import load_dotenv
from llm_client import strip_thinking

if TYPE_CHECKING:
    from pipeline import TripleRecord, ChunkArtifacts

load_dotenv()


class ObjectResolution:
    """
    Resolves triples where objects contain refined entities as substrings.
    Restructures them to ensure entities are proper nodes in the knowledge graph.
    """
    
    def __init__(self, llm_client=None, api_key: Optional[str] = None, model: str = "chatgpt-4o-latest", logger: Optional[logging.Logger] = None):
        """
        Initialize the ObjectResolution processor.
        
        Args:
            llm_client: Optional LLMClient instance for making LLM calls (preferred)
            api_key: OpenAI API key (if None, will try to read from environment) - used only if llm_client not provided
            model: OpenAI model to use for repair - used only if llm_client not provided
            logger: Optional logger instance (if None, will use module logger)
        """
        self.lemmatizer = WordNetLemmatizer()
        self.determiners = {"a", "an", "the"}
        self.model = model
        # Use provided logger or fall back to module logger
        self.logger = logger or logging.getLogger("KGPipeline")
        
        # If a logger is provided, add its handlers to the root logger
        # so that module-level logging calls also go to the same outputs
        if logger:
            root_logger = logging.getLogger()
            for handler in logger.handlers:
                if handler not in root_logger.handlers:
                    root_logger.addHandler(handler)
            root_logger.setLevel(logging.DEBUG)
        
        # Use provided LLMClient or fall back to direct OpenAI client
        self.llm_client = llm_client
        self.client = None
        
        if llm_client is not None:
            # Use the provided LLMClient for all LLM calls
            self.model = llm_client.model
            logging.debug(f"[ObjectResolution] Using provided LLMClient (model: {self.model}, backend: {llm_client.model_type})")
        else:
            # Fall back to direct OpenAI client (legacy behavior)
            logging.debug(f"[ObjectResolution] No LLMClient provided, falling back to direct OpenAI client")
            if api_key:
                openai.api_key = api_key
            elif os.getenv("OPENAI_API_KEY"):
                openai.api_key = os.getenv("OPENAI_API_KEY")
            else:
                raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
            
            self.client = openai.OpenAI(api_key=openai.api_key)

    def _call_llm(self, system_message: str, user_message: str, temperature: float = 0.3, max_tokens: int = 500, json_mode: bool = True) -> str:
        """
        Unified LLM call method that uses either LLMClient or direct OpenAI client.

        Args:
            system_message: System prompt
            user_message: User prompt
            temperature: Temperature for generation
            max_tokens: Maximum tokens for output
            json_mode: If True (default), force JSON response and use generate_json's
                retry/repair logic. If False, request plain text — required for prompts
                that explicitly ask for a single natural-language sentence (otherwise
                the model wraps the answer in JSON and downstream parsing breaks).

        Returns:
            LLM response text (with <think> tags stripped)
        """
        if self.llm_client is not None:
            if json_mode:
                # Use generate_json for proper JSON parsing, retry, and fix logic
                payload = self.llm_client.generate_json(
                    system_prompt=system_message,
                    user_prompt=user_message,
                    max_output_tokens=max_tokens,
                )
                # Return as JSON string so downstream parsing still works
                return json.dumps(payload)
            # Plain-text path: skip response_format=json_object so the model returns
            # the free-form sentence the prompt actually asks for.
            return strip_thinking(
                self.llm_client._run_request(
                    system_prompt=system_message,
                    user_prompt=user_message,
                    response_format=None,
                    max_output_tokens=max_tokens,
                )
            )
        else:
            # Use direct OpenAI client (legacy)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content.strip()
            # Strip <think>...</think> blocks from response
            return strip_thinking(content)
  

    def process_multiple_entities(
        self,
        triple: Dict[str, Any],
        matched_entities: List[str],
        chunk_text: str,
        entities_refined: List[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Use LLM with explicit step-by-step guidance to resolve multi-entity objects.
        
        Provides the LLM with:
        1. Entity positions in the object
        2. Step-by-step decomposition guide
        3. Clear validation criteria
        
        Args:
            triple: Original triple dict
            matched_entities: List of entities found in object (in order of length, not position)
            chunk_text: Chunk text for context
            entities_refined: List of all refined entities
            
        Returns:
            List of repaired triples, or None if repair fails
        """
        subject = triple["subject"]
        relation = triple["relation"]
        obj = triple["object"]
        
        # IMPORTANT: Re-sort matched_entities by their position in the object text
        # The input list is sorted by length (longest first for overlap resolution),
        # but we need them in order of appearance for sequential position analysis
        obj_lower = obj.lower()
        
        def find_position(entity: str) -> int:
            """Find the first occurrence position of entity in object."""
            entity_lower = entity.lower()
            pattern = r'\b' + re.escape(entity_lower) + r'\b'
            match = re.search(pattern, obj_lower)
            if match:
                return match.start()
            # Fallback to simple find
            idx = obj_lower.find(entity_lower)
            return idx if idx != -1 else len(obj_lower)  # Put unfound entities at end
        
        # Sort by position in text (left-to-right order)
        matched_entities_sorted = sorted(matched_entities, key=find_position)
        
        logging.debug(f"    [LLM-GUIDED] Processing {len(matched_entities_sorted)} entities (sorted by position): {matched_entities_sorted}")
        
        # Log if order changed
        if matched_entities_sorted != matched_entities:
            logging.debug(f"    [LLM-GUIDED] Re-sorted from original order: {matched_entities}")
        
        # Use the sorted list from here on
        matched_entities = matched_entities_sorted
        
        # Step 1: Analyze entity positions
        entity_analysis = self._analyze_entity_positions(obj, matched_entities)
        if not entity_analysis:
            logging.warning(f"    [LLM-GUIDED] Could not analyze entity positions")
            return None
        
        logging.debug(f"    [LLM-GUIDED] Entity analysis:")
        for i, analysis in enumerate(entity_analysis['entities']):
            logging.debug(f"      Entity {i+1}: '{analysis['entity']}' | "
                        f"Prefix: '{analysis['prefix']}' | "
                        f"Suffix: '{analysis['suffix']}'")
        
        # Step 2: Create structured prompt with reasoning guide
        system_msg, user_msg = self._create_reasoning_prompt(
            triple, matched_entities, entity_analysis, chunk_text
        )
        
        # Step 3: Call LLM with reasoning instructions
        try:
            logging.debug(f"    [LLM-GUIDED] Calling LLM with structured reasoning prompt...")
            
            response_text = self._call_llm(
                system_message=system_msg,
                user_message=user_msg,
                temperature=0.1,
                max_tokens=3000
            )
            
            logging.debug(f"    [LLM-GUIDED] Raw response:\n{response_text}")
            
            # Step 4: Parse response
            repaired_triples = self._parse_reasoning_response(response_text)
            
            if not repaired_triples:
                logging.warning(f"    [LLM-GUIDED] Failed to parse response")
                return None
            
            # Step 5: Validate repaired triples
            if self._validate_multi_entity_repair(repaired_triples, matched_entities, subject):
                logging.debug(f"    [LLM-GUIDED] ✓ Successfully created {len(repaired_triples)} triples")
                return repaired_triples
            else:
                logging.warning(f"    [LLM-GUIDED] Validation failed")
                return None
                
        except Exception as e:
            logging.error(f"    [LLM-GUIDED] Error: {str(e)}")
            return None


    def _analyze_entity_positions(
        self,
        obj: str,
        matched_entities: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze where each entity appears in the object and extract surrounding context.
        
        Returns:
            Dict with entity positions, prefixes, suffixes, and connectors
        """
        import re
        
        # Find character positions of each entity
        entity_positions = []
        remaining_text = obj.lower()
        offset = 0
        
        for entity in matched_entities:
            entity_lower = entity.lower()
            # Use word boundaries for accurate matching
            pattern = r'\b' + re.escape(entity_lower) + r'\b'
            match = re.search(pattern, remaining_text)
            
            if match:
                start = offset + match.start()
                end = offset + match.end()
                entity_positions.append({
                    'entity': entity,
                    'start': start,
                    'end': end
                })
                # Update for next search
                offset = end
                remaining_text = obj.lower()[end:]
            else:
                # Fallback: find anywhere
                idx = remaining_text.find(entity_lower)
                if idx != -1:
                    start = offset + idx
                    end = start + len(entity_lower)
                    entity_positions.append({
                        'entity': entity,
                        'start': start,
                        'end': end
                    })
                    offset = end
                    remaining_text = obj.lower()[end:]
                else:
                    logging.warning(f"    Could not find '{entity}' in remaining text")
                    return None
        
        if not entity_positions:
            return None
        
        # Extract text segments
        analysis = {
            'original_object': obj,
            'entities': []
        }
        
        for i, pos in enumerate(entity_positions):
            # Prefix: text from start (or previous entity end) to current entity start
            if i == 0:
                prefix = obj[:pos['start']]
            else:
                prefix = obj[entity_positions[i-1]['end']:pos['start']]
            
            # Suffix: text from current entity end to next entity start (or end)
            if i < len(entity_positions) - 1:
                suffix = obj[pos['end']:entity_positions[i+1]['start']]
            else:
                suffix = obj[pos['end']:]
            
            analysis['entities'].append({
                'entity': pos['entity'],
                'position': i + 1,
                'prefix': prefix.strip(),
                'suffix': suffix.strip()
            })
        
        return analysis


    def _create_reasoning_prompt(
        self,
        triple: Dict[str, Any],
        matched_entities: List[str],
        entity_analysis: Dict[str, Any],
        chunk_text: str
    ) -> Tuple[str, str]:
        """
        Create a structured prompt with step-by-step reasoning guide.
        """
        
        system_msg = """You are a knowledge graph expert specializing in triple decomposition.

        Your task: Decompose a triple where the object contains MULTIPLE entities into clean, meaningful sub-triples.

        CORE PRINCIPLE:
        Each entity must become a proper object in at least one triple. No entity should remain embedded within another object string.

        STEP-BY-STEP REASONING PROCESS:

        Step 1: UNDERSTAND THE STRUCTURE
        - Examine how entities are connected in the original object
        - Identify connector words (and, or, through, to, of, in, etc.)
        - Determine relationship type: List, Chain, or Nested

        Step 2: CHOOSE DECOMPOSITION STRATEGY

        Strategy A - HUB (for lists/collections):
        When entities are parallel items (A, B, and C) or (A and B):
        - Create one triple per entity
        - All triples share the same subject
        - Absorb descriptive prefix into relation

        Example:
        Input: Subject="System" | Relation="uses" | Object="advanced algorithm A, technique B, and method C"
        → Strategy: HUB (comma-separated list)
        → Output:
        [
            {"subject": "System", "relation": "uses advanced", "object": "algorithm A"},
            {"subject": "System", "relation": "uses", "object": "technique B"},
            {"subject": "System", "relation": "uses", "object": "method C"}
        ]

        Strategy B - CHAIN (for sequential/causal):
        When entities connect sequentially (A through B to C) or (A leading to B):
        - First triple: Subject → Entity1
        - Subsequent: Entity[i] → Entity[i+1]
        - Use connector words as relations

        Example:
        Input: Subject="Process" | Relation="transforms" | Object="data A into format B through stage C"
        → Strategy: CHAIN (sequential with 'into', 'through')
        → Output:
        [
            {"subject": "Process", "relation": "transforms", "object": "data A"},
            {"subject": "data A", "relation": "into", "object": "format B"},
            {"subject": "format B", "relation": "through", "object": "stage C"}
        ]

        Strategy C - NESTED (for hierarchical):
        When entities have parent-child relationships (A of B) or (A in B):
        - First triple: Subject → First Entity
        - Subsequent: Entity[i] → Entity[i+1] (with preposition)

        Example:
        Input: Subject="Study" | Relation="examines" | Object="component A of system B in domain C"
        → Strategy: NESTED (hierarchical with 'of', 'in')
        → Output:
        [
            {"subject": "Study", "relation": "examines", "object": "component A"},
            {"subject": "component A", "relation": "part of", "object": "system B"},
            {"subject": "system B", "relation": "within", "object": "domain C"}
        ]

        Step 3: VALIDATE OUTPUT
        ✓ Every matched entity appears as an object
        ✓ No empty subjects, relations, or objects
        ✓ Relations are concise (2-7 words)
        ✓ Semantic meaning preserved

        OUTPUT FORMAT:
        Return ONLY a JSON object with a "triples" key containing the array. No explanation before or after.
        {"triples": [
            {"subject": "...", "relation": "...", "object": "..."},
            {"subject": "...", "relation": "...", "object": "..."}
        ]}"""

        # Build detailed entity breakdown
        entity_breakdown = []
        for i, ent_info in enumerate(entity_analysis['entities'], 1):
            entity_breakdown.append(
                f"  Entity {i}: \"{ent_info['entity']}\"\n"
                f"    - Text before: \"{ent_info['prefix']}\"\n"
                f"    - Text after: \"{ent_info['suffix']}\""
            )
        
        entity_breakdown_str = "\n".join(entity_breakdown)
        
        user_msg = f"""CONTEXT (for semantic understanding):
        {chunk_text[:500]}...

        ORIGINAL TRIPLE TO DECOMPOSE:
        Subject: "{triple['subject']}"
        Relation: "{triple['relation']}"
        Object: "{triple['object']}"

        ENTITIES DETECTED IN OBJECT (in order):
        {matched_entities}

        DETAILED ENTITY ANALYSIS:
        {entity_breakdown_str}

        YOUR TASK:
        Follow the 3-step reasoning process:
        1. Understand the structure (List/Chain/Nested?)
        2. Choose appropriate decomposition strategy
        3. Generate and validate triples

        Return ONLY the JSON array of decomposed triples."""

        return system_msg, user_msg


    def _parse_reasoning_response(
        self,
        response_text: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Parse LLM response that may contain reasoning followed by JSON.
        Handles both {"triples": [...]} dict format and raw [...] array format.
        """
        try:
            # First, try parsing as JSON directly (covers generate_json path)
            parsed = json.loads(response_text)
            if isinstance(parsed, dict):
                triples = parsed.get("triples", [])
            elif isinstance(parsed, list):
                triples = parsed
            else:
                logging.warning(f"    [PARSE] Unexpected JSON type: {type(parsed)}")
                return None

            if not isinstance(triples, list):
                logging.warning(f"    [PARSE] 'triples' value is not a list")
                return None
            
            # Validate and clean each triple
            validated = []
            for t in triples:
                if not isinstance(t, dict):
                    continue
                if not all(k in t for k in ['subject', 'relation', 'object']):
                    continue
                
                validated.append({
                    'subject': str(t['subject']).strip(),
                    'relation': str(t['relation']).strip(),
                    'object': str(t['object']).strip()
                })
            
            return validated if validated else None
            
        except json.JSONDecodeError as e:
            logging.error(f"    [PARSE] JSON decode error: {e}")
            return None
        except Exception as e:
            logging.error(f"    [PARSE] Unexpected error: {e}")
            return None


    def _validate_multi_entity_repair(
        self,
        repaired_triples: List[Dict[str, Any]],
        matched_entities: List[str],
        original_subject: str
    ) -> bool:
        """
        Validate that the repaired triples are acceptable.
        
        Returns:
            True if validation passes, False otherwise
        """
        if not repaired_triples:
            logging.warning(f"    Validation failed: No repaired triples")
            return False
        
        # Check 1: All matched entities appear as objects
        objects_in_repaired = [t['object'] for t in repaired_triples]
        
        # Normalize for comparison
        def normalize(s):
            return s.lower().strip()
        
        objects_norm = [normalize(o) for o in objects_in_repaired]
        entities_norm = [normalize(e) for e in matched_entities]
        
        missing_entities = []
        for entity in entities_norm:
            # Check if entity appears in any object
            if not any(entity in obj or obj in entity for obj in objects_norm):
                missing_entities.append(entity)
        
        if missing_entities:
            logging.warning(
                f"    Validation failed: Entities not covered: {missing_entities}"
            )
            return False
        
        # Check 2: At least one triple starts with original subject
        subjects_in_repaired = [normalize(t['subject']) for t in repaired_triples]
        if normalize(original_subject) not in subjects_in_repaired:
            logging.warning(
                f"    Validation warning: Original subject '{original_subject}' "
                f"not found in repaired subjects"
            )
            # This is a warning, not a failure
        
        # Check 3: No empty fields
        for i, t in enumerate(repaired_triples):
            if not t['subject'] or not t['relation'] or not t['object']:
                logging.warning(
                    f"    Validation failed: Empty field in triple {i+1}: {t}"
                )
                return False
        
        # Check 4: Relations are reasonable length (not too long)
        for i, t in enumerate(repaired_triples):
            rel_words = len(t['relation'].split())
            if rel_words > 10:
                logging.warning(
                    f"    Validation warning: Relation too long ({rel_words} words) "
                    f"in triple {i+1}"
                )
        
        logging.debug(f"    ✓ Validation passed: {len(repaired_triples)} triples")
        return True
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent matching.
        
        Steps:
        1. Lowercase
        2. Strip edge punctuation
        3. Lemmatize words
        4. Remove leading determiners
        5. Trim whitespace
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower().strip()
        
        # Strip punctuation at edges
        text = text.strip(".,!?;:'\"()[]{}").strip()
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        if not tokens:
            return ""
        
        # Lemmatize each token
        lemmatized = []
        for token in tokens:
            # Simple lemmatization (assume noun by default)
            lemma = self.lemmatizer.lemmatize(token, pos='n')
            # Also try verb
            lemma_v = self.lemmatizer.lemmatize(token, pos='v')
            # Use shorter form (more normalized)
            if len(lemma_v) < len(lemma):
                lemma = lemma_v
            lemmatized.append(lemma)
        
        # Remove leading determiners
        while lemmatized and lemmatized[0] in self.determiners:
            lemmatized = lemmatized[1:]
        
        # Join back
        normalized = " ".join(lemmatized)
        
        return normalized.strip()
   
    def prepare_entities(self, entities_refined: List[str]) -> List[Dict[str, str]]:
        """
        Prepare entities with normalized forms for matching.
        
        Args:
            entities_refined: List of refined entity strings
            
        Returns:
            List of dicts with 'original' and 'norm' keys
        """
        prepared = []
        for entity in entities_refined:
            norm = self.normalize_text(entity)
            if norm:  # Only keep non-empty normalized entities
                prepared.append({
                    "original": entity,
                    "norm": norm
                })
        return prepared
    
    def find_entity_in_object(self, object_norm: str, entity_norm: str) -> Optional[Tuple[int, int]]:
        """
        Find entity in object using word boundary matching.
        
        Args:
            object_norm: Normalized object string
            entity_norm: Normalized entity string
            
        Returns:
            Tuple of (start, end) character indices if found, None otherwise
        """
        if not entity_norm or not object_norm:
            return None
        
        # Use word boundary regex for more accurate matching
        pattern = r'\b' + re.escape(entity_norm) + r'\b'
        match = re.search(pattern, object_norm)
        
        if match:
            return (match.start(), match.end())
        
        return None

    def detect_candidate_triples(
        self, 
        triples_raw: List[Dict[str, Any]], 
        entities_prepared: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Detect which triples need repair (contain refined entities in objects).
        
        Args:
            triples_raw: List of raw triples
            entities_prepared: List of prepared entities with normalized forms
            
        Returns:
            Tuple of (candidate_triples, clean_triples)
        """
        candidates = []
        clean_triples = []
        
        for idx, triple in enumerate(triples_raw, 1):
            subj = triple.subject
            rel = triple.relation
            obj = triple.object
            
            logging.debug(f"\n    --- Evaluating Triple #{idx} ---")
            logging.debug(f"    Subject: '{subj}'")
            logging.debug(f"    Relation: '{rel}'")
            logging.debug(f"    Object: '{obj}'")
            
            obj_norm = self.normalize_text(obj)
            logging.debug(f"    Object normalized: '{obj_norm}'")
            
            # Find all matching entities in this object
            matches = []
            for entity in entities_prepared:
                span = self.find_entity_in_object(obj_norm, entity["norm"])
                if span:
                    matches.append({
                        "entity": entity,
                        "span": span,
                        "length": len(entity["norm"])
                    })
                    logging.debug(f"      ✓ Found entity match: '{entity['original']}' at position {span}")
            
            # Resolve overlapping matches - prefer longest
            if matches:
                logging.debug(f"    Found {len(matches)} entity match(es) in object")
                
                # Sort by length (descending) then by start position
                matches.sort(key=lambda x: (-x["length"], x["span"][0]))
                
                # Remove overlapping matches (keep longest)
                non_overlapping = []
                for match in matches:
                    overlaps = False
                    for kept in non_overlapping:
                        # Check if spans overlap
                        if not (match["span"][1] <= kept["span"][0] or match["span"][0] >= kept["span"][1]):
                            overlaps = True
                            break
                    if not overlaps:
                        non_overlapping.append(match)
                
                # If we have at least one match, this is a candidate
                if non_overlapping:
                    matched_entities = [m["entity"]["original"] for m in non_overlapping]
                    logging.debug(f"    ❌ NEEDS REPAIR - Contains entities: {matched_entities}")
                    
                    # Use the first (longest) match as the primary matched entity
                    # Convert TripleRecord to dict for easier manipulation
                    triple_dict = {
                        "subject": triple.subject,
                        "relation": triple.relation,
                        "object": triple.object,
                        "evidence": triple.evidence if hasattr(triple, 'evidence') else None,
                        "chunk_id": triple.chunk_id if hasattr(triple, 'chunk_id') else None
                    }
                    candidates.append({
                        "triple": triple_dict,
                        "matched_entity": non_overlapping[0]["entity"]["original"],
                        "all_matches": matched_entities
                    })
                else:
                    logging.debug(f"    ✓ CLEAN - No valid entity matches")
                    # Convert TripleRecord to dict
                    triple_dict = {
                        "subject": triple.subject,
                        "relation": triple.relation,
                        "object": triple.object,
                        "evidence": triple.evidence if hasattr(triple, 'evidence') else None,
                        "chunk_id": triple.chunk_id if hasattr(triple, 'chunk_id') else None
                    }
                    clean_triples.append(triple_dict)
            else:
                # No matches, triple is clean
                logging.debug(f"    ✓ CLEAN - No entity matches found")
                # Convert TripleRecord to dict
                triple_dict = {
                    "subject": triple.subject,
                    "relation": triple.relation,
                    "object": triple.object,
                    "evidence": triple.evidence if hasattr(triple, 'evidence') else None,
                    "chunk_id": triple.chunk_id if hasattr(triple, 'chunk_id') else None
                }
                clean_triples.append(triple_dict)
        
        return candidates, clean_triples
    
    def process_single_entity(
        self,
        triple: Dict[str, Any],
        matched_entity: str,
        chunk_text: str
    ) -> Optional[List[Dict[str, str]]]:
        """
        Process a triple where the object contains exactly ONE refined entity.
        
        This uses a precise approach:
        1. Find exact token span of entity in object
        2. Extract prefix and suffix around entity
        3. Normalize to form [prefix, entity] (entity at end)
        4. If suffix exists, use LLM to rewrite safely
        5. Merge prefix into relation
        
        Args:
            triple: The triple to repair
            matched_entity: The single matched entity in the object
            chunk_text: Original chunk text for context
            
        Returns:
            List containing the repaired triple, or None on error
        """
        subj = triple["subject"]
        relation = triple["relation"]
        obj = triple["object"]
        
        logging.debug(f"      [SINGLE-ENTITY] Processing with precise method...")
        logging.debug(f"      [SINGLE-ENTITY] Object: '{obj}'")
        logging.debug(f"      [SINGLE-ENTITY] Matched entity: '{matched_entity}'")
        
        # Step 1: Tokenize and normalize
        obj_tokens = nltk.word_tokenize(obj)
        entity_tokens = nltk.word_tokenize(matched_entity)
        
        logging.debug(f"      [SINGLE-ENTITY] Object tokens: {obj_tokens}")
        logging.debug(f"      [SINGLE-ENTITY] Entity tokens: {entity_tokens}")
        
        # Normalize tokens for matching (lemmatize, remove determiners)
        def normalize_token(token):
            token_lower = token.lower()
            if token_lower in self.determiners:
                return None
            lemma = self.lemmatizer.lemmatize(token_lower, pos='n')
            lemma_v = self.lemmatizer.lemmatize(token_lower, pos='v')
            return lemma_v if len(lemma_v) < len(lemma) else lemma
        
        obj_tokens_norm = [normalize_token(t) for t in obj_tokens]
        entity_tokens_norm = [normalize_token(t) for t in entity_tokens if normalize_token(t) is not None]
        
        logging.debug(f"      [SINGLE-ENTITY] Object tokens normalized: {obj_tokens_norm}")
        logging.debug(f"      [SINGLE-ENTITY] Entity tokens normalized: {entity_tokens_norm}")
        
        # Step 1.2: Find entity span in object using token-level matching
        entity_start_idx = None
        entity_end_idx = None
        
        for i in range(len(obj_tokens_norm) - len(entity_tokens_norm) + 1):
            # Check if entity tokens match at position i (allowing None/determiners)
            match = True
            j = 0
            for k in range(i, len(obj_tokens_norm)):
                if j >= len(entity_tokens_norm):
                    break
                if obj_tokens_norm[k] is None:
                    continue  # Skip determiners in object
                if obj_tokens_norm[k] == entity_tokens_norm[j]:
                    j += 1
                elif j == 0:
                    match = False
                    break
                else:
                    match = False
                    break
            
            if match and j == len(entity_tokens_norm):
                entity_start_idx = i
                # Find actual end index (including any determiners)
                entity_end_idx = i
                matched_count = 0
                for k in range(i, len(obj_tokens_norm)):
                    if obj_tokens_norm[k] is not None:
                        matched_count += 1
                    entity_end_idx = k + 1
                    if matched_count == len(entity_tokens_norm):
                        break
                break
        
        if entity_start_idx is None:
            logging.warning(f"      [SINGLE-ENTITY] Could not find entity span in object, falling back to generic repair")
            return None
        
        logging.debug(f"      [SINGLE-ENTITY] Found entity span: tokens [{entity_start_idx}:{entity_end_idx}]")
        
        # Step 2: Extract prefix, entity, suffix
        prefix_tokens = obj_tokens[:entity_start_idx]
        entity_span_tokens = obj_tokens[entity_start_idx:entity_end_idx]
        suffix_tokens = obj_tokens[entity_end_idx:]
        
        prefix = " ".join(prefix_tokens).strip()
        entity_text = " ".join(entity_span_tokens).strip()
        suffix = " ".join(suffix_tokens).strip()
        
        logging.debug(f"      [SINGLE-ENTITY] Decomposed object:")
        logging.debug(f"        Prefix: '{prefix}'")
        logging.debug(f"        Entity: '{entity_text}'")
        logging.debug(f"        Suffix: '{suffix}'")
        
        # Determine which case we're in
        has_prefix = bool(prefix)
        has_suffix = bool(suffix)
        
        # Case 1: Only prefix exists → merge prefix into relation (current approach)
        if has_prefix and not has_suffix:
            logging.debug(f"      [SINGLE-ENTITY] Case 1: Only prefix exists, merging into relation...")
            
            # Clean up prefix before merging
            prefix_clean = prefix.strip(",.:;!?")
            
            # Remove leading trivial words that break grammar
            prefix_tokens_clean = prefix_clean.split()
            while prefix_tokens_clean and prefix_tokens_clean[0].lower() in ["and", "then", "or"]:
                prefix_tokens_clean = prefix_tokens_clean[1:]
            prefix_clean = " ".join(prefix_tokens_clean)
            
            # Build new relation
            new_relation = f"{relation} {prefix_clean}".strip()
            
            # Ensure relation isn't too long (max ~10 tokens)
            rel_tokens = new_relation.split()
            if len(rel_tokens) > 10:
                logging.warning(f"      [SINGLE-ENTITY] Relation too long ({len(rel_tokens)} tokens), truncating prefix")
                new_relation = f"{relation} {' '.join(rel_tokens[len(relation.split()):8])}"
            
            logging.debug(f"      [SINGLE-ENTITY] New relation: '{new_relation}'")
            new_object = matched_entity
        
        # Case 2 & 3: Suffix exists (with or without prefix) → use LLM to restructure triplet
        else:
            if has_suffix and not has_prefix:
                logging.debug(f"      [SINGLE-ENTITY] Case 2: Only suffix exists, using LLM to restructure triplet...")
            elif has_suffix and has_prefix:
                logging.debug(f"      [SINGLE-ENTITY] Case 3: Both prefix and suffix exist, using LLM to restructure triplet...")
            else:
                # No prefix and no suffix - entity IS the object, no changes needed
                logging.debug(f"      [SINGLE-ENTITY] No prefix or suffix, entity is the object, no changes needed")
                new_relation = relation
                new_object = matched_entity
                repaired_triple = {
                    "subject": subj,
                    "relation": new_relation,
                    "object": new_object
                }
                logging.debug(f"      [SINGLE-ENTITY] ✓ Repaired triple:")
                logging.debug(f"        Subject: '{repaired_triple['subject']}'")
                logging.debug(f"        Relation: '{repaired_triple['relation']}'")
                logging.debug(f"        Object: '{repaired_triple['object']}'")
                return [repaired_triple]
            
            # Call LLM to restructure the entire triplet
            new_triplet = self._rewrite_to_triplet_form(subj, relation, obj, matched_entity)
            
            if new_triplet is None:
                logging.warning(f"      [SINGLE-ENTITY] LLM restructuring failed, falling back to manual approach")
                # Fallback: combine prefix and suffix into relation
                combined = f"{prefix} {suffix}".strip() if prefix else suffix
                combined_clean = combined.strip(",.:;!?")
                new_relation = f"{relation} {combined_clean}".strip() if combined_clean else relation
                new_object = matched_entity
            else:
                logging.debug(f"      [SINGLE-ENTITY] LLM rewritten triplet: '{new_triplet}'")
                
                # Extract new relation from the restructured triplet
                new_relation = self._extract_relation_from_triplet(new_triplet, subj, matched_entity)
                
                if new_relation is None:
                    logging.warning(f"      [SINGLE-ENTITY] Could not extract relation from triplet, using fallback")
                    # Fallback: combine prefix and suffix into relation
                    combined = f"{prefix} {suffix}".strip() if prefix else suffix
                    combined_clean = combined.strip(",.:;!?")
                    new_relation = f"{relation} {combined_clean}".strip() if combined_clean else relation
                else:
                    logging.debug(f"      [SINGLE-ENTITY] Extracted new relation: '{new_relation}'")
                
                new_object = matched_entity
        
        # Create repaired triple
        repaired_triple = {
            "subject": subj,
            "relation": new_relation,
            "object": new_object
        }
        
        logging.debug(f"      [SINGLE-ENTITY] ✓ Repaired triple:")
        logging.debug(f"        Subject: '{repaired_triple['subject']}'")
        logging.debug(f"        Relation: '{repaired_triple['relation']}'")
        logging.debug(f"        Object: '{repaired_triple['object']}'")
        
        return [repaired_triple]


    def _rewrite_to_triplet_form(
        self,
        subject: str,
        relation: str,
        obj: str,
        entity_text: str
    ) -> Optional[str]:
        """
        Use LLM to restructure object text so entity appears at the end,
        absorbing and rewriting surrounding context to preserve meaning.
        """

        input_sentence = f"{subject} {relation} {obj}"
        target_phrase = entity_text
        user_message = f"""

        You are an expert in rewording the sentences. You are given a sentence and a target phrase ( where the target phrase is present in the sentence ). You need to reword the sentence in a way that the target phrase appears only at the end of the sentence. The reworded sentence should be grammatically correct and should not change the meaning of the original sentence.

        Input Sentence: {input_sentence}
        Target Phrase: {target_phrase}

        Only return the reworded sentence. NO JSON. NO EXPLANATION. NO MARKDOWN. NO ANYTHING ELSE.
        """

        try:
            logging.debug(f"        [LLM-REWRITE] Calling LLM for semantic restructuring...")
            new_triplet = self._call_llm(
                system_message="You are an expert in rewording sentences.",
                user_message=user_message,
                temperature=0.6,
                max_tokens=3000,
                json_mode=False,
            )
            
            new_triplet = new_triplet.strip('"\'`*')
            
            # Remove any markdown or formatting
            new_triplet = new_triplet.replace('```', '').replace('**', '').strip()
            
            return new_triplet
            
        except Exception as e:
            logging.error(f"        [LLM-REWRITE] Error: {e}")
            return None

    def _extract_relation_from_triplet(
        self,
        triplet: str,
        subject: str,
        entity: str
    ) -> Optional[str]:
        """
        Extract the relation from a restructured triplet sentence.
        
        The triplet should be in the form: "Subject [relation] Entity"
        We remove the subject from the beginning and entity from the end to get the relation.
        
        Args:
            triplet: The full restructured triplet sentence
            subject: The subject to remove from the beginning
            entity: The entity to remove from the end
            
        Returns:
            The extracted relation, or None if extraction fails
        """
        triplet_clean = triplet.strip()
        
        # Prepare for case-insensitive matching
        triplet_lower = triplet_clean.lower()
        subject_lower = subject.lower()
        entity_lower = entity.lower()
        
        # Step 1: Remove subject from beginning
        if triplet_lower.startswith(subject_lower):
            relation_part = triplet_clean[len(subject):].strip()
        else:
            # Try to find subject anywhere at the start (might have slight variations)
            # Look for subject tokens at the beginning
            subject_tokens = subject_lower.split()
            triplet_tokens = triplet_clean.split()
            
            if len(triplet_tokens) >= len(subject_tokens):
                # Check if first N tokens match subject
                triplet_start_lower = " ".join(triplet_tokens[:len(subject_tokens)]).lower()
                if triplet_start_lower == subject_lower or self._fuzzy_match(triplet_start_lower, subject_lower):
                    relation_part = " ".join(triplet_tokens[len(subject_tokens):]).strip()
                else:
                    # Subject not found at start, use the whole triplet
                    relation_part = triplet_clean
            else:
                relation_part = triplet_clean
        
        # Step 2: Remove entity from end
        relation_part_lower = relation_part.lower()
        
        if relation_part_lower.endswith(entity_lower):
            relation = relation_part[:-len(entity)].strip()
        else:
            # Try to find entity at the end (might have slight variations)
            entity_tokens = entity_lower.split()
            relation_tokens = relation_part.split()
            
            if len(relation_tokens) >= len(entity_tokens):
                # Check if last N tokens match entity
                relation_end_lower = " ".join(relation_tokens[-len(entity_tokens):]).lower()
                if relation_end_lower == entity_lower or self._fuzzy_match(relation_end_lower, entity_lower):
                    relation = " ".join(relation_tokens[:-len(entity_tokens)]).strip()
                else:
                    # Entity not found at end, try rfind
                    idx = relation_part_lower.rfind(entity_lower)
                    if idx != -1:
                        relation = relation_part[:idx].strip()
                    else:
                        return None
            else:
                return None
        
        # Clean up any trailing/leading punctuation
        relation = relation.strip(".,;:!? ")
        
        # Validate: relation shouldn't be empty or too short
        if not relation or len(relation) < 2:
            return None
        
        return relation

    def _fuzzy_match(self, text1: str, text2: str, threshold: float = 0.85) -> bool:
        """
        Simple fuzzy matching based on character overlap.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            threshold: Minimum similarity ratio (0.0 to 1.0)
            
        Returns:
            True if texts are similar enough
        """
        if not text1 or not text2:
            return False
        
        # Simple character-level Jaccard similarity
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return False
        
        similarity = intersection / union
        return similarity >= threshold  
    
    def create_repair_prompt(
        self, 
        candidate: Dict[str, Any], 
        chunk_text: str, 
        entities_refined: List[str]
    ) -> Tuple[str, str]:
        """
        Create system and user prompts for LLM repair.
        
        Args:
            candidate: Candidate triple with matched entity
            chunk_text: Original chunk text for context
            entities_refined: List of all refined entities
            
        Returns:
            Tuple of (system_message, user_message)
        """
        triple = candidate["triple"]
        matched_entity = candidate["matched_entity"]
        
        system_message = """You are a knowledge graph post-processing assistant.

        Your task is to rewrite triples so that **no object string contains a refined entity as a substring**.

        Rules:
        1. Subjects and objects represent entities or phrases
        2. Objects must NOT embed entities from the refined list as substrings
        3. You may output one or multiple triples per input triple
        4. You must output a valid JSON array of triples with fields: "subject", "relation", "object"

        Restructuring Options:

        **Option A (intermediate node)**: Split the object into:
        - A new phrase as object of the original subject
        - A new triple from that phrase to the matched entity
        
        Example:
        Input: {"subject": "Butterfly", "relation": "takes", "object": "an incredible journey to reach adulthood"}
        Output: [
        {"subject": "Butterfly", "relation": "takes", "object": "an incredible journey"},
        {"subject": "an incredible journey", "relation": "leads_to", "object": "adulthood"}
        ]

        **Option B (merged relation)**: Move words around the matched entity into the relation:
        - Extend the relation with descriptive words
        - Set object = matched entity
        
        Example:
        Input: {"subject": "Butterfly", "relation": "takes", "object": "an incredible journey to reach adulthood"}
        Output: [
        {"subject": "Butterfly", "relation": "takes_an_incredible_journey_to_reach", "object": "adulthood"}
        ]

        Choose the option that best preserves the semantic meaning and creates a clear knowledge graph structure.

        IMPORTANT:
        - Return ONLY a JSON array of objects
        - Do not include any explanation text before or after the JSON
        - Ensure none of the resulting objects contain any refined entities as substrings
        - If restructuring is not needed (shouldn't happen for candidates), return the original triple"""
                
        # Create compact entity list (just names)
        entities_str = ", ".join(entities_refined)
        
        user_message = f"""Context (original chunk text):
        {chunk_text}

        Refined entities (do not use these as substrings in objects):
        {entities_str}

        Candidate triple to repair:
        {json.dumps(triple, indent=2)}

        Matched entity found in object: "{matched_entity}"

        Please restructure this triple so that "{matched_entity}" is not embedded in the object string. Return a JSON array of the restructured triple(s)."""
                
        return system_message, user_message
    
    def call_llm_for_repair(
        self, 
        system_message: str, 
        user_message: str
    ) -> Optional[List[Dict[str, str]]]:
        """
        Call LLM to repair a single triple.
        
        Args:
            system_message: System prompt
            user_message: User prompt with context and triple
            
        Returns:
            List of repaired triples or None on error
        """
        try:
            logging.debug(f"\n      [LLM] Calling {self.model}...")
            logging.debug(f"      [LLM] System message length: {len(system_message)} chars")
            logging.debug(f"      [LLM] User message length: {len(user_message)} chars")
            
            content = self._call_llm(
                system_message=system_message,
                user_message=user_message,
                temperature=0.3,
                max_tokens=3000
            )
            
            logging.debug(f"      [LLM] Raw response:\n{content}")
            
            # Try to extract JSON from the response
            # Sometimes the model wraps it in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
                logging.debug(f"      [LLM] Extracted JSON from markdown block")
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                logging.debug(f"      [LLM] Extracted content from code block")
            
            # Parse JSON
            repaired_triples = json.loads(content)
            logging.debug(f"      [LLM] Successfully parsed {len(repaired_triples) if isinstance(repaired_triples, list) else 1} triple(s)")
            
            # Validate structure
            if isinstance(repaired_triples, list):
                for i, triple in enumerate(repaired_triples, 1):
                    if not all(key in triple for key in ["subject", "relation", "object"]):
                        logging.error(f"      [LLM] Invalid triple structure: {triple}")
                        return None
                    logging.debug(f"      [LLM] Repaired triple {i}: {triple['subject']} -[{triple['relation']}]-> {triple['object']}")
                return repaired_triples
            elif isinstance(repaired_triples, dict):
                # Single triple returned as dict, wrap in list
                if all(key in repaired_triples for key in ["subject", "relation", "object"]):
                    logging.debug(f"      [LLM] Repaired triple: {repaired_triples['subject']} -[{repaired_triples['relation']}]-> {repaired_triples['object']}")
                    return [repaired_triples]
                else:
                    logging.error(f"      [LLM] Invalid triple structure: {repaired_triples}")
                    return None
            else:
                logging.error(f"      [LLM] Unexpected response format: {type(repaired_triples)}")
                return None
                
        except json.JSONDecodeError as e:
            logging.error(f"      [LLM] Failed to parse JSON: {e}")
            logging.error(f"      [LLM] Response content: {content}")
            return None
        except Exception as e:
            logging.error(f"      [LLM] Error: {e}")
            return None
    
    def repair_triples(
        self, 
        chunks: List['ChunkArtifacts'], 
        entities_refined: List[str]
    ) -> Tuple[List['ChunkArtifacts'], List[Dict[str, Any]]]:
        """
        Main function to repair triples in chunks.
        
        Args:
            chunks: List of ChunkArtifacts objects with triples_raw
            entities_refined: List of refined entity strings
            
        Returns:
            Tuple of (updated chunks with triples_clean populated, resolution_history)
            resolution_history contains before/after details for each resolved triple
        """
        # Lazy import to avoid circular dependency
        from pipeline import TripleRecord
        
        # Track resolution history for logging
        resolution_history: List[Dict[str, Any]] = []
        
        # Step 1: Prepare entities
        logging.debug("="*80)
        logging.debug("STEP 1: Preparing entities with normalization...")
        logging.debug("="*80)
        entities_prepared = self.prepare_entities(entities_refined)
        logging.debug(f"Prepared {len(entities_prepared)} entities")
        for ent in entities_prepared:
            logging.debug(f"  '{ent['original']}' -> '{ent['norm']}'")
        
        # Process each chunk
        for chunk in chunks:
            chunk_id = chunk.chunk_id
            chunk_text = chunk.text
            triples_raw = chunk.triples_raw
            
            logging.debug("\n" + "="*80)
            logging.debug(f"PROCESSING CHUNK {chunk_id}")
            logging.debug("="*80)
            logging.debug(f"Chunk text preview: {chunk_text[:100]}...")
            logging.debug(f"Total triples to process: {len(triples_raw)}")
            

            logging.debug("\n  Detecting candidate triples...")
            candidates, clean_triples = self.detect_candidate_triples(triples_raw, entities_prepared)
            
            logging.debug("\n" + "-"*80)
            logging.debug(f"  DETECTION SUMMARY:")
            logging.debug(f"    ✓ Clean triples (no repair needed): {len(clean_triples)}")
            logging.debug(f"    ❌ Candidate triples (need repair): {len(candidates)}")
            logging.debug("-"*80)
            
            all_clean = list(clean_triples)  
            
            for clean_triple in clean_triples:
                resolution_history.append({
                    "chunk_id": chunk_id,
                    "status": "clean",
                    "repair_method": None,
                    "matched_entities": [],
                    "before": {
                        "subject": clean_triple["subject"],
                        "relation": clean_triple["relation"],
                        "object": clean_triple["object"]
                    },
                    "after": [{
                        "subject": clean_triple["subject"],
                        "relation": clean_triple["relation"],
                        "object": clean_triple["object"]
                    }]
                })
            

            if candidates:
                logging.debug(f"\n  Repairing {len(candidates)} candidate triple(s)...")
            
            for i, candidate in enumerate(candidates, 1):
                triple = candidate["triple"]
                matched_entity = candidate["matched_entity"]
                all_matches = candidate.get("all_matches", [matched_entity])
                
                logging.debug("\n" + "  " + "-"*76)
                logging.debug(f"  REPAIRING CANDIDATE {i}/{len(candidates)}")
                logging.debug("  " + "-"*76)
                logging.debug(f"    Original Triple:")
                logging.debug(f"      Subject: '{triple['subject']}'")
                logging.debug(f"      Relation: '{triple['relation']}'")
                logging.debug(f"      Object: '{triple['object']}'")
                logging.debug(f"    Matched entities in object: {all_matches}")
                logging.debug(f"    Primary matched entity: '{matched_entity}'")
    
                original_triple = {
                    "subject": triple["subject"],
                    "relation": triple["relation"],
                    "object": triple["object"]
                }
                
                repair_method = "single_entity" if len(all_matches) == 1 else "multiple_entities"
                
                if len(all_matches) == 1:
                    logging.debug(f"    → Single entity detected, using PRECISE method")
                    repaired = self.process_single_entity(
                        triple, matched_entity, chunk_text
                    )
                    
                    if repaired is None:
                        logging.warning(f"    → Precise method failed, falling back to GENERIC method")
                        repair_method = "single_entity_fallback_llm"
                        logging.debug(f"\n    Creating repair prompt...")
                        system_msg, user_msg = self.create_repair_prompt(
                            candidate, chunk_text, entities_refined
                        )
                            
                        logging.debug(f"    Calling LLM for generic repair...")
                        repaired = self.call_llm_for_repair(system_msg, user_msg)
                else:
                    logging.debug(f"    → Multiple entities detected ({len(all_matches)}), using GENERIC method")
                    repaired = self.process_multiple_entities(triple, all_matches, chunk_text, entities_refined)
                
                if repaired:
                    logging.debug(f"\n    ✓ SUCCESS: Repaired into {len(repaired)} triple(s)")
                    for j, rep in enumerate(repaired, 1):
                        logging.debug(f"      Repaired #{j}:")
                        logging.debug(f"        Subject: '{rep['subject']}'")
                        logging.debug(f"        Relation: '{rep['relation']}'")
                        logging.debug(f"        Object: '{rep['object']}'")
                    
                    for rep_triple in repaired:
                        if "chunk_id" not in rep_triple:
                            rep_triple["chunk_id"] = chunk_id
                        if "evidence" not in rep_triple and "evidence" in triple:
                            rep_triple["evidence"] = triple.get("evidence", "")
                    all_clean.extend(repaired)
                    
                    resolution_history.append({
                        "chunk_id": chunk_id,
                        "status": "repaired",
                        "repair_method": repair_method,
                        "matched_entities": all_matches,
                        "before": original_triple,
                        "after": [
                            {
                                "subject": rep["subject"],
                                "relation": rep["relation"],
                                "object": rep["object"]
                            }
                            for rep in repaired
                        ]
                    })
                else:
                    logging.warning(f"    ⚠ FAILED: Repair failed, keeping original triple")
                    all_clean.append(triple)
                    
                    resolution_history.append({
                        "chunk_id": chunk_id,
                        "status": "failed",
                        "repair_method": repair_method,
                        "matched_entities": all_matches,
                        "before": original_triple,
                        "after": [original_triple]  # Kept original
                    })
            
            chunk.triples_clean = [
                TripleRecord(
                    subject=t["subject"],
                    relation=t["relation"],
                    object=t["object"],
                    evidence=t.get("evidence"),
                    chunk_id=t.get("chunk_id")
                )
                for t in all_clean
            ]
            logging.debug("\n" + "="*80)
            logging.debug(f"CHUNK {chunk_id} COMPLETE")
            logging.debug(f"  Input triples: {len(triples_raw)}")
            logging.debug(f"  Output triples: {len(all_clean)}")
            logging.debug(f"  Difference: {len(all_clean) - len(triples_raw):+d}")
            logging.debug("="*80)
        
        return chunks, resolution_history
