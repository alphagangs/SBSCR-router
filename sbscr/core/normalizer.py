"""
Semantic Normalizer for SBSCR.
Fixes LSH brittleness by normalizing synonymous terms before hashing.

Example:
  "Write a binary search" -> "write ALGO:BINARY_SEARCH"
  "Implement halving interval search" -> "implement ALGO:BINARY_SEARCH"
  
Both now produce similar LSH signatures.
"""

import os
import re
import yaml
from typing import Dict, List, Set, Tuple

class SemanticNormalizer:
    """
    Lightweight semantic normalization for LSH stability.
    Replaces synonymous phrases with canonical concept tokens.
    """
    
    def __init__(self, synonyms_path: str = "data/synonyms.yaml"):
        """
        Initialize normalizer with synonym dictionary.
        
        Args:
            synonyms_path: Path to YAML file with synonym mappings
        """
        self.synonyms: Dict[str, Dict[str, List[str]]] = {}
        self.fast_path_markers: Dict[str, List[str]] = {}
        self.reverse_map: Dict[str, str] = {}  # phrase -> canonical
        
        self._load_synonyms(synonyms_path)
        self._build_reverse_map()
        
    def _load_synonyms(self, path: str):
        """Load synonym dictionary from YAML."""
        if not os.path.exists(path):
            print(f"⚠️ Synonyms file not found at {path}. Using empty dictionary.")
            return
            
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        
        # Extract fast path markers separately
        self.fast_path_markers = data.pop('fast_path_markers', {})
        
        # Rest is domain -> concept -> synonyms
        self.synonyms = data
        
    def _build_reverse_map(self):
        """Build reverse lookup: phrase -> DOMAIN:CONCEPT."""
        for domain, concepts in self.synonyms.items():
            for concept, phrases in concepts.items():
                canonical = f"{domain.upper()}:{concept.upper()}"
                for phrase in phrases:
                    # Store lowercase for matching
                    self.reverse_map[phrase.lower()] = canonical
    
    def normalize(self, query: str) -> str:
        """
        Normalize query by replacing synonymous phrases with canonical tokens.
        
        Args:
            query: Raw input query
            
        Returns:
            Normalized query with canonical concept tokens
        """
        result = query.lower()
        
        # Sort by phrase length (longest first) to handle overlapping matches
        sorted_phrases = sorted(self.reverse_map.keys(), key=len, reverse=True)
        
        for phrase in sorted_phrases:
            if phrase in result:
                canonical = self.reverse_map[phrase]
                result = result.replace(phrase, canonical)
        
        return result
    
    def detect_intent_fast(self, query: str) -> Tuple[str, float]:
        """
        Fast path intent detection using keyword markers.
        Returns (intent, confidence) or ("unknown", 0.0) if no match.
        
        This is Stage 1 of the routing pipeline - pure keyword matching.
        Runs in ~0.1ms (no ML inference).
        """
        query_lower = query.lower()
        
        best_intent = "unknown"
        best_score = 0.0
        
        for intent, markers in self.fast_path_markers.items():
            matches = sum(1 for m in markers if m in query_lower)
            if matches > 0:
                # Score based on number of matches
                score = min(matches / 2.0, 1.0)  # Cap at 1.0
                if score > best_score:
                    best_score = score
                    best_intent = intent
        
        return best_intent, best_score
    
    def extract_keywords(self, query: str) -> Set[str]:
        """
        Extract significant keywords from query for LSH.
        Filters out stopwords and short words.
        """
        # Simple stopword list
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
            'because', 'until', 'while', 'that', 'this', 'these', 'those',
            'what', 'which', 'who', 'whom', 'it', 'its', 'i', 'me', 'my',
            'you', 'your', 'he', 'him', 'his', 'she', 'her', 'we', 'us',
            'our', 'they', 'them', 'their', 'write', 'create', 'make'
        }
        
        # Tokenize
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query.lower())
        
        # Filter
        keywords = {w for w in words if w not in stopwords and len(w) > 2}
        
        return keywords


class TrivialQueryDetector:
    """
    Stage 0: Ultra-fast detection of trivial queries.
    Routes to CHEAP_CHAT immediately without further processing.
    """
    
    # Common greetings and trivial patterns
    TRIVIAL_PATTERNS = [
        r'^hi\b', r'^hello\b', r'^hey\b', r'^greet',
        r'^what is your name', r'^who are you',
        r'^thank', r'^thanks', r'^ok\b', r'^okay\b',
        r'^yes\b', r'^no\b', r'^bye\b', r'^goodbye',
    ]
    
    # Simple arithmetic pattern
    ARITHMETIC_PATTERN = r'^(what is |calculate )?[\d\s\+\-\*\/\(\)\.]+$'
    
    def __init__(self):
        self.trivial_re = [re.compile(p, re.IGNORECASE) for p in self.TRIVIAL_PATTERNS]
        self.arithmetic_re = re.compile(self.ARITHMETIC_PATTERN, re.IGNORECASE)
    
    def is_trivial(self, query: str) -> bool:
        """
        Check if query is trivial and should skip to CHEAP_CHAT.
        
        Criteria:
        1. Very short queries (< 15 chars)
        2. Common greetings
        3. Simple arithmetic
        4. Single-word queries
        """
        query = query.strip()
        
        # Length check
        if len(query) < 15:
            return True
        
        # Single word
        if len(query.split()) <= 2:
            return True
        
        # Greeting patterns
        for pattern in self.trivial_re:
            if pattern.search(query):
                return True
        
        # Arithmetic
        if self.arithmetic_re.match(query):
            return True
        
        return False


if __name__ == "__main__":
    # Test the normalizer
    normalizer = SemanticNormalizer()
    detector = TrivialQueryDetector()
    
    test_queries = [
        "Write a binary search in Python",
        "Implement a halving interval search algorithm",
        "Solve for x: 2x + 5 = 10",
        "Write a poem about autumn",
        "hello",
        "2 + 2",
        "Implement a distributed consensus algorithm",
    ]
    
    print("\n--- Semantic Normalizer Test ---")
    for q in test_queries:
        is_trivial = detector.is_trivial(q)
        normalized = normalizer.normalize(q)
        intent, conf = normalizer.detect_intent_fast(q)
        
        print(f"\nQuery: '{q}'")
        print(f"  Trivial: {is_trivial}")
        print(f"  Fast Path: {intent} ({conf:.0%})")
        print(f"  Normalized: '{normalized}'")
