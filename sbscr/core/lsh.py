"""
LSH (Locality-Sensitive Hashing) signature generator for semantic bucketing.
Uses MinHash for efficient similarity estimation.

SBSCR v6: Now includes bucket-to-intent routing for sub-millisecond semantic classification.
"""

from typing import List, Set, Dict, Tuple, Optional
import mmh3
from datasketch import MinHash
import re
import json
import os


class LSHSignatureGenerator:
    """Generate LSH signatures from text queries for fast semantic bucketing."""
    
    def __init__(self, num_perm: int = 64, ngram_size: int = 3, seed: int = 42):
        """
        Initialize LSH signature generator.
        
        Args:
            num_perm: Number of hash permutations (higher = more accurate, slower)
                      Increased from 16 to 64 for better stability
            ngram_size: Character n-gram size for hashing
            seed: Random seed for reproducibility
        """
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.seed = seed
        
    def _preprocess(self, text: str) -> str:
        """Normalize and clean text."""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Strip
        text = text.strip()
        return text
    
    def _generate_ngrams(self, text: str) -> Set[str]:
        """Generate character n-grams from text."""
        text = self._preprocess(text)
        ngrams = set()
        
        # Character n-grams
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.add(text[i:i + self.ngram_size])
        
        # Also add word tokens for better semantic representation
        words = text.split()
        for word in words:
            if len(word) > 2:  # Skip very short words
                ngrams.add(f"_WORD_{word}")
        
        return ngrams
    
    def generate_signature(self, query: str) -> MinHash:
        """
        Generate MinHash signature for a query.
        
        Args:
            query: Input text query
            
        Returns:
            MinHash signature object
        """
        minhash = MinHash(num_perm=self.num_perm, seed=self.seed)
        ngrams = self._generate_ngrams(query)
        
        for ngram in ngrams:
            minhash.update(ngram.encode('utf-8'))
        
        return minhash
    
    def estimate_similarity(self, sig1: MinHash, sig2: MinHash) -> float:
        """
        Estimate Jaccard similarity between two signatures.
        
        Args:
            sig1: First MinHash signature
            sig2: Second MinHash signature
            
        Returns:
            Similarity score between 0 and 1
        """
        return sig1.jaccard(sig2)
    
    def get_signature_hash(self, query: str) -> int:
        """
        Get a compact hash representation of the query signature.
        Useful for bucketing queries into discrete categories.
        
        Args:
            query: Input text query
            
        Returns:
            Integer hash value
        """
        minhash = self.generate_signature(query)
        # Convert first few hash values to a single integer
        hashvalues = minhash.hashvalues[:8]  # Use first 8 for bucketing
        combined = ''.join(str(h) for h in hashvalues)
        return mmh3.hash(combined, seed=self.seed)
    
    def get_bucket_id(self, query: str, num_buckets: int = 100) -> int:
        """
        Assign query to a semantic bucket.
        
        Args:
            query: Input text query
            num_buckets: Number of buckets to use
            
        Returns:
            Bucket ID (0 to num_buckets-1)
        """
        hash_val = self.get_signature_hash(query)
        return abs(hash_val) % num_buckets
    
    def get_bucket_id_fast(self, query: str, num_buckets: int = 100) -> int:
        """
        FAST bucket assignment using direct hash (no MinHash).
        Trades accuracy for speed - target latency: <0.1ms.
        
        Uses sorted keywords for some position-invariance.
        
        Args:
            query: Input text query
            num_buckets: Number of buckets to use
            
        Returns:
            Bucket ID (0 to num_buckets-1)
        """
        # Preprocess
        text = self._preprocess(query)
        
        # Extract and sort keywords for position-invariance
        words = sorted(set(text.split()))
        
        # Create a canonical representation
        canonical = ' '.join(words)
        
        # Direct hash
        hash_val = mmh3.hash(canonical, seed=self.seed)
        return abs(hash_val) % num_buckets
    
    def get_signature_vector(self, query: str) -> List[int]:
        """
        Get the raw hash values as a vector for ML features.
        
        Args:
            query: Input text query
            
        Returns:
            List of hash values (length = num_perm)
        """
        minhash = self.generate_signature(query)
        return list(minhash.hashvalues)


class LSHBucketRouter:
    """
    LSH-based semantic router.
    Maps queries to intent buckets without neural network inference.
    
    This is Stage 2 of the SBSCR v6 pipeline.
    Target latency: ~0.3ms
    """
    
    # Default bucket mappings (used when no calibration data exists)
    # These are based on common query patterns
    DEFAULT_BUCKET_BIASES = {
        'coding': 0.3,    # 30% of queries tend to be code
        'math': 0.15,
        'creative': 0.15,
        'reasoning': 0.15,
        'general': 0.25,
    }
    
    def __init__(self, 
                 num_buckets: int = 100,
                 bucket_map_path: str = "data/bucket_map.json",
                 num_perm: int = 64):
        """
        Initialize bucket router.
        
        Args:
            num_buckets: Number of semantic buckets
            bucket_map_path: Path to calibrated bucket->intent mappings
            num_perm: Number of hash permutations for LSH
        """
        self.num_buckets = num_buckets
        self.bucket_map_path = bucket_map_path
        self.generator = LSHSignatureGenerator(num_perm=num_perm)
        
        # Bucket -> intent mapping
        # Format: {bucket_id: {"intent": str, "confidence": float}}
        self.bucket_map: Dict[int, Dict] = {}
        
        self._load_bucket_map()
        
    def _load_bucket_map(self):
        """Load pre-calibrated bucket mappings."""
        if os.path.exists(self.bucket_map_path):
            with open(self.bucket_map_path, 'r') as f:
                data = json.load(f)
                # Convert string keys back to int
                self.bucket_map = {int(k): v for k, v in data.items()}
            print(f"✅ Loaded {len(self.bucket_map)} bucket mappings from {self.bucket_map_path}")
        else:
            print(f"⚠️ No bucket map found at {self.bucket_map_path}. Using keyword fallback.")
    
    def save_bucket_map(self, path: Optional[str] = None):
        """Save current bucket mappings to file."""
        path = path or self.bucket_map_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.bucket_map, f, indent=2)
        print(f"✅ Saved bucket map to {path}")
    
    def calibrate(self, training_data: List[Tuple[str, str]]):
        """
        Calibrate bucket mappings from labeled training data.
        
        Args:
            training_data: List of (query, intent) tuples
        """
        from collections import defaultdict
        
        # Count intent occurrences per bucket
        bucket_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for query, intent in training_data:
            bucket_id = self.generator.get_bucket_id_fast(query, self.num_buckets)
            bucket_counts[bucket_id][intent] += 1
        
        # Assign each bucket to its majority intent
        for bucket_id, intent_counts in bucket_counts.items():
            total = sum(intent_counts.values())
            best_intent = max(intent_counts, key=intent_counts.get)
            confidence = intent_counts[best_intent] / total
            
            self.bucket_map[bucket_id] = {
                "intent": best_intent,
                "confidence": confidence,
                "samples": total
            }
        
        print(f"✅ Calibrated {len(self.bucket_map)} buckets from {len(training_data)} samples")
    
    def route(self, query: str, normalized_query: Optional[str] = None) -> Tuple[str, float]:
        """
        Route query to intent using LSH buckets.
        
        Args:
            query: Raw input query
            normalized_query: Optional pre-normalized query (from SemanticNormalizer)
            
        Returns:
            Tuple of (intent, confidence)
        """
        # Use normalized query for bucketing if provided
        bucket_query = normalized_query if normalized_query else query
        
        bucket_id = self.generator.get_bucket_id_fast(bucket_query, self.num_buckets)
        
        if bucket_id in self.bucket_map:
            mapping = self.bucket_map[bucket_id]
            return mapping["intent"], mapping["confidence"]
        else:
            # Uncalibrated bucket - return low confidence
            return "general", 0.3
    
    def get_bucket_stats(self) -> Dict:
        """Get statistics about bucket distribution."""
        from collections import Counter
        
        intent_counts = Counter(m["intent"] for m in self.bucket_map.values())
        avg_confidence = sum(m["confidence"] for m in self.bucket_map.values()) / max(len(self.bucket_map), 1)
        
        return {
            "total_buckets": self.num_buckets,
            "calibrated_buckets": len(self.bucket_map),
            "intent_distribution": dict(intent_counts),
            "average_confidence": round(avg_confidence, 3)
        }


class LSHIndex:
    """
    LSH index for fast nearest neighbor search.
    Maps queries to semantic buckets.
    """
    
    def __init__(self, num_perm: int = 64, threshold: float = 0.5):
        """
        Initialize LSH index.
        
        Args:
            num_perm: Number of hash permutations
            threshold: Similarity threshold for matching
        """
        from datasketch import MinHashLSH
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.generator = LSHSignatureGenerator(num_perm=num_perm)
        self.signatures = {}
        
    def insert(self, key: str, query: str):
        """
        Insert a query into the index.
        
        Args:
            key: Unique identifier for this query
            query: Query text
        """
        signature = self.generator.generate_signature(query)
        self.lsh.insert(key, signature)
        self.signatures[key] = signature
        
    def query(self, query: str) -> List[str]:
        """
        Find similar queries in the index.
        
        Args:
            query: Query text to search for
            
        Returns:
            List of matching keys
        """
        signature = self.generator.generate_signature(query)
        return self.lsh.query(signature)


if __name__ == "__main__":
    # Test the LSH bucket router
    generator = LSHSignatureGenerator(num_perm=64)
    router = LSHBucketRouter(num_buckets=100)
    
    test_queries = [
        "Write a binary search in Python",
        "Implement a halving interval search algorithm",
        "Solve the equation 2x + 5 = 10",
        "Write a poem about the ocean",
    ]
    
    print("\n--- LSH Bucket Router Test ---")
    for q in test_queries:
        bucket_id = generator.get_bucket_id(q, 100)
        intent, conf = router.route(q)
        print(f"\nQuery: '{q}'")
        print(f"  Bucket: {bucket_id}")
        print(f"  Intent: {intent} ({conf:.0%})")

