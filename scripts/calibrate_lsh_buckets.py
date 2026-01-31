"""
LSH Bucket Calibration Script.
Trains bucket->intent mappings from labeled training data.

Usage:
    python scripts/calibrate_lsh_buckets.py

This creates data/bucket_map.json which the router uses for Stage 2 LSH routing.
"""

import os
import sys
import json
from typing import List, Tuple
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sbscr.core.lsh import LSHBucketRouter
from sbscr.core.normalizer import SemanticNormalizer





def load_training_data_from_file(path: str) -> List[Tuple[str, str]]:
    """Load training data from JSONL file."""
    print(f"📂 Loading real-world data from {path}...")
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                if item.get("intent") != "unknown":
                    data.append((item["query"], item["intent"]))
            except Exception:
                continue
    print(f"   Loaded {len(data)} samples from file")
    return data


def calibrate():
    """Run calibration and save bucket mappings."""
    print("=" * 60)
    print("LSH Bucket Calibration")
    print("=" * 60)
    
    # Check for external data
    external_path = "data/training_data.jsonl"
    if os.path.exists(external_path):
        training_data = load_training_data_from_file(external_path)
    else:
        print("\n❌ Error: Real-world training data not found!")
        print(f"   Missing file: {external_path}")
        print("   Please run 'python scripts/prepare_training_data.py' first.")
        print("   Synthetic data generation is disabled.")
        return
        
    # Process all data
    print("🎯 Calibrating LSH buckets...")
    bucket_counts = defaultdict(lambda: defaultdict(float))
    
    # Weight configuration tuned for dataset "strength":
    # Attempt 8: Final Polish. 
    # Target: Creative ~25%, Math ~30%, Coding ~30%.
    INTENT_WEIGHTS = {
        "coding": 4.5,
        "math": 3.8,
        "creative": 3.3,
        "reasoning": 5.0,
        "general": 1.0     # Baseline
    }
    
    # Initialize normalizer and router
    print("\n🔧 Loading semantic normalizer and LSH generator...")
    normalizer = SemanticNormalizer()
    num_buckets = 100 
    # Valid LSHBucketRouter instance to access its generator
    router_instance = LSHBucketRouter(num_buckets=num_buckets, bucket_map_path="data/bucket_map.json")
    
    print("\n🔄 Normalizing queries and assigning to buckets...")
    for query, intent in training_data:
        # Normalize query first
        normalized_query = normalizer.normalize(query)
        
        # Get bucket using FAST hash (accessing the internal generator)
        bucket_id = router_instance.generator.get_bucket_id_fast(normalized_query, num_buckets)
        
        # Add weighted count
        weight = INTENT_WEIGHTS.get(intent, 1.0)
        bucket_counts[bucket_id][intent] += weight
    
    # Determine dominant intent for each bucket
    bucket_map = {}
    calibrated_count = 0
    intent_stats = defaultdict(int)
    
    for bucket_id, scores in bucket_counts.items():
        # Find intent with highest weighted score
        top_intent = max(scores, key=scores.get)
        
        # Calculate consistency (confidence) based on scores
        total_score = sum(scores.values())
        confidence = scores[top_intent] / total_score if total_score > 0 else 0.0
        
        # Only map if reasonable confidence or it's a specialized intent
        if confidence > 0.15:  # Loose threshold (lowered from 0.2 for better recall)
            bucket_map[str(bucket_id)] = {
                "intent": top_intent,
                "confidence": min(0.95, float(confidence)) # Cap at 0.95
            }
            calibrated_count += 1
            intent_stats[top_intent] += 1
            
    # Save the map
    output_path = "data/bucket_map.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(bucket_map, f, indent=2)
        
    print(f"✅ Loaded {len(bucket_map)} bucket mappings from {output_path}")
    print(f"✅ Calibrated {calibrated_count} buckets from {len(training_data)} samples")
    print(f"✅ Saved bucket map to {output_path}")
    
    # Stats
    print("\n📈 Calibration Statistics:")
    print(f"   Total buckets: {num_buckets}")
    print(f"   Calibrated buckets: {len(bucket_map)}")
    avg_conf = sum(m["confidence"] for m in bucket_map.values()) / len(bucket_map) if bucket_map else 0
    print(f"   Average confidence: {avg_conf:.1%}")
    print(f"   Intent distribution: {dict(intent_stats)}")
    
    # Reload router with NEW map for testing
    print("\n🔄 Reloading router for verification...")
    router = LSHBucketRouter(num_buckets=num_buckets, bucket_map_path="data/bucket_map.json")

    # Test consistency
    print("\n🧪 Testing synonym consistency...")
    test_pairs = [
        ("Write a binary search", "Implement a halving interval search"),
        ("Create a REST API", "Build an API endpoint"),
        ("Write a poem about love", "Compose poetry about romance"),
    ]
    
    for q1, q2 in test_pairs:
        n1 = normalizer.normalize(q1)
        n2 = normalizer.normalize(q2)
        i1, c1 = router.route(q1, n1)
        i2, c2 = router.route(q2, n2)
        
        match = "✅" if i1 == i2 else "❌"
        print(f"\n   {match} '{q1[:30]}...'")
        print(f"      vs '{q2[:30]}...'")
        print(f"      Intent 1: {i1} ({c1:.0%})")
        print(f"      Intent 2: {i2} ({c2:.0%})")
    
    print("\n✅ Calibration complete!")
    print(f"   Bucket map saved to: data/bucket_map.json")


if __name__ == "__main__":
    calibrate()
