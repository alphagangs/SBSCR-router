"""
SBSCR v6 - Tiered LLM Router (Sub-Millisecond Design)

This router implements the original SBSCR vision:
- No neural network passes (removed DistilBART dependency)
- LSH as primary semantic signal
- Sub-millisecond routing latency (<1ms target)

Pipeline Stages:
  Stage 0: Trivial Filter      (~0.01ms) - Very short/simple queries
  Stage 1: Keyword Fast Path   (~0.1ms)  - Domain marker detection
  Stage 2: LSH Bucket Routing  (~0.3ms)  - Semantic bucketing
  Stage 3: XGBoost Complexity  (~0.2ms)  - Tier selection within intent
"""

import numpy as np
import xgboost as xgb
import os
import time
from typing import Dict, Any, Optional, List, Tuple

from sbscr.core.metadata import ComplexityExtractor
from sbscr.core.lsh import LSHSignatureGenerator, LSHBucketRouter
from sbscr.core.registry import ModelRegistry, ModelCluster
from sbscr.core.normalizer import SemanticNormalizer, TrivialQueryDetector


class SBSCRRouterV6:
    """
    SBSCR v6 - Tiered Sub-Millisecond Router.
    
    Designed to match the original SBSCR vision:
    - No heavy neural network passes
    - LSH-based semantic bucketing
    - Target: <1ms routing latency
    """
    
    def __init__(self, 
                 registry_path: str = "data/models.yaml",
                 model_path: str = "sbscr/models/complexity_xgboost.json",
                 synonyms_path: str = "data/synonyms.yaml",
                 bucket_map_path: str = "data/bucket_map.json"):
        
        print("🚀 Initializing SBSCR v6 (Sub-Millisecond Tiered Router)...")
        start = time.time()
        
        # 1. Load Registry
        self.registry = ModelRegistry(registry_path)
        print(f"📚 Loaded {len(self.registry.models)} models from registry.")
        
        # 2. Load XGBoost Model (Stage 3: Complexity)
        self.model = xgb.XGBRegressor()
        if os.path.exists(model_path):
            self.model.load_model(model_path)
            print("🧠 Loaded XGBoost Complexity Scorer.")
            self.has_model = True
        else:
            print(f"⚠️ Model not found at {model_path}. Using heuristic complexity.")
            self.has_model = False
            
        # 3. Feature Extractors
        self.extractor = ComplexityExtractor()
        self.lsh = LSHSignatureGenerator(num_perm=64)
        
        # 4. NEW: Semantic Normalizer (Fixes LSH brittleness)
        self.normalizer = SemanticNormalizer(synonyms_path)
        
        # 5. NEW: Trivial Query Detector (Stage 0)
        self.trivial_detector = TrivialQueryDetector()
        
        # 6. NEW: LSH Bucket Router (Stage 2 - Replaces DistilBART!)
        self.bucket_router = LSHBucketRouter(
            num_buckets=100,
            bucket_map_path=bucket_map_path,
            num_perm=64
        )
        
        print(f"✅ Router initialized in {time.time() - start:.3f}s")
        
    def route(self, query: str) -> str:
        """
        Route query to optimal model name.
        Returns the single best model (Top 1).
        """
        candidates = self.route_with_fallbacks(query)
        return candidates[0] if candidates else self.registry.get_best_model(ModelCluster.CHEAP_CHAT)

    def route_with_fallbacks(self, query: str) -> List[str]:
        """
        Route query and return a prioritized list of candidates.
        
        4-Stage Tiered Pipeline:
          Stage 0: Trivial Filter      -> CHEAP_CHAT (skip all ML)
          Stage 1: Keyword Fast Path   -> Intent via markers
          Stage 2: LSH Bucket Routing  -> Intent via semantic hash
          Stage 3: XGBoost Complexity  -> Tier within intent
        """
        
        # ========== STAGE 0: Trivial Filter (~0.01ms) ==========
        if not query or len(query.strip()) == 0:
            return [self.registry.get_best_model(ModelCluster.CHEAP_CHAT)]
        
        if self.trivial_detector.is_trivial(query):
            # Ultra-short/simple queries -> cheap model immediately
            return self._get_fallback_chain(ModelCluster.CHEAP_CHAT, "phi-3-mini", 0)
        
        # ========== STAGE 1: Keyword Fast Path (~0.1ms) ==========
        intent, fast_conf = self.normalizer.detect_intent_fast(query)
        
        # If high confidence from keywords, skip LSH
        if fast_conf >= 0.5:
            # Use keyword-detected intent directly
            pass  # intent already set
        else:
            # ========== STAGE 2: LSH Bucket Routing (~0.3ms) ==========
            # Normalize query to fix synonym brittleness
            normalized = self.normalizer.normalize(query)
            
            # Route via LSH buckets
            lsh_intent, lsh_conf = self.bucket_router.route(query, normalized)
            
            # Use LSH result if confidence is reasonable
            if lsh_conf > 0.4:
                intent = lsh_intent
            elif intent == "unknown":
                intent = lsh_intent  # Fallback to LSH even with low conf
        
        # ========== STAGE 3: Heuristic Complexity (~0.1ms) ==========
        # Use fast heuristic instead of XGBoost to achieve sub-ms latency
        # XGBoost + MinHash signature add ~4ms - unacceptable for sub-ms target
        score = self.extractor.estimate_complexity(query) / 10.0
        
        # ========== MODEL SELECTION ==========
        # Calculate context requirement
        query_tokens = len(query) // 4
        min_context = query_tokens + 512
        
        target_cluster = ModelCluster.CHEAP_CHAT
        target_fallback = "phi-3-mini"
        
        # A. Creative/Reasoning
        if intent in ['creative', 'reasoning']:
            if score > 0.15:
                target_cluster = ModelCluster.SOTA
                target_fallback = "gpt-4-turbo"
            else:
                target_cluster = ModelCluster.HIGH_PERF
                target_fallback = "llama-3-70b"
        
        # B. Coding
        elif intent == 'coding':
            if score > 0.75:
                target_cluster = ModelCluster.SOTA
                target_fallback = "gpt-4-turbo"
            elif score > 0.3:
                target_cluster = ModelCluster.FAST_CODE
                target_fallback = "deepseek-coder-v2"
            else:
                target_cluster = ModelCluster.CHEAP_CHAT
                target_fallback = "phi-3-mini"
        
        # C. Math
        elif intent == 'math':
            if score > 0.15:
                target_cluster = ModelCluster.SOTA
                target_fallback = "gpt-4-turbo"
            else:
                target_cluster = ModelCluster.HIGH_PERF
                target_fallback = "llama-3-70b"
        
        # D. General/Unknown
        else:
            if score > 0.85:
                target_cluster = ModelCluster.SOTA
                target_fallback = "gpt-4-turbo"
            elif score > 0.6:
                target_cluster = ModelCluster.HIGH_PERF
                target_fallback = "llama-3-70b"
            else:
                target_cluster = ModelCluster.CHEAP_CHAT
                target_fallback = "phi-3-mini"
        
        return self._get_fallback_chain(target_cluster, target_fallback, min_context)
    
    def route_with_debug(self, query: str) -> Dict[str, Any]:
        """
        Route with full debug information for analysis.
        Returns dict with routing decision and all intermediate values.
        """
        start = time.perf_counter()
        
        result = {
            "query": query,
            "stages": {},
        }
        
        # Stage 0
        t0 = time.perf_counter()
        is_trivial = self.trivial_detector.is_trivial(query)
        result["stages"]["stage0_trivial"] = {
            "is_trivial": is_trivial,
            "latency_ms": (time.perf_counter() - t0) * 1000
        }
        
        if is_trivial:
            result["final_model"] = self.registry.get_best_model(ModelCluster.CHEAP_CHAT)
            result["total_latency_ms"] = (time.perf_counter() - start) * 1000
            return result
        
        # Stage 1
        t1 = time.perf_counter()
        fast_intent, fast_conf = self.normalizer.detect_intent_fast(query)
        result["stages"]["stage1_fast_path"] = {
            "intent": fast_intent,
            "confidence": fast_conf,
            "latency_ms": (time.perf_counter() - t1) * 1000
        }
        
        # Stage 2
        t2 = time.perf_counter()
        normalized = self.normalizer.normalize(query)
        lsh_intent, lsh_conf = self.bucket_router.route(query, normalized)
        bucket_id = self.lsh.get_bucket_id(normalized, 100)
        result["stages"]["stage2_lsh"] = {
            "normalized_query": normalized,
            "bucket_id": bucket_id,
            "intent": lsh_intent,
            "confidence": lsh_conf,
            "latency_ms": (time.perf_counter() - t2) * 1000
        }
        
        # Stage 3
        t3 = time.perf_counter()
        features = self.extractor.extract_features(query)
        score = self.extractor.estimate_complexity(query) / 10.0
        result["stages"]["stage3_complexity"] = {
            "score": score,
            "key_features": {
                "word_count": features.get('word_count'),
                "is_code": features.get('is_code_related'),
                "domain": features.get('domain'),
            },
            "latency_ms": (time.perf_counter() - t3) * 1000
        }
        
        # Final decision
        final_intent = fast_intent if fast_conf >= 0.5 else lsh_intent
        candidates = self.route_with_fallbacks(query)
        
        result["final_intent"] = final_intent
        result["final_model"] = candidates[0] if candidates else "unknown"
        result["fallback_chain"] = candidates
        result["total_latency_ms"] = (time.perf_counter() - start) * 1000
        
        return result

    def _get_fallback_chain(self, cluster: ModelCluster, fallback: str, min_ctx: int = 0) -> List[str]:
        """Generate a robust list of candidate models."""
        # 1. Primary Candidates (Same Cluster)
        candidates = self.registry.get_candidates(cluster, min_context=min_ctx)
        
        # 2. Constraint Failure Upgrade
        if not candidates and min_ctx > 4000:
            candidates = self.registry.get_candidates(ModelCluster.SOTA, min_context=min_ctx)
        
        # 3. Safety Net
        safety_net = "gpt-4-turbo"
        
        final_list = []
        if candidates:
            final_list.extend(candidates)
        else:
            final_list.append(fallback)
            
        if safety_net not in final_list:
            final_list.append(safety_net)
            
        if fallback not in final_list:
            final_list.insert(1, fallback)
              
        return final_list[:3]


# Backward compatibility alias
SBSCRRouter = SBSCRRouterV6


if __name__ == "__main__":
    # Latency benchmark
    print("\n" + "="*60)
    print("SBSCR v6 - Sub-Millisecond Router Test")
    print("="*60)
    
    router = SBSCRRouterV6()
    
    test_queries = [
        ("hello", "trivial"),
        ("2 + 2", "trivial"),
        ("Write a Python function to parse JSON.", "code"),
        ("Implement a halving interval search algorithm in Python", "code"),
        ("Write a sonnet about the singularity.", "creative"),
        ("Calculate the integral of x^2.", "math"),
        ("What is the capital of France?", "general"),
        ("Design a distributed consensus algorithm", "complex"),
    ]
    
    print("\n--- Routing Tests with Latency ---\n")
    
    total_latency = 0
    for query, expected in test_queries:
        debug = router.route_with_debug(query)
        
        latency = debug["total_latency_ms"]
        total_latency += latency
        status = "✅" if latency < 1.0 else "⚠️ SLOW"
        
        print(f"{status} [{latency:.3f}ms] {expected:8} -> {debug['final_model']}")
        print(f"   Query: \"{query[:50]}...\"" if len(query) > 50 else f"   Query: \"{query}\"")
        print()
    
    print(f"\n📊 Average Latency: {total_latency / len(test_queries):.3f}ms")
    print(f"   Target: <1.0ms")
