import numpy as np
import xgboost as xgb
import os
from typing import Dict, Any, Optional

from sbscr.core.metadata import ComplexityExtractor
from sbscr.core.lsh import LSHSignatureGenerator
from sbscr.core.registry import ModelRegistry, ModelCluster
from sbscr.core.intent import IntentClassifier

class SBSCRRouter:
    """
    Enterprise Structural Router (v5).
    Uses XGBoost complexity scoring + Dynamic Registry for model selection.
    Now with Semantic Intent Layer (Zero-Shot).
    """
    def __init__(self, 
                 registry_path: str = "data/models.yaml",
                 model_path: str = "sbscr/models/complexity_xgboost.json"):
        
        print("🚀 Initializing Enterprise Router v5 (Hybrid Semantic)...")
        
        # 1. Load Registry
        self.registry = ModelRegistry(registry_path)
        print(f"📚 Loaded {len(self.registry.models)} models from registry.")
        
        # 2. Load XGBoost Model
        self.model = xgb.XGBRegressor()
        if os.path.exists(model_path):
            self.model.load_model(model_path)
            print("🧠 Loaded XGBoost Complexity Scorer.")
            self.has_model = True
        else:
            print(f"⚠️ Model not found at {model_path}. Router will strictly fallback to heuristics.")
            self.has_model = False
            
        # 3. Feature Extractors
        self.extractor = ComplexityExtractor()
        self.lsh = LSHSignatureGenerator(num_perm=16)
        
        # 4. Intent Layer (New)
        self.intent_classifier = IntentClassifier()
        
    def route(self, query: str) -> str:
        """
        Route query to optimal model name.
        Returns the single best model (Top 1).
        Wrapper around route_with_fallbacks.
        """
        candidates = self.route_with_fallbacks(query)
        return candidates[0] if candidates else self.registry.get_best_model(ModelCluster.CHEAP_CHAT)

    def route_with_fallbacks(self, query: str) -> list[str]:
        """
        Route query and return a prioritized list of candidates for retry/fallback.
        Order: [Primary Model, Cluster Backup, SOTA Safety Net]
        """
        
        # 1. Safety & Fast Path (optional, keep simple for now)
        if not query or len(query.strip()) == 0:
            return [self.registry.get_best_model(ModelCluster.CHEAP_CHAT)]

        # 2. Semantic Intent Classification (Cascade Optimization)
        # Fast Path: Check for obvious coding cues to save ~25ms
        intent = "general"
        intent_conf = 0.0
        
        if any(w in query for w in ["def ", "class ", "import ", "from ", "return", " -> ", "print(", "```", "{", "}", "    "]):
            intent = "coding"
            intent_conf = 1.0
        else:
            # Slow Path: Zero-Shot Transformer
            intent, intent_conf = self.intent_classifier.classify(query)
            
        # 3. Feature Extraction (Must match training exactly)
        features = self.extractor.extract_features(query)
        
        sig = self.lsh.generate_signature(query)
        if hasattr(sig, 'hashvalues'):
            sig_vals = sig.hashvalues
        elif isinstance(sig, (list, np.ndarray)):
            sig_vals = sig
        else:
            sig_vals = []
        sig_mean = np.mean(sig_vals) if len(sig_vals) > 0 else 0.0
        
        feat_vec = [
            features.get('word_count', 0),
            features.get('unique_token_ratio', 0),
            features.get('avg_word_length', 0),
            features.get('max_line_length', 0),
            features.get('code_density', 0),
            1 if features.get('is_code', False) else 0,
            float(sig_mean),
            # New AST Features (Must match training!)
            features.get('ast_node_proxy', 0), # Fixed: key was 'ast_depth'
            features.get('import_count', 0)
        ]
        
        # 4. Prediction
        score = 0.5 # Default
        if self.has_model:
            # Predict expects 2D array
            preds = self.model.predict(np.array([feat_vec]))
            score = float(preds[0])
            
        # 5. Hybrid Selection Logic
        
        # Calculate approx tokens for Constraint Layer
        # Crude approximation: 1 token ~= 4 chars
        query_tokens = len(query) // 4
        # Add buffer for response space
        min_context = query_tokens + 512 

        target_cluster = ModelCluster.CHEAP_CHAT # Default
        target_fallback = "phi-3-mini"

        # A. Creative/Reasoning Override (Fixes "Creative Gap")
        if intent in ['creative', 'reasoning'] and intent_conf > 0.4:
            if score > 0.15: # Was 0.7. Lowered because XGBoost underestimates "Reasoning" complexity.
                 target_cluster = ModelCluster.SOTA
                 target_fallback = "gpt-4-turbo"
            else:
                 target_cluster = ModelCluster.HIGH_PERF
                 target_fallback = "llama-3-70b"
                 
        # B. Coding Domain
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

        # C. Math Domain
        elif intent == 'math':
            # Boost Math significantly to avoid 0% accuracy
            if score > 0.15: # Was 0.4. Lowered significantly for GSM8K.
                 target_cluster = ModelCluster.SOTA
                 target_fallback = "gpt-4-turbo"
            else:
                 target_cluster = ModelCluster.HIGH_PERF
                 target_fallback = "llama-3-70b"

        # D. General / Default (Rely on Score)
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

    def _get_fallback_chain(self, cluster: ModelCluster, fallback: str, min_ctx: int = 0) -> list[str]:
        """
        Generate a robust list of candidate models.
        """
        # 1. Primary Candidates (Same Cluster)
        candidates = self.registry.get_candidates(cluster, min_context=min_ctx)
        
        # 2. Constraint Failure Upgrade (if needed)
        # If cluster empty or filtered out by context, try SOTA
        if not candidates and min_ctx > 4000:
             candidates = self.registry.get_candidates(ModelCluster.SOTA, min_context=min_ctx)
        
        # 3. Reliability Safety Net
        # Always append a high-availability model (SOTA) at the end if not present
        safety_net = "gpt-4-turbo" # Or gemini-1.5-pro
        
        final_list = []
        if candidates:
            final_list.extend(candidates)
        else:
            final_list.append(fallback)
            
        if safety_net not in final_list:
            final_list.append(safety_net)
            
        # Ensure fallback is in list if primary failed (redundancy)
        if fallback not in final_list:
             final_list.insert(1, fallback) # Try fallback as second option
             
        return final_list[:3] # Return top 3 candidates

    def _select(self, cluster: ModelCluster, fallback: str, min_ctx: int = 0) -> str:
        """Legacy helper, now deprecated but kept for compatibility."""
        chain = self._get_fallback_chain(cluster, fallback, min_ctx)
        return chain[0]

if __name__ == "__main__":
    # verification
    router = SBSCRRouter()
    
    test_queries = [
        "Write a Python function to parse JSON.",
        "Write a sonnet about the singularity.",
        "Calculate the integral of x^2.",
        "What is the capital of France?",
        "Design a distributed system for chat."
    ]
    
    print("\n--- Routing Verification (Semantic + Structural) ---")
    for q in test_queries:
        print(f"\nQuery: {q}")
        decision = router.route(q)
        print(f" -> Routed to: {decision}")
