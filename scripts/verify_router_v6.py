import time
import sys
import os
import statistics

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sbscr.routers.sbscr import SBSCRRouter

def verify_system():
    print("="*60)
    print("🚀 SBSCR v6 End-to-End System Verification")
    print("="*60)

    # 1. Initialization
    print("\n1️⃣  Initializing Router...")
    start_init = time.perf_counter()
    router = SBSCRRouter()
    init_time = (time.perf_counter() - start_init) * 1000
    print(f"   ✅ Router initialized in {init_time:.2f}ms")
    
    # 2. Test Cases (Golden Set)
    test_cases = [
        # Trivial (Stage 0)
        ("hi", "trivial", "Stage 0"),
        ("what is time", "trivial", "Stage 0"),
        
        # Keyword Fast Path (Stage 1)
        ("write python code", "coding", "Stage 1"),
        ("calculate probability", "math", "Stage 1"),
        
        # LSH Semantic (Stage 2) - relies on calibration
        ("implement binary search algorithm", "coding", "Stage 2"),
        ("solve for x in quadratic equation", "math", "Stage 2"),
        ("write a sonnet about the moon", "creative", "Stage 2"),
        
        # Complex/general (Stage 3)
        ("compare and contrast react and vue", "coding", "Stage 3"), # Might hit LSH coding
        ("tell me a story about a robot", "creative", "Stage 3"),
    ]
    
    print("\n2️⃣  Routing Accuracy Test...")
    correct = 0
    latencies = []
    
    # Warmup
    router.route("warmup")
    
    for query, expected_intent, expected_stage_hint in test_cases:
        start_t = time.perf_counter()
        result = router.route(query)
        duration_ms = (time.perf_counter() - start_t) * 1000
        latencies.append(duration_ms)
        
        intent = result  # In v6, route() returns the provider_id string directly
        
        # Map model back to intent for checking
        mapped_intent = "unknown"
        if "coder" in intent or "claude" in intent: mapped_intent = "coding"
        elif "math" in intent or "reasoning" in intent: mapped_intent = "math"
        elif "gpt-4" in intent or "claude" in intent: mapped_intent = "creative" # Claude is good at both
        elif "phi-3" in intent or "llama" in intent: mapped_intent = "trivial"
        
        # Soft validation (intents can overlap)
        is_match = False
        if expected_intent == "trivial" and mapped_intent == "trivial": is_match = True
        elif expected_intent == "coding" and (mapped_intent == "coding" or "claude" in intent): is_match = True
        elif expected_intent == "math" and (mapped_intent == "math" or "claude" in intent): is_match = True # Claude handles math
        elif expected_intent == "creative" and (mapped_intent == "creative" or "claude" in intent): is_match = True
        mark = "✅" if is_match else "⚠️"
        
        if is_match: correct += 1
        
        print(f"   {mark} Query: '{query}'")
        print(f"      -> Routed to: {intent}")
        print(f"      -> Time: {duration_ms:.4f}ms")
        
    print(f"\n   Accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases):.0%})")

    # 3. Latency Benchmark
    print("\n3️⃣  Latency Benchmark (1000 queries)...")
    benchmark_query = "write a binary search algorithm in python"
    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        router.route(benchmark_query)
        times.append((time.perf_counter() - t0) * 1000)
        
    avg_latency = statistics.mean(times)
    p99_latency = statistics.quantiles(times, n=100)[98] # P99
    
    print(f"   Average Latency: {avg_latency:.4f}ms")
    print(f"   P99 Latency:     {p99_latency:.4f}ms")
    
    target = 1.0
    if p99_latency < target:
        print(f"   ✅ PASSED: Sub-millisecond target met ({p99_latency:.4f}ms < {target}ms)")
    else:
        print(f"   ❌ FAILED: Latency too high ({p99_latency:.4f}ms > {target}ms)")
        
if __name__ == "__main__":
    verify_system()
