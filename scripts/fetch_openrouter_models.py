import requests
import yaml
import sys
import os

# Ensure we can import from sbscr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def fetch_openrouter_models():
    print("🚀 Fetching live models from OpenRouter...")
    try:
        response = requests.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ Failed to fetch models: {e}")
        return

    raw_models = data.get("data", [])
    print(f"📚 Found {len(raw_models)} models from OpenRouter.")

    processed_models = {}
    
    # Heuristics to assign clusters based on ID/Name
    for model in raw_models:
        mid = model.get("id", "")
        name = model.get("name", mid)
        context = model.get("context_length", 4096)
        
        # Pricing is usually per 1M tokens in OpenRouter API? 
        # API returns pricing strings like '0.000005' per token.
        # We need per 1M.
        pricing = model.get("pricing", {})
        prompt_price_token = float(pricing.get("prompt", 0) or 0)
        completion_price_token = float(pricing.get("completion", 0) or 0)
        
        price_in = prompt_price_token * 1_000_000
        price_out = completion_price_token * 1_000_000
        
        # --- CLUSTERING LOGIC ---
        cluster = "unknown"
        mid_lower = mid.lower()
        
        # SOTA
        if "gpt-4" in mid_lower or "claude-3-opus" in mid_lower or "sonnet-3.5" in mid_lower or "gemini-1.5-pro" in mid_lower or "o1-" in mid_lower:
            cluster = "sota"
        
        # HIGH PERF
        elif "llama-3-70b" in mid_lower or "mistral-large" in mid_lower or "qwen-2-72b" in mid_lower or "mixtral" in mid_lower:
            cluster = "high_perf"
            
        # FAST CODE
        elif "coder" in mid_lower or "deepseek-v2" in mid_lower or "starcoder" in mid_lower:
            cluster = "fast_code"
            
        # CHEAP CHAT
        elif "llama-3-8b" in mid_lower or "phi-3" in mid_lower or "haiku" in mid_lower or "gemma" in mid_lower or "flash" in mid_lower:
            cluster = "cheap_chat" 
            
        # Default fallback for unknown
        elif price_in < 1.0:
            cluster = "cheap_chat"
        elif price_in > 5.0:
            cluster = "high_perf"

        # --- SCORING HEURISTICS (Mock based on cluster) ---
        # In a real app, we would join this with a benchmark database (OpenCompass/LMSYS)
        reasoning = 60.0
        coding = 60.0
        
        if cluster == "sota":
            reasoning = 90.0
            coding = 90.0
        elif cluster == "high_perf":
            reasoning = 80.0
            coding = 75.0
        elif cluster == "fast_code":
            reasoning = 70.0
            coding = 90.0
        elif cluster == "cheap_chat":
            reasoning = 60.0
            coding = 50.0

        # Create entry
        processed_models[mid] = {
            "provider": mid.split("/")[0] if "/" in mid else "openrouter",
            "cluster": cluster,
            "context_window": context,
            "price_in": round(price_in, 4),
            "price_out": round(price_out, 4),
            "reasoning": reasoning,
            "coding": coding,
            "description": model.get("description", "")
        }

    # Sort by cluster for readability
    sorted_models = dict(sorted(processed_models.items(), key=lambda x: (x[1]['cluster'], x[0])))

    # Write to YAML
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "models.yaml")
    
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump({"models": sorted_models}, f, sort_keys=False)
        
    print(f"✅ Automatically generated registry with {len(processed_models)} models at: {output_path}")

if __name__ == "__main__":
    fetch_openrouter_models()
