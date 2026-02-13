# serve.py - Full Multi-Provider Server (12 Free Models)
"""
API Router with Multi-Provider Infrastructure
- Groq: 5 models (free, fast)
- Hugging Face: 5 models (free)
- Google: 2 models (free)
Total: 12 models across all clusters
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import time
import os

from sbscr.routers.sbscr import SBSCRRouter
from sbscr.providers.base import ProviderRegistry
from sbscr.providers.groq_provider import GroqProvider
from sbscr.providers.huggingface_provider import HuggingFaceProvider
from sbscr.providers.google_provider import GoogleProvider

app = FastAPI(title="SBSCR Router API")

# Initialize Router
print("🚀 Initializing Router API...")
router = SBSCRRouter()

# Initialize Provider Registry
print("🔌 Initializing 3 FREE Providers (Groq, HF, Google)...")
providers = ProviderRegistry()
try:
    providers.register(GroqProvider())
    providers.register(HuggingFaceProvider())
    providers.register(GoogleProvider())
except Exception as e:
    print(f"⚠️ Warning: Provider init failed: {e}")

print(f"✅ Available models: {providers.list_available_models()}")

# Model mapping: Router output -> Provider model
# Fallback strategy: Always map to a model that exists in the free cluster
MODEL_MAP = {
    # SOTA Tier (Claude/GPT-4 replacements)
    "claude-3-5-sonnet": "llama-3.1-70b",      # Groq (Best reasoning)
    "gpt-4-turbo": "qwen2.5-72b",              # HF (Strong alternative)
    "gemini-1.5-pro": "gemini-1.5-pro",        # Google
    
    # High Performance Tier
    "llama-3-70b": "llama-3.3-70b",            # Groq
    "mistral-large": "mixtral-8x7b",           # Groq
    "mixtral-8x22b": "mistral-7b",             # HF
    
    # Code Specialist Tier
    "deepseek-coder-v2": "deepseek-coder-v2",  # HF (Best free code model)
    "starcoder2-15b": "starcoder2-15b",        # HF
    
    # Cheap/Fast Tier
    "llama-3-8b": "llama-3.1-8b",              # Groq (Super fast)
    "phi-3-mini": "phi-3-mini",                # HF
    "gemma-7b": "gemini-1.5-flash",            # Google (Fastest)
    "haiku-3": "llama-3.1-8b",                 # Groq
}


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "sbscr-auto"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not request.messages:
        raise HTTPException(400, "No messages provided")
    
    query = request.messages[-1].content
    
    # Step 1: Route using Router
    start = time.time()
    selected_model = router.route(query)
    routing_latency = (time.time() - start) * 1000
    
    # Step 2: Map to provider model
    provider_model = MODEL_MAP.get(selected_model, "llama-3.1-8b")
    
    # Step 3: Execute via provider
    try:
        inference_start = time.time()
        messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
        
        response = providers.call(
            provider_model,
            messages_dict,
            request.max_tokens,
            request.temperature
        )
        inference_latency = (time.time() - inference_start) * 1000
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": selected_model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(query) // 4,
                "completion_tokens": len(response) // 4,
                "total_tokens": (len(query) + len(response)) // 4,
                "routing_decision": selected_model,
                "provider_model": provider_model,
                "routing_latency_ms": round(routing_latency, 2),
                "inference_latency_ms": round(inference_latency, 2),
                "total_latency_ms": round(routing_latency + inference_latency, 2),
                "cost": 0.0  # Free tier!
            }
        }
        
    except Exception as e:
        # Fallback chain
        fallback_chain = router.route_with_fallbacks(query)
        
        for fallback_model in fallback_chain[1:]:  # Skip primary
            try:
                provider_fallback = MODEL_MAP.get(fallback_model, "llama-3.1-8b")
                messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
                
                response = providers.call(
                    provider_fallback,
                    messages_dict,
                    request.max_tokens,
                    request.temperature
                )
                
                return {
                    "id": f"chatcmpl-fallback-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": fallback_model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "fallback_used": True,
                        "original_model": selected_model,
                        "fallback_model": fallback_model,
                        "provider_fallback": provider_fallback
                    }
                }
            except:
                continue
        
        raise HTTPException(503, f"All providers failed: {str(e)}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "available_models": providers.list_available_models(),
        "providers": {
            "groq": os.getenv("GROQ_API_KEY") is not None,
            "huggingface": os.getenv("HF_TOKEN") is not None,
            "google": os.getenv("GOOGLE_API_KEY") is not None
        }
    }

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏢 SBSCR ROUTER API - ONLINE")
    print("="*60)
    print("\n📊 Model Output Mapping:")
    print("   - Claude Sonnet  -> Groq: Llama 3.1 70B")
    print("   - GPT-4 Turbo    -> HF: Qwen 2.5 72B")
    print("   - DeepSeek Coder -> HF: DeepSeek V2 Lite")
    print("\n🔑 Required API Keys (FREE):")
    print(f"   GROQ_API_KEY: {'✅ Set' if os.getenv('GROQ_API_KEY') else '❌ Missing'}")
    print(f"   HF_TOKEN:     {'✅ Set' if os.getenv('HF_TOKEN') else '❌ Missing'}")
    print(f"   GOOGLE_API_KEY:{'✅ Set' if os.getenv('GOOGLE_API_KEY') else '❌ Missing'}")
    print("\n🌐 Server: http://localhost:8000")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
