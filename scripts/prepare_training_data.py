"""
Unified Data Preparation Script for SBSCR Router.

This script automates the acquisition and formatting of real-world datasets:
1. Specialized Data (HumanEval, GSM8K) -> Hard-labeled as 'coding'/'math'.
2. General Data (LMSYS-Chat-1M) -> Auto-labeled using the router's SemanticNormalizer.

Usage:
    pip install datasets pandas
    python scripts/prepare_training_data.py

Output:
    data/training_data.jsonl: Unified dataset for LSH calibration.
    data/dataset_stats.json: Statistics distribution.
"""

import os
import sys
import json
import random
import logging
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sbscr.core.normalizer import SemanticNormalizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_huggingface_dataset(dataset_name: str, config_name: str = None, split: str = 'train', limit: int = None):
    """Safe loading of HF datasets."""
    try:
        from datasets import load_dataset
        name_str = f"{dataset_name}/{config_name}" if config_name else dataset_name
        logger.info(f"Downloading {name_str} ({split})...")
        ds = load_dataset(dataset_name, name=config_name, split=split, streaming=True)
        return ds.take(limit) if limit else ds
    except ImportError:
        logger.error("❌ 'datasets' library not found. Please run: pip install datasets")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to load {dataset_name}: {e}")
        return []

def process_specialized_datasets() -> List[Dict]:
    """Load specialized datasets with known intents."""
    processed = []
    
    # 1. HumanEval (Coding)
    # Using 'openai_humaneval' which is standard
    ds = load_huggingface_dataset("openai_humaneval", split="test", limit=2000) # It's small
    for item in ds:
        processed.append({
            "query": item["prompt"],
            "intent": "coding",
            "source": "humaneval"
        })
    logger.info(f"Loaded {len(processed)} coding samples from HumanEval")

    # 2. GSM8K (Math)
    count = 0
    # Pass "main" as positional arg for config_name
    ds = load_huggingface_dataset("gsm8k", "main", split="train", limit=2000)
    for item in ds:
        processed.append({
            "query": item["question"],
            "intent": "math",
            "source": "gsm8k"
        })
        count += 1
    logger.info(f"Loaded {count} math samples from GSM8K")

    # 3. Creative Writing (WritingPrompts)
    count = 0
    ds = load_huggingface_dataset("euclaise/writingprompts", split="train", limit=2000)
    for item in ds:
        # Tries 'prompt' or 'text' column
        text = item.get("prompt") or item.get("text")
        if text:
            processed.append({
                "query": text,
                "intent": "creative",
                "source": "writingprompts"
            })
            count += 1
    logger.info(f"Loaded {count} creative samples from WritingPrompts")
    
    return processed

def process_general_datasets(limit: int = 10000) -> List[Dict]:
    """Load general datasets and auto-label them."""
    processed = []
    normalizer = SemanticNormalizer("data/synonyms.yaml")
    
    # LMSYS-Chat-1M (using a subset)
    ds = load_huggingface_dataset("lmsys/lmsys-chat-1m", split="train", limit=limit)
    
    logger.info("Auto-labeling general data...")
    count = 0
    for item in ds:
        # Extract the first user message
        conversation = item.get("conversation", [])
        if not conversation:
            continue
            
        user_msg = conversation[0]["content"]
        
        # Heuristic Auto-Labeling
        # We use the semantic normalizer's detection logic
        intent, conf = normalizer.detect_intent_fast(user_msg)
        
        # Only keep if we have some confidence, otherwise mark as 'general'
        # or distinct 'unlabeled' for semi-supervised approaches.
        # For calibration, we want explicit buckets.
        
        final_intent = intent if conf > 0.3 else "general"
        
        processed.append({
            "query": user_msg,
            "intent": final_intent,
            "source": "lmsys",
            "auto_labeled": True
        })
        count += 1
        
        if count % 1000 == 0:
            print(f"Processed {count} general samples...", end="\r")
            
    print()
    return processed

def main():
    logger.info("🚀 Starting training data preparation...")
    
    # 1. Get Specialized Data
    specialized = process_specialized_datasets()
    
    # 2. Get General Data
    general = process_general_datasets(limit=10000) # Start with 10k for speed
    
    combined = specialized + general
    
    # 3. Save
    output_path = "data/training_data.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in combined:
            f.write(json.dumps(item) + "\n")
            
    # 4. Stats
    from collections import Counter
    stats = Counter(item["intent"] for item in combined)
    
    logger.info("="*40)
    logger.info("✅ Data Preparation Complete")
    logger.info(f"Saved to: {output_path}")
    logger.info(f"Total Samples: {len(combined)}")
    logger.info("Distribution:")
    for intent, count in stats.most_common():
        logger.info(f"  - {intent}: {count}")
    logger.info("="*40)
    
    print("\nNext Step: Run 'python scripts/calibrate_lsh_buckets.py' to use this data.")

if __name__ == "__main__":
    main()
