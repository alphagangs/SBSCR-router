import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys

# Ensure path is set (for when running as script)
sys.path.append(os.getcwd())

from sklearn.model_selection import train_test_split
from sbscr.core.metadata import ComplexityExtractor
from sbscr.core.lsh import LSHSignatureGenerator

def train_xgboost_scorer():
    print("🚀 Starting Scorer Training...")
    
    # Try 1M file first, then fallback to 50k
    data_path = "data/lmsys/processed_lmsys_1m.csv"
    if not os.path.exists(data_path):
        data_path = "data/lmsys/processed_lmsys_50k.csv"
        
    if not os.path.exists(data_path):
        print("❌ Data file not found.")
        return
        
    df = pd.read_csv(data_path)
    print(f"📊 Loaded {len(df)} samples.")
    
    # Full 1M Training Mode
    # df = df.sample(n=100000, random_state=42) # Uncomment for fast debugging
    pass
    
    extractor = ComplexityExtractor()
    lsh = LSHSignatureGenerator(num_perm=16)
    
    feature_rows = []
    labels = []
    
    print("⚙️  Extracting Structural Features...")
    
    for idx, row in df.iterrows():
        try:
            prompt = str(row['prompt'])
            
            # Target: Distill the Heuristic (Pseudo-labeling)
            # Since we don't have human labels for "complexity" on LMSYS, 
            # we use our domain-expert heuristic as the 'Teacher' and train XGBoost as the 'Student'.
            target = extractor.estimate_complexity(prompt) / 10.0 # Normalize 0-1
            
            # Features
            features = extractor.extract_features(prompt)
            
            # LSH Handling
            sig = lsh.generate_signature(prompt)
            # Check if sig is MinHash object or array
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
                features.get('code_density', 0),        # Existing
                1 if features.get('is_code', False) else 0,
                float(sig_mean),
                # New AST Features
                features.get('ast_depth', 0),
                features.get('import_count', 0)
            ]
            
            feature_rows.append(feat_vec)
            labels.append(target)
            
            if idx > 0 and idx % 5000 == 0:
                print(f"Processed {idx} rows...")

        except Exception as e:
            print(f"❌ Error on row {idx}: {e}")
            if idx == 0: raise e # Fail fast
            continue
            
    print(f"📉 Extracted {len(feature_rows)} valid training pairs.")
    
    if len(feature_rows) == 0:
        print("❌ No valid data to train on!")
        return

    X = np.array(feature_rows)
    y = np.array(labels)
    
    print("🧠 Training XGBoost...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"📈 Test R^2 Score: {score:.4f}")
    
    os.makedirs("sbscr/models", exist_ok=True)
    model.save_model("sbscr/models/complexity_xgboost.json")
    print("💾 Model saved to sbscr/models/complexity_xgboost.json")

if __name__ == "__main__":
    train_xgboost_scorer()
