# run_judgment.py - Custom MT-Bench Judgment (Uses Free LLM as Judge)
"""
Judges MT-Bench answers using your local server (Groq/HF/Google).
No GPT-4 API key required - uses Llama 3.3 70B as judge.
"""

import json
import requests
import time
from pathlib import Path

# Configuration
SERVER_URL = "http://localhost:8000/v1/chat/completions"
ANSWERS_FILE = "data/mt_bench/model_answer/sbscr-auto.jsonl"
QUESTIONS_FILE = "data/mt_bench/question.jsonl"
OUTPUT_FILE = "data/mt_bench/model_judgment/sbscr-auto-judgment.jsonl"

JUDGE_PROMPT = """You are an expert judge evaluating AI assistant responses. 

**Question:** {question}

**Assistant's Answer:** {answer}

**Evaluation Criteria:**
1. Helpfulness: Does the answer address the question fully?
2. Accuracy: Is the information correct?
3. Clarity: Is the answer well-organized and easy to understand?
4. Depth: Does it provide sufficient detail?

**Your Task:**
Rate the answer on a scale of 1-10, where:
- 1-3: Poor (incorrect, unhelpful, or confusing)
- 4-6: Average (partially helpful, some issues)
- 7-8: Good (helpful, accurate, clear)
- 9-10: Excellent (exceptional quality, comprehensive)

Respond with ONLY a JSON object in this exact format:
{{"score": <number>, "explanation": "<brief explanation>"}}
"""

def load_questions():
    """Load questions."""
    questions = {}
    with open(QUESTIONS_FILE, 'r') as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                questions[q["question_id"]] = q
    return questions

def load_answers():
    """Load generated answers."""
    answers = []
    with open(ANSWERS_FILE, 'r') as f:
        for line in f:
            if line.strip():
                answers.append(json.loads(line))
    return answers

def call_judge(question, answer):
    """Ask the LLM to judge an answer."""
    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    
    payload = {
        "model": "sbscr-auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.3
    }
    
    response = requests.post(SERVER_URL, json=payload, timeout=120)
    
    if response.status_code != 200:
        return {"score": 0, "explanation": f"Error: {response.text}"}
    
    content = response.json()["choices"][0]["message"]["content"]
    
    # Parse JSON response
    try:
        # Try to extract JSON from response
        if "{" in content and "}" in content:
            json_str = content[content.find("{"):content.rfind("}")+1]
            return json.loads(json_str)
        else:
            # Try to extract score from text
            import re
            score_match = re.search(r'(\d+)(?:/10)?', content)
            if score_match:
                return {"score": int(score_match.group(1)), "explanation": content}
            return {"score": 5, "explanation": content}
    except:
        return {"score": 5, "explanation": content}

def run_judgment():
    """Run judgment on all answers."""
    print("="*60)
    print("🏛️ MT-BENCH JUDGMENT (Free LLM Judge)")
    print("="*60)
    print(f"Using: {SERVER_URL} (Llama 3.3 70B as judge)")
    print("="*60)
    
    questions = load_questions()
    answers = load_answers()
    
    results = []
    total_score = 0
    total_turns = 0
    
    for answer in answers:
        qid = answer["question_id"]
        question_data = questions.get(qid, {})
        category = question_data.get("category", "unknown")
        turns = question_data.get("turns", [])
        answer_turns = answer["choices"][0]["turns"]
        
        print(f"\n📝 Question {qid} ({category})")
        
        question_scores = []
        
        for i, (q_turn, a_turn) in enumerate(zip(turns, answer_turns)):
            if a_turn.startswith("$ERROR$"):
                print(f"   Turn {i+1}: ⚠️ Skipped (error response)")
                continue
            
            print(f"   Turn {i+1}: Judging...", end=" ", flush=True)
            
            judgment = call_judge(q_turn, a_turn)
            score = judgment.get("score", 5)
            explanation = judgment.get("explanation", "")[:100]
            
            print(f"Score: {score}/10")
            
            question_scores.append(score)
            total_score += score
            total_turns += 1
        
        avg_score = sum(question_scores) / len(question_scores) if question_scores else 0
        
        result = {
            "question_id": qid,
            "category": category,
            "turn_scores": question_scores,
            "average_score": round(avg_score, 2),
            "tstamp": time.time()
        }
        results.append(result)
        
        print(f"   Average: {avg_score:.1f}/10")
    
    # Save results
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    # Summary
    avg_total = total_score / total_turns if total_turns > 0 else 0
    
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    for r in results:
        print(f"   Q{r['question_id']} ({r['category']}): {r['average_score']:.1f}/10")
    
    print("-"*60)
    print(f"   OVERALL SCORE: {avg_total:.2f}/10")
    print("="*60)
    
    # Interpretation
    if avg_total >= 8:
        print("🏆 EXCELLENT - Ready!")
    elif avg_total >= 6:
        print("✅ GOOD - Suitable for most use cases")
    elif avg_total >= 4:
        print("⚠️ AVERAGE - Consider improvements")
    else:
        print("❌ POOR - Needs significant work")
    
    print(f"\n💾 Results saved to {OUTPUT_FILE}")
    
    return avg_total

if __name__ == "__main__":
    run_judgment()
