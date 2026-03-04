# Signature-Based Structural Complexity Routing (SBSCR) ⚡

**Sub-Millisecond LLM Routing via Locality-Sensitive Hashing and Structural Metadata**

A lightweight framework that replaces dense embeddings with Locality-Sensitive Hashing (LSH) signatures combined with structural metadata extraction to solve the Decision-Cost Paradox in LLM routing. Routes queries to the optimal LLM in **<1ms**.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Astro](https://img.shields.io/badge/Astro-Frontend-FF5D01?logo=astro&logoColor=white)](https://astro.build)
[![Vercel](https://img.shields.io/badge/Deployed-Vercel-000000?logo=vercel&logoColor=white)](https://frontend-seven-eta-98.vercel.app)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Latency](https://img.shields.io/badge/P99_Latency-0.81ms-brightgreen)](/)

---

## 🎯 Live Demo

👉 **[https://frontend-seven-eta-98.vercel.app](https://frontend-seven-eta-98.vercel.app)**

Try the interactive routing simulation with real-time pipeline visualization!

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **⚡ Sub-Millisecond Routing** | 0.14ms average latency (7,000x faster than neural classifiers) |
| **🧠 LSH Semantic Bucketing** | O(1) query classification using locality-sensitive hashing |
| **🌐 OpenRouter Integration** | Live access to 346+ models (GPT-4o, Claude 3.5, Llama 3.1, etc.) |
| **📊 Glass-Box Observability** | Full visibility into every routing decision |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SBSCR Pipeline                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Query ──▶ [Stage 0: Trivial Filter] ──▶ CHEAP_CHAT (if < 10 chars)│
│              │                                                      │
│              ▼                                                      │
│         [Stage 1: Keyword Fast Path] ──▶ Intent Match (coding/math) │
│              │                                                      │
│              ▼                                                      │
│         [Stage 2: LSH Bucket Routing] ──▶ Semantic Classification   │
│              │                                                      │
│              ▼                                                      │
│         [Stage 3: Complexity Scoring] ──▶ Model Selection           │
│              │                                                      │
│              ▼                                                      │
│         [Model Registry] ──▶ 346 OpenRouter Models                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Stage Breakdown

| Stage | Name | Latency | Purpose |
|-------|------|---------|---------|
| 0 | Trivial Filter | ~0.01ms | Catch "hi", "hello", simple math |
| 1 | Keyword Fast Path | ~0.02ms | Detect `def`, `solve`, `write` patterns |
| 2 | LSH Bucket | ~0.08ms | Semantic hash → intent bucket |
| 3 | Complexity Score | ~0.03ms | Keyword-based difficulty estimation |

**Total: ~0.14ms** per routing decision

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+ (for frontend)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/alphagangs/sbscr-router.git
cd sbscr-router

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run dashboard.py
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:4321](http://localhost:4321)

---

## 📊 Performance Benchmarks

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Average Latency | **0.14ms** | <1ms | ✅ PASS |
| P99 Latency | **0.81ms** | <1ms | ✅ PASS |
| Model Registry Size | 346 | - | 🌐 Live |
| Routing Accuracy | ~85%* | - | ✅ Good |

*Accuracy depends on bucket calibration quality

---

## 🔧 Key Components

```
sbscr-router/
├── sbscr/
│   ├── core/
│   │   ├── lsh.py          # LSH signature generator & bucket router
│   │   ├── normalizer.py   # Semantic normalizer & trivial detector
│   │   └── registry.py     # Model registry (loads OpenRouter data)
│   └── routers/
│       └── sbscr.py        # Main router implementation
├── data/
│   ├── models.yaml         # 346 models from OpenRouter
│   ├── bucket_map.json     # Calibrated LSH bucket → intent mapping
│   └── synonyms.yaml       # Domain synonym dictionary
├── scripts/
│   ├── fetch_openrouter_models.py  # Refresh model registry
│   ├── calibrate_lsh_buckets.py    # Bucket calibration
│   └── verify_router.py            # End-to-end benchmarks
├── frontend/               # Astro + Vercel frontend
│   └── src/pages/index.astro
└── dashboard.py            # Streamlit observability UI
```

---

## 🌐 OpenRouter Integration

The router fetches live model data from OpenRouter's API:

```bash
# Refresh model registry
python scripts/fetch_openrouter_models.py
```

This updates `data/models.yaml` with:
- Model IDs (e.g., `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`)
- Pricing (per 1M tokens)
- Context window sizes
- Auto-clustered into: **SOTA**, **High Perf**, **Fast Code**, **Cheap Chat**



## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🔗 Links

- **Live Demo**: [frontend-seven-eta-98.vercel.app](https://frontend-seven-eta-98.vercel.app)
- **GitHub**: [github.com/alphagangs/sbscr-router](https://github.com/alphagangs/sbscr-router)
- **OpenRouter**: [openrouter.ai](https://openrouter.ai)

---

<p align="center">
  Built with ❤️
</p>
