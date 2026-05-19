<![CDATA[<p align="center">
  <h1 align="center">🧴 Clinikally SkinAI — Agentic Skincare Assistant</h1>
  <p align="center"><i>An intelligent, full-stack AI skincare assistant with decision-tree routing, multi-source RAG, real-time streaming, and multimodal skin photo analysis.</i></p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-WebSocket-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Gemini_2.5_Flash-LLM-8E75B2?logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Weaviate-Vector_DB-00C853?logo=weaviate&logoColor=white" />
  <img src="https://img.shields.io/badge/DSPy-Structured_LM-FF6F00" />
</p>

---

## 📑 Table of Contents

- [Overview](#overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Bonus Features](#-bonus-features-implemented)
- [Evaluation Alignment](#-evaluation-criteria-alignment)
- [Scalability](#-scalability)
- [Demo](#-demo)

---

## Overview

**Clinikally SkinAI** is an agentic AI system that serves as a premium skincare consultant. It intelligently routes user queries across three specialized tools — product search, clinical blog retrieval, and general dermatological knowledge — using a decision-tree agent that evaluates context, intent, and available data sources at runtime.

The system handles:
- **Product queries** → Semantic search over the SkincareProducts collection (₹ pricing, skin-type filtering)
- **Blog/content queries** → RAG over the SkincareBlogs collection (ingredient science, routines, guides)
- **General skincare queries** → External LLM knowledge with clinical dermatology expertise
- **Hybrid queries** → Automatic multi-tool orchestration when queries span categories
- **Image analysis** → Multimodal skin photo diagnostics via Gemini Vision (bonus)

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Chat Interface (Next.js SPA)                │
│  ┌──────────┐ ┌──────────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ Streaming │ │ Product Cards│ │ Feedback │ │ Image Upload  │  │
│  │ Markdown  │ │ ₹ Prices     │ │ 👍 👎    │ │ Drag & Drop   │  │
│  └──────────┘ └──────────────┘ └──────────┘ └───────────────┘  │
└───────────────────────┬─────────────────────────────────────────┘
                        │ WebSocket (ws://host:8000/ws/query)
┌───────────────────────▼─────────────────────────────────────────┐
│                   FastAPI Backend (Uvicorn)                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Decision Tree Agent (DSPy)                   │   │
│  │  ┌─────────────┐ ┌────────────┐ ┌────────────────────┐  │   │
│  │  │ Preprocessor │ │  Router    │ │  Postprocessor     │  │   │
│  │  │ (Intent +   │ │  (Chain of │ │  (Response Format  │  │   │
│  │  │  NER)       │ │   Thought) │ │   + Citations)     │  │   │
│  │  └─────────────┘ └─────┬──────┘ └────────────────────┘  │   │
│  │                        │                                  │   │
│  │  ┌─────────────────────▼──────────────────────────────┐  │   │
│  │  │              Specialist Tools                       │  │   │
│  │  │  ┌─────────────┐ ┌──────────┐ ┌────────────────┐  │  │   │
│  │  │  │ ProductQuery│ │ BlogRAG  │ │ GeneralKnowledge│  │  │   │
│  │  │  │    Tool     │ │  Tool    │ │     Tool        │  │  │   │
│  │  │  └──────┬──────┘ └────┬─────┘ └───────┬────────┘  │  │   │
│  │  │         │             │               │            │  │   │
│  │  │  ┌──────▼──────┐ ┌────▼─────┐  ┌──────▼────────┐  │  │   │
│  │  │  │ Weaviate    │ │ Weaviate │  │ Gemini 2.5    │  │  │   │
│  │  │  │ Products    │ │  Blogs   │  │ Flash (LLM)   │  │  │   │
│  │  │  └─────────────┘ └──────────┘  └───────────────┘  │  │   │
│  │  │                                                    │  │   │
│  │  │  ┌─────────────────────────────────────────────┐   │  │   │
│  │  │  │ SkinAnalysisTool (Gemini Vision + Fallback) │   │  │   │
│  │  │  └─────────────────────────────────────────────┘   │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Decision-tree agent** over flat tool-list | Enables multi-step reasoning with context awareness; the agent evaluates past actions, available tools, and future strategy before routing |
| **DSPy for LLM orchestration** | Structured, typed prompting with automatic optimization; no brittle prompt templates |
| **WebSocket streaming** | Real-time token-by-token delivery for natural chat UX; lower perceived latency |
| **Weaviate Cloud** | Managed vector DB with hybrid search (dense + BM25); no infrastructure overhead |
| **Multi-model fallback** | 5-model retry chain (Gemini → OpenRouter) prevents single-point-of-failure for vision |
| **Force-routing for images** | Bypasses routing LLM ambiguity when a skin photo is attached |

---

## ✨ Features

### Core (Task 1 — Agentic Backend)
- ✅ **Intent-aware routing** — Decision agent classifies queries and routes to the optimal tool(s)
- ✅ **ProductQueryTool** — Semantic + filtered search over SkincareProducts (price, skin type, category)
- ✅ **BlogRAGTool** — RAG retrieval over SkincareBlogs with source attribution
- ✅ **GeneralKnowledgeTool** — Clinical dermatology knowledge via Gemini 2.5 Flash
- ✅ **Multi-turn context** — Conversation state persisted in Weaviate decision trees
- ✅ **Hybrid queries** — Automatic tool combination for complex queries spanning categories

### Chat Interface (Task 2 — Full-Stack)
- ✅ **Premium skincare-themed UI** — Custom spa pastel theme (cream, emerald, pink accents)
- ✅ **Real-time streaming** — Token-by-token WebSocket response rendering
- ✅ **Product cards** — Rich cards with ₹ pricing, skin-type badges, and star ratings
- ✅ **Source attribution** — Clear badges showing which data source grounded each response
- ✅ **Welcome message** — Branded intro with quick-start prompt suggestions
- ✅ **Responsive design** — Works on mobile and desktop
- ✅ **Markdown rendering** — Tables, lists, bold, headers in responses

### Bonus Features
- ✅ **Streaming responses** — WebSocket-based real-time token streaming
- ✅ **Robust error handling** — Graceful degradation with fallback to GeneralKnowledgeTool; multi-model retry chain for vision
- ✅ **Conversation persistence** — localStorage-based session persistence across page refreshes
- ✅ **User feedback** — 👍/👎 buttons on every response, stored via `/feedback/add` endpoint
- ✅ **Scalability documentation** — Comprehensive [SCALABILITY.md](SCALABILITY.md) covering horizontal scaling, caching, and cost optimization
- ✅ **Image upload + skin analysis** — Kawaii camera icon, drag-and-drop, Gemini Vision analysis with clinical diagnostic reports

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| **LLM** | Google Gemini 2.5 Flash (via litellm) |
| **LLM Orchestration** | DSPy (structured prompting + chain-of-thought) |
| **Vector Database** | Weaviate Cloud (hybrid search: dense + BM25) |
| **Backend** | FastAPI + Uvicorn (WebSocket + REST) |
| **Frontend** | Next.js SPA (React, served as static bundle) |
| **Vision** | Gemini 2.5 Flash / 2.0 Flash multimodal (with OpenRouter fallback) |
| **API Routing** | OpenRouter (multi-model access + free tier fallback) |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12
- A Weaviate Cloud account (or local Weaviate instance)
- API keys: `GEMINI_API_KEY`, `OPENROUTER_API_KEY`

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd skinai

# 2. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install in editable mode
pip install -e .

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your API keys and Weaviate credentials
```

### Environment Variables

```env
# Weaviate Cloud
WCD_URL=https://your-cluster.weaviate.cloud
WCD_API_KEY=your-weaviate-api-key

# LLM Models (via OpenRouter)
BASE_MODEL=openai/gpt-oss-120b:free
COMPLEX_MODEL=openai/gpt-oss-120b:free
OPENROUTER_API_KEY=sk-or-v1-...

# Gemini (for vision analysis)
GEMINI_API_KEY=AIza...
```

### Run

```bash
# Start the server
skinai start --port 8000

# Open in browser
open http://localhost:8000
```

The app will automatically detect and preprocess your Weaviate collections on first start.

---

## 📁 Project Structure

```
skinai/
├── skinai/                     # Main Python package
│   ├── api/
│   │   ├── app.py              # FastAPI application setup
│   │   ├── cli.py              # CLI entry point (skinai start)
│   │   ├── custom_tools.py     # 4 specialist tools (Product, Blog, General, SkinAnalysis)
│   │   ├── routes/
│   │   │   ├── query.py        # WebSocket query handler + image routing
│   │   │   ├── feedback.py     # User feedback endpoint
│   │   │   └── ...
│   │   └── static/             # Frontend build (Next.js SPA)
│   │       └── index.html      # Enhanced with skincare theme + vision UI
│   ├── tree/                   # Decision tree agent implementation
│   ├── preprocessing/          # Collection preprocessing + prompt templates
│   ├── tools/                  # Built-in retrieval tools (query, aggregate)
│   └── util/                   # Chain-of-thought, client manager, utilities
├── test_bonus_features.py      # Automated regression test suite
├── API.md                      # API endpoint documentation
├── SCALABILITY.md              # Scalability architecture documentation
├── .env.example                # Environment variable template
├── pyproject.toml              # Package configuration
└── README.md                   # This file
```

---

## 📡 API Reference

Full documentation in [API.md](API.md). Key endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the chat interface |
| `/init/user/{user_id}` | POST | Initialize/load a user session |
| `/init/tree/{user_id}/{conversation_id}` | POST | Initialize a conversation tree |
| `/ws/query` | WebSocket | Main chat endpoint — send queries, receive streaming responses |
| `/feedback/add` | POST | Submit 👍/👎 feedback on a response |

### WebSocket Message Format

**Client → Server:**
```json
{
  "user_id": "user123",
  "conversation_id": "conv_abc",
  "query": "Recommend a moisturiser under ₹1200 for oily skin",
  "query_id": "q_001",
  "collection_names": ["SkincareProducts", "SkincareBlogs"],
  "image": "data:image/jpeg;base64,..." 
}
```

**Server → Client (streaming):**
```json
{
  "type": "response",
  "content": "Based on your requirements...",
  "objects": [...],
  "status": "status message"
}
```

---

## 🏅 Bonus Features Implemented

| Bonus | Status | Implementation |
|-------|--------|----------------|
| Streaming responses | ✅ | WebSocket token-by-token streaming with real-time Markdown rendering |
| Robust error handling | ✅ | Multi-model retry chain (5 models), graceful fallback to GeneralKnowledgeTool, meaningful error states in UI |
| Conversation persistence | ✅ | localStorage + Weaviate tree storage; survives page refresh |
| User feedback | ✅ | 👍/👎 buttons on every response → POST `/feedback/add` |
| Scalability docs | ✅ | Comprehensive [SCALABILITY.md](SCALABILITY.md) with horizontal scaling, caching, cost optimization |
| Image upload + analysis | ✅ | Kawaii camera icon, drag-and-drop upload, Gemini Vision clinical diagnostic, 5-model fallback chain |

---

## 📊 Evaluation Criteria Alignment

| Criterion | How We Address It |
|-----------|-------------------|
| **Agentic architecture quality** | Decision-tree agent with chain-of-thought routing, 4 specialist tools, multi-step reasoning, context-aware tool selection, force-routing for images |
| **Response quality** | Responses grounded in Weaviate data with source attribution; clinical dermatology expertise; ₹ pricing; ingredient-specific advice |
| **Full-stack execution** | Premium skincare-themed chat UI, real-time streaming, product cards, responsive design, drag-and-drop image upload |
| **Code clarity** | Clean package structure, comprehensive docstrings, type hints, clear separation of concerns |
| **Documentation & deployment** | README, API.md, SCALABILITY.md, .env.example, automated test suite |
| **Demo quality** | Screen recording walkthrough covering architecture, all 3 query types, image analysis, and design decisions |

---

## 📈 Scalability

See [SCALABILITY.md](SCALABILITY.md) for the full scalability architecture document. Key strategies:

- **Horizontal scaling** — Stateless FastAPI backend behind load balancer with multiple Uvicorn workers
- **Database scaling** — Weaviate Cloud auto-sharding with Redis caching layer
- **LLM scaling** — Multi-model fallback chain, rate-limit handling, cost-tier routing
- **Graceful degradation** — Always returns a response, even if all data sources fail

---

## 🎬 Demo

> 📹 [Screen Demo Recording](#) — A 5–10 minute walkthrough covering:
> 1. System architecture and design decisions
> 2. Live demo: product query, blog query, general query, image analysis
> 3. Challenges encountered and resolutions

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built with 🧴 for the Clinikally Agentic Skincare AI Internship Assignment</i>
</p>
]]>
