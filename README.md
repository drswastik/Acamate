```markdown
# AcaMate: Multi-Agent Hybrid RAG & Hallucination Guardrail System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-00599C?style=flat)](https://github.com/facebookresearch/faiss)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ollama](https://img.shields.io/badge/Local_LLM-Ollama_(Llama_3.1)-000000?style=flat)](https://ollama.ai/)
[![Gemini API](https://img.shields.io/badge/Cloud_LLM-Gemini_Flash--Lite-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev/)

AcaMate is an enterprise-grade academic research assistant and conversational engine designed to eliminate hallucination risks, maintain infinite conversational memory without token bloat, and dynamically balance compute workloads between local and cloud inference engines.

---

## 🏗️ System Architecture

```text
                              [ User Prompt ]
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │   Dense Semantic Vector Router  │
                    │      (all-MiniLM-L6-v2 EBR)     │
                    └────────────────┬────────────────┘
                                     │
               ┌─────────────────────┴─────────────────────┐
               ▼                                           ▼
     [ RAG Needed: TRUE ]                        [ RAG Needed: FALSE ]
               │                                           │
   ┌───────────┴───────────┐                               │
   ▼                       ▼                               │
┌──────────────┐      ┌──────────────────┐                 │
│ PDF Vector   │      │ Episodic Memory  │                 │
│ Index (FAISS)│      │  Index (FAISS)   │                 │
└──────┬───────┘      └────────┬─────────┘                 │
│                       │                                  │
│ (Top 12 Candidates)   │ (Relevant Past Turns)            │
▼                       │                                  │
┌──────────────┐        │                                  │
│ Cross-Encoder│        │                                  │
│  Re-ranking  │        │                                  │
└──────┬───────┘        │                                  │
│ (Top 5 Citations)     │                                  │
└───────────┬───────────┘                                  │
            │                                              │
            └─────────────────────┬────────────────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │      LLM Generation       │
                    │ (Local Llama 3 / Gemini)  │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ DeBERTa-v3 Guardrail      │
                    │ (Sentence-Level Auditing) │
                    └─────────────┬─────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
      [ Score >= 0.45: PASS ]             [ Score < 0.45: FAIL ]
                │                                   │
                ▼                                   ▼
      [ Verified Response ]              [ Auto-Escalation Loop ]
                                          (Escalate / Fallback)

```

---

## ⚡ Key Engineering Features

### 1. Dense Semantic Vector Routing (EBR)

* Replaces fragile regex and error-prone generative zero-shot classifiers with **Embedding-Based Routing (EBR)** using `all-MiniLM-L6-v2`.
* Maps user queries against pre-computed Intent Anchor tensors in <3ms using pure cosine similarity math, reliably routing between **Local (Ollama)** and **Cloud (Gemini Flash-Lite)** while identifying RAG necessity without hallucination.

### 2. Dual-RAG Episodic Memory

* Overcomes LLM context-window limits by treating session chat histories as a secondary vector store.
* Dynamically generates an on-the-fly in-memory FAISS index of historical turns per query, injecting only the 2-3 most semantically relevant interactions into the prompt context.
* Preserves multi-turn state across model switches while maintaining a constant token footprint.

### 3. Two-Stage Retrieval & Re-ranking

* Solves the "Lost in the Middle" phenomenon by decoupling candidate retrieval from final context injection.
* **Stage 1:** FAISS dense vector search retrieves the top 12 candidate snippets.
* **Stage 2:** A Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) re-ranks and filters snippets down to the top 5 highest-density citations.

### 4. Sentence-Level NLI Hallucination Guardrail

* Integrates a dedicated `DeBERTa-v3` Natural Language Inference model to independently audit responses against raw text context.
* Splits generated responses into discrete sentences, evaluates entailment metrics on each statement, and computes a composite faithfulness score.
* Incorporates a **closed-loop recovery mechanism**: responses scoring <0.45 automatically escalate to cloud inference or revert to a deterministic extraction prompt.

### 5. Multi-Threaded State Synchronization

* Configured SQLite connection detachment (`check_same_thread=False`) and auto-commit mode (`isolation_level=None`) to eliminate database locks during asynchronous Streamlit executions.
* Implemented programmatic UI selection snapping to handle newly ingested documents seamlessly without state corruption.

---

## 📂 Repository Structure

```text
Acamate/
├── config/
│   └── config.py               # Environment configuration and API loader
├── core/
│   ├── context_pruner.py       # Cross-Encoder re-ranker
│   ├── faithfulness_checker.py # DeBERTa-v3 hallucination guardrail
│   ├── llm_router.py           # Dense semantic vector routing engine
│   ├── memory_manager.py       # Episodic Memory FAISS manager
│   ├── mia_manager.py          # Global document mindscape summary builder
│   ├── pdf_handler.py          # Document parser, text chunker, and FAISS ingester
│   ├── query_refiner.py        # Conversational query refiner
│   ├── response_generator.py   # Multi-agent orchestrator & auto-escalation loop
│   └── vector_manager.py       # Vector indexing and similarity search wrapper
├── db/
│   └── db_manager.py           # SQLite connection pool and schema manager
├── gui.py                      # Streamlit UI with execution trace inspection
├── .env.example                # Sample environment configuration file
├── .gitignore                  # Git untracked pattern definitions
├── requirements.txt            # Project dependencies
└── README.md                   # System documentation

```

---

## 🚀 Getting Started

### Prerequisites

1. **Python 3.10** or higher.
2. **Ollama** installed with the `llama3.1` model downloaded:
```bash
ollama pull llama3.1

```


3. A **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/?utm_source=gemini).

### Installation & Run

**1. Clone or extract the repository:**

```bash
cd Acamate

```

**2. Create and activate a virtual environment:**

```bash
conda create -n acamate python=3.10 -y
conda activate acamate
pip install -r requirements.txt

```

**3. Configure environment variables:**

* Copy `.env.example` to `.env`.
* Add your Google Gemini API Key to the `.env` file:
```env
GEMINI_API_KEY=your_actual_api_key_here

```



**4. Start Ollama (in a separate terminal):**

```bash
ollama serve

```

**5. Launch the Streamlit application:**

```bash
streamlit run gui.py

```

---

## 🧪 Testing the Pipeline

**General Chat & Vector Routing:**

* Select **🌍 General AI Chat (No PDF)** in the sidebar.
* **Query:** *"Write a quick sort algorithm in C++"* → Evaluates to **LOCAL** (Ollama), bypassing FAISS.
* **Query:** *"Use online model: explain quantum entanglement"* → Routes to **GEMINI** via semantic vector math.

**Episodic Context Retention:**

* State a personal project constraint using Ollama, then switch models:
* **Query:** *"Use online model: what did I say my project was?"* → Context is successfully recalled via episodic memory retrieval.

**Document Ingestion & Execution Trace:**

* Upload an academic paper using the sidebar interface.
* Expand the **🔍 Backend Workflow & Evidence** panel to view real-time intent classification, query refinement, and raw citation chunks as the system processes your document.

```

```