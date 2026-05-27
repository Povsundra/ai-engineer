# 🔬 RAG Comparison Lab

A hands-on demo comparing **4 Contextual RAG approaches** on a research paper.
Built as a Final Project for CS 695 · AI Engineering.

---

## 📌 What is This?

This lab demonstrates how adding **context** to document chunks dramatically improves
Retrieval-Augmented Generation (RAG) quality. We compare 4 approaches on the same
document and same questions so the difference is clear.

**Paper used for demo:**
> Iterative Multi-Granular RAG with Contextual Hierarchical Graph (MGranRAG)
> Yanli Hu, Teng Liu et al. — AAAI-26 (2026)

---

## 🛠️ Tech Stack

| Step | Tool | Mode |
|---|---|---|
| PDF Extraction | PyMuPDF (fitz) | Auto |
| Chunking | LangChain RecursiveTextSplitter | Auto |
| Section Detection | Rule-based regex patterns | Semi-auto |
| Context Generation | OpenRouter (Gemini 3.5 Flash) | Auto |
| Embedding | Sentence-Transformers (all-MiniLM-L6-v2) | Auto |
| Vector Store | FAISS (IndexFlatIP) | Auto |
| QA Generation | OpenRouter (Gemini 3.5 Flash) | Auto |
| Demo UI | Streamlit | Auto |

---

## 🔬 4 RAG Types Compared

| # | Type | What is Added | Chunks | LLM Calls at Index |
|---|---|---|---|---|
| 1 | **Traditional** | Nothing — baseline | 100 | 0 |
| 2 | **Window** | Neighbor chunks (±1) | 100 | 0 |
| 3 | **Semantic** | LLM-generated topic summary | 100 | 100 |
| 4 | **Hierarchical** | Document structure path | 74 | 74 |

### How Each Type Works

**📦 Traditional RAG**
```
chunk → embed → store → retrieve → answer
No context added. Chunks lose their location in the document.
```

**🪟 Window RAG**
```
[chunk N-1] + [chunk N] + [chunk N+1] → LLM sees surrounding text
Embedding done on center chunk only. Neighbors added at answer time.
```

**🧠 Semantic RAG**
```
LLM generates topic summary → prepend to chunk → embed enriched chunk
Each chunk explicitly states its topic before being embedded.
```

**🏛️ Hierarchical RAG**
```
Paper → Section → Subsection → chunk
LLM generates structure path → prepend to chunk → embed enriched chunk
Each chunk knows exactly where it is in the document.
```

---

## 📁 Project Structure

```
rag_lab/
├── data/
│   └── paper.pdf                  ← input document
├── utils/
│   ├── pdf_extractor.py           ← PDF → text + sections
│   ├── chunker.py                 ← text → chunks
│   ├── embedder.py                ← chunks → vectors + FAISS
│   ├── retriever.py               ← query → top-k chunks
│   └── llm.py                     ← OpenRouter API calls
├── context_builders/
│   ├── traditional.py             ← no context (baseline)
│   ├── window.py                  ← neighbor chunks
│   ├── semantic.py                ← LLM topic summary
│   └── hierarchical.py            ← section path prefix
├── notebooks/
│   ├── 00_setup_test.ipynb        ← verify environment
│   └── 01_extract_chunk.ipynb     ← see chunking output
├── streamlit_app/
│   ├── app.py                     ← home page
│   ├── pages/
│   │   ├── 1_Upload.py            ← upload + process PDF
│   │   ├── 2_Traditional_RAG.py   ← traditional demo
│   │   ├── 3_Window_RAG.py        ← window demo
│   │   ├── 4_Semantic_RAG.py      ← semantic demo
│   │   ├── 5_Hierarchical_RAG.py  ← hierarchical demo
│   │   └── 6_Compare_All.py       ← side-by-side comparison
│   └── components/
│       ├── chunk_viewer.py         ← chunk display cards
│       ├── answer_card.py          ← answer display cards
│       └── score_chart.py          ← plotly charts
└── outputs/                        ← saved FAISS indexes (auto-generated)
```

---

## ⚙️ Setup

### 1. Clone and Install
```bash
git clone https://github.com/yourusername/rag-lab.git
cd rag_lab
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

### 2. Configure API Key
Create `.env` file:
```bash
OPENROUTER_API_KEY=sk-or-your-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=google/gemini-3.5-flash
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=3
WINDOW_SIZE=1
```

### 3. Run Demo App
```bash
cd streamlit_app
streamlit run app.py
```

### 4. Run Notebooks (Optional)
```bash
jupyter notebook
```

---

## 🎮 How to Use the Demo

```
Step 1: Go to Upload & Process page
        → Upload PDF or use demo paper
        → Click Process (takes 3-5 min first time)
        → Indexes saved automatically

Step 2: Go to each RAG page
        → Ask a question
        → See retrieved chunks + answer

Step 3: Go to Compare ALL
        → Ask same question
        → See all 4 answers side by side
        → See score comparison chart

Next time you open the app:
        → Indexes load automatically (no re-upload needed!)
```

---

## ❓ Test Questions Used

| # | Question | Tests |
|---|---|---|
| Q1 | What problem does MGranRAG solve? | Abstract/Introduction retrieval |
| Q2 | How is the Contextual Hierarchical Graph constructed? | Methodology retrieval |
| Q3 | What datasets were used in the experiments? | Experiments retrieval |
| Q4 | How does MGranRAG compare to HippoRAG 2? | Results retrieval |
| Q5 | What are the limitations of MGranRAG? | Conclusion retrieval |

---

## 📊 Evaluation Rubric

Each RAG type is scored per question on 4 metrics:

| Metric | Description | Score |
|---|---|---|
| **Context Quality** | Does retrieved chunk make sense alone? | 1-5 |
| **Answer Accuracy** | Is the answer factually correct? | 1-5 |
| **Answer Completeness** | Does it fully answer the question? | 1-5 |
| **Hallucination** | Does it add info NOT in the paper? | 1-5 |

### Scoring Guide

**Context Quality**
```
1 = Math/numbers table retrieved (wrong content)
2 = Wrong section retrieved
3 = Related but incomplete
4 = Correct content retrieved
5 = Correct content + knows exact location (Hierarchical)
```

**Hallucination**
```
1 = Major invented facts
2 = Some invented details
3 = Minor extrapolation
4 = Mostly grounded in chunks
5 = 100% from retrieved chunks only
```

---

## 📈 Actual Demo Results

### Retrieval Scores (Q1: "What problem does MGranRAG solve?")

| RAG Type | Retrieval Score | Answer Quality |
|---|---|---|
| Traditional | 0.402 | Retrieved Results section — missed core problem |
| Window | 0.402 | Same chunks + neighbors — similar answer |
| Semantic | 0.440 ⬆️ | Best score — still somewhat vague |
| Hierarchical | 0.370 | Lowest score — BUT best answer ✅ |

### Estimated Rubric Scores (All 5 Questions)

| Question | Traditional | Window | Semantic | Hierarchical |
|---|---|---|---|---|
| Q1: Problem? | 2/5 | 2/5 | 3/5 | **5/5** |
| Q2: CHG construction? | 3/5 | 3/5 | 4/5 | **5/5** |
| Q3: Datasets? | 1/5 | 2/5 | **4/5** | 4/5 |
| Q4: vs HippoRAG 2? | 2/5 | 3/5 | 3/5 | **4/5** |
| Q5: Limitations? | 2/5 | **4/5** | 3/5 | 4/5 |
| **Total** | 10/25 | 14/25 | 17/25 | **22/25** |
| **Score %** | 40% | 56% | 68% | **88%** |

---

## 🏆 Key Findings

### Finding 1 — Higher Retrieval Score ≠ Better Answer
```
Semantic RAG:     score 0.440 (highest) → vague answer
Hierarchical RAG: score 0.370 (lowest)  → best answer ✅

Lesson: Context quality matters MORE than retrieval score.
        A chunk that knows its section always wins.
```

### Finding 2 — Traditional RAG Retrieves Wrong Sections
```
Q: "What problem does MGranRAG solve?"
Traditional retrieved: Results section (performance numbers)
Should have retrieved: Introduction (problem statement)

Fix: Hierarchical RAG uses section path to guide retrieval.
```

### Finding 3 — Window RAG Same Score as Traditional
```
Window score = Traditional score (both 0.402)
Reason: Embedding uses CENTER chunk only.
        Neighbors help answer quality but NOT retrieval score.
```

### Finding 4 — Each Type Wins on Different Questions
```
Structure questions  → 🏛️ Hierarchical wins
Topic questions      → 🧠 Semantic wins
Flow questions       → 🪟 Window wins
Simple questions     → 📦 All similar
```

---

## 🚀 Future Work & Improvements

### 1. LLM-as-Judge Evaluation (Not Yet Implemented)
```
Current: Manual scoring by observation
Planned: Automated LLM scoring pipeline

Implementation:
- Generate ground truth Q&A pairs from paper
- Use LLM to score each answer 1-5
- Metrics: Accuracy, Completeness, Faithfulness
- Compare all 4 RAG types automatically

Prompt example:
"Rate this answer 1-5 for accuracy, completeness,
 and hallucination. Return JSON scores."
```

### 2. Hybrid Search (BM25 + Semantic)
```
Current: Dense vector search only (FAISS)
Planned: Combine keyword + semantic search

Why: Some queries match better with exact keywords
     Some queries match better with semantic meaning
     Hybrid gives best of both worlds

Tool: Combine BM25 (sparse) + FAISS (dense)
      Use Reciprocal Rank Fusion (RRF) to merge results
```

### 3. Reranking Layer
```
Current: Top-k by vector similarity only
Planned: Add cross-encoder reranker after retrieval

Why: Reranker is more accurate than cosine similarity
     Fixes cases where score ≠ relevance

Tool: Cohere Rerank API or local cross-encoder model
```

### 4. Multi-Document Support
```
Current: One paper at a time
Planned: Upload multiple papers simultaneously

Why: Research often spans multiple papers
     Compare findings across documents

Implementation:
- Store each paper in separate namespace
- Query across all papers
- Cite which paper each answer comes from
```

### 5. Query Rewriting
```
Current: Raw user query sent directly to retriever
Planned: LLM rewrites query before retrieval

Why: User queries often don't match document style
     "Tell me about the graph thing" →
     "Explain the Contextual Hierarchical Graph structure"

Types:
- Query expansion (multiple versions)
- HyDE (hypothetical document embeddings)
- Multi-query retrieval
```

### 6. Evaluation Dataset
```
Current: 5 manually designed questions
Planned: Synthetic Q&A dataset from paper

Steps:
1. Extract key facts from each section
2. LLM generates 20+ questions per section
3. LLM-as-judge filters good questions
4. Use as standard benchmark
5. Score all 4 RAG types automatically
```

### 7. Streaming Answers
```
Current: Wait for full answer then display
Planned: Stream tokens as they generate

Why: Better UX for long answers
     Feels more responsive

Tool: OpenRouter streaming API
```

### 8. Cost Tracking
```
Current: No cost visibility
Planned: Show API cost per query

Why: Important for production deployment
     Students can see cost difference between types

Display: "This query cost $0.002"
         "Semantic indexing cost $0.15 total"
```

---

## 📋 Known Limitations

```
1. Session-based: Indexes lost on browser refresh
   (Fixed with auto-load from disk)

2. Processing time: 3-5 minutes first time
   (Semantic + Hierarchical require many LLM calls)

3. Math chunks: PDF tables extracted as raw numbers
   (Affects Traditional RAG retrieval quality)

4. Single paper: Only one paper loaded at a time
   (No multi-document support yet)

5. No ground truth: Rubric scores are estimated
   (LLM-as-judge not yet implemented)

6. English only: Embedding model optimized for English
   (Other languages may have lower quality)
```

---

## 🤖 AI Use Disclosure

This project was built with AI assistance following course requirements:

| Component | AI Tool Used | How Validated |
|---|---|---|
| Code generation | Claude (Anthropic) | Tested each function manually |
| Context generation | Gemini 3.5 Flash (OpenRouter) | Reviewed output quality |
| Embedding | all-MiniLM-L6-v2 (local) | Standard open-source model |
| Answer generation | Gemini 3.5 Flash (OpenRouter) | Compared against paper content |

All AI-generated code was reviewed, tested, and understood before use.

---

## 📚 References

- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Anthropic (2024). Contextual Retrieval. https://www.anthropic.com/news/contextual-retrieval
- Hu et al. (2026). Iterative Multi-Granular RAG with Contextual Hierarchical Graph. AAAI-26.
- Pinecone. Chunking Strategies. https://www.pinecone.io/learn/chunking-strategies/

---

## 👤 Author

**CS 695 · AI Engineering — Assignment Project**
Track B: RAG Application