"""
streamlit_app/pages/6_Compare_All.py
Side-by-side comparison of all 4 RAG approaches.
"""

import streamlit as st
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

from utils.retriever import retrieve_top_k, format_context
from utils.llm import generate_answer
import plotly.graph_objects as go

# Page
st.title("📊 Compare All RAG Approaches")
st.markdown("Ask one question and see all 4 RAG types answer side by side!")
st.markdown("---")

# Check if processed
if "traditional_ready" not in st.session_state:
    st.warning("⚠️ Please upload and process a PDF first!")
    st.page_link("pages/1_Upload.py", label="Go to Upload Page", icon="📄")
    st.stop()

# Sample questions
sample_questions = [
    "What problem does MGranRAG solve?",
    "How is the Contextual Hierarchical Graph constructed?",
    "What datasets were used in the experiments?",
    "How does MGranRAG compare to HippoRAG 2?",
    "What are the limitations of MGranRAG?"
]

# Question input
st.markdown("### Ask a Question")
selected = st.selectbox("Sample questions:", [""] + sample_questions)
question = st.text_input(
    "Or type your own:",
    value=selected,
    placeholder="Type your question here..."
)

col1, col2 = st.columns([1, 4])
with col1:
    compare_button = st.button("🔍 Compare All", type="primary")

if compare_button and question:

    results = {}

    # Progress bar
    progress = st.progress(0)
    status = st.empty()

    # Traditional RAG
    status.text("📦 Running Traditional RAG...")
    progress.progress(10)
    t_retrieved = retrieve_top_k(
        query=question,
        index=st.session_state.traditional_index,
        chunks=st.session_state.traditional_chunks,
        model=st.session_state.embedding_model,
        k=st.session_state.get("top_k", 3)
    )
    t_context = format_context(t_retrieved)
    t_answer = generate_answer(question, t_context)
    t_score = sum(c["retrieval_score"] for c in t_retrieved) / len(t_retrieved)
    results["traditional"] = {
        "answer": t_answer,
        "chunks": t_retrieved,
        "score": t_score
    }
    progress.progress(30)

    # Window RAG
    status.text("🪟 Running Window RAG...")
    w_retrieved = retrieve_top_k(
        query=question,
        index=st.session_state.window_index,
        chunks=st.session_state.window_chunks,
        model=st.session_state.embedding_model,
        k=st.session_state.get("top_k", 3)
    )
    w_context = "\n\n---\n\n".join([
        c.get("window_text", c["text"]) for c in w_retrieved
    ])
    w_answer = generate_answer(question, w_context)
    w_score = sum(c["retrieval_score"] for c in w_retrieved) / len(w_retrieved)
    results["window"] = {
        "answer": w_answer,
        "chunks": w_retrieved,
        "score": w_score
    }
    progress.progress(55)

    # Semantic RAG
    status.text("🧠 Running Semantic RAG...")
    s_retrieved = retrieve_top_k(
        query=question,
        index=st.session_state.semantic_index,
        chunks=st.session_state.semantic_chunks,
        model=st.session_state.embedding_model,
        k=st.session_state.get("top_k", 3)
    )
    s_context = format_context(s_retrieved)
    s_answer = generate_answer(question, s_context)
    s_score = sum(c["retrieval_score"] for c in s_retrieved) / len(s_retrieved)
    results["semantic"] = {
        "answer": s_answer,
        "chunks": s_retrieved,
        "score": s_score
    }
    progress.progress(80)

    # Hierarchical RAG
    status.text("🏛️ Running Hierarchical RAG...")
    h_retrieved = retrieve_top_k(
        query=question,
        index=st.session_state.hierarchical_index,
        chunks=st.session_state.hierarchical_chunks,
        model=st.session_state.embedding_model,
        k=st.session_state.get("top_k", 3)
    )
    h_context = format_context(h_retrieved)
    h_answer = generate_answer(question, h_context)
    h_score = sum(c["retrieval_score"] for c in h_retrieved) / len(h_retrieved)
    results["hierarchical"] = {
        "answer": h_answer,
        "chunks": h_retrieved,
        "score": h_score
    }
    progress.progress(100)
    status.text("✅ All done!")

    st.markdown("---")

    # Score comparison chart
    st.markdown("### 📊 Retrieval Score Comparison")

    colors = ["#757575", "#1976d2", "#388e3c", "#f57c00"]
    rag_types = ["Traditional", "Window", "Semantic", "Hierarchical"]
    scores = [
        results["traditional"]["score"],
        results["window"]["score"],
        results["semantic"]["score"],
        results["hierarchical"]["score"]
    ]

    fig = go.Figure(go.Bar(
        x=rag_types,
        y=scores,
        marker_color=colors,
        text=[f"{s:.3f}" for s in scores],
        textposition="outside"
    ))
    fig.update_layout(
        yaxis_title="Avg Retrieval Score",
        yaxis_range=[0, 1],
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Side by side answers
    st.markdown(f"### 💬 Answers Side by Side")
    st.markdown(f"**Question:** {question}")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### 📦 Traditional")
        st.caption(f"Score: {results['traditional']['score']:.3f}")
        st.markdown(results["traditional"]["answer"])

    with col2:
        st.markdown("#### 🪟 Window")
        st.caption(f"Score: {results['window']['score']:.3f}")
        st.markdown(results["window"]["answer"])

    with col3:
        st.markdown("#### 🧠 Semantic")
        st.caption(f"Score: {results['semantic']['score']:.3f}")
        st.markdown(results["semantic"]["answer"])

    with col4:
        st.markdown("#### 🏛️ Hierarchical")
        st.caption(f"Score: {results['hierarchical']['score']:.3f}")
        st.markdown(results["hierarchical"]["answer"])

    st.markdown("---")

    # Retrieved chunks comparison
    st.markdown("### 📦 Retrieved Chunks Comparison")
    st.caption("See what each RAG type actually retrieved for this question")

    tabs = st.tabs([
        "📦 Traditional",
        "🪟 Window",
        "🧠 Semantic",
        "🏛️ Hierarchical"
    ])

    rag_keys = ["traditional", "window", "semantic", "hierarchical"]

    for tab, key in zip(tabs, rag_keys):
        with tab:
            chunks = results[key]["chunks"]
            for i, chunk in enumerate(chunks):
                score = chunk.get("retrieval_score", 0)
                section = chunk.get("section", "Unknown")
                st.markdown(f"**Chunk {i+1}** | Score: {score:.3f} | Section: {section}")

                # Show context type specific info
                if key == "window" and "window_range" in chunk:
                    st.caption(f"Window: {chunk['window_range']}")
                if key == "semantic" and "semantic_summary" in chunk:
                    st.caption(f"Topic: {chunk['semantic_summary']}")
                if key == "hierarchical" and "hierarchy_path" in chunk:
                    st.caption(f"Path: {chunk['hierarchy_path'][:100]}")

                with st.expander("View text"):
                    original = chunk.get("original_text", chunk["text"])
                    st.text(original[:400])

    # Key observations
    st.markdown("---")
    st.markdown("### 💡 Key Observations")

    best_rag = max(results.items(), key=lambda x: x[1]["score"])
    worst_rag = min(results.items(), key=lambda x: x[1]["score"])

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"✅ Best retrieval: **{best_rag[0].title()}** (score: {best_rag[1]['score']:.3f})")
    with col2:
        st.warning(f"⚠️ Lowest retrieval: **{worst_rag[0].title()}** (score: {worst_rag[1]['score']:.3f})")

elif compare_button and not question:
    st.warning("Please enter a question!")

# Show saved results from individual pages
elif "comparison_results" in st.session_state and st.session_state.comparison_results:
    st.markdown("### 📋 Results from Individual Pages")
    saved = st.session_state.comparison_results

    for rag_type, result in saved.items():
        st.markdown(f"**{rag_type.title()} RAG** — Q: {result['question']}")
        st.markdown(f"> {result['answer'][:200]}...")
        st.markdown("---")