"""
streamlit_app/pages/2_Traditional_RAG.py
Traditional RAG demo page.
"""

import streamlit as st
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

from utils.retriever import retrieve_top_k, format_context
from utils.llm import generate_answer
import importlib.util

def load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base = os.path.join(os.path.dirname(__file__), '../../context_builders')
traditional = load_module("traditional", f"{base}/traditional.py")

# Page
st.title("📦 Traditional RAG")
st.markdown(traditional.describe())
st.markdown("---")

# Check if processed
if "traditional_ready" not in st.session_state:
    st.warning("⚠️ Please upload and process a PDF first!")
    st.page_link("pages/1_Upload.py", label="Go to Upload Page", icon="📄")
    st.stop()

# How it works
with st.expander("ℹ️ How Traditional RAG Works"):
    st.markdown("""
    ```
    1. Split document into fixed-size chunks
    2. Embed each chunk as-is (no context added)
    3. Store in FAISS vector database
    4. Query → embed → find similar chunks → LLM answer
    
    Problem: Chunks have NO context about where they came from!
    ```
    """)

st.markdown("---")

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
    ask_button = st.button("🔍 Ask", type="primary")

if ask_button and question:
    with st.spinner("Retrieving chunks and generating answer..."):

        # Retrieve
        retrieved = retrieve_top_k(
            query=question,
            index=st.session_state.traditional_index,
            chunks=st.session_state.traditional_chunks,
            model=st.session_state.embedding_model,
            k=st.session_state.get("top_k", 3)
        )

        # Generate answer
        context = format_context(retrieved)
        answer = generate_answer(question, context)

    # Show results
    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📦 Retrieved Chunks")
        st.caption("These are the raw chunks retrieved — notice they have NO section info!")

        for i, chunk in enumerate(retrieved):
            score = chunk.get("retrieval_score", 0)
            color = "green" if score > 0.5 else "orange" if score > 0.3 else "red"

            st.markdown(f"**Chunk {i+1}** | Score: :{color}[{score:.3f}]")
            st.caption(f"Section: {chunk.get('section', '❌ UNKNOWN')}")

            with st.expander(f"View chunk text"):
                st.text(chunk["text"][:500])

    with col2:
        st.markdown("### 💬 Answer")
        st.markdown(f"> {answer}")

        # Avg score
        avg_score = sum(
            c["retrieval_score"] for c in retrieved
        ) / len(retrieved)

        st.metric("Avg Retrieval Score", f"{avg_score:.3f}")

        if avg_score < 0.3:
            st.warning("⚠️ Low scores — chunks may not be relevant")
        elif avg_score < 0.5:
            st.info("ℹ️ Moderate scores — partial relevance")
        else:
            st.success("✅ Good scores — relevant chunks found")

    # Save to session for comparison
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = {}

    st.session_state.comparison_results["traditional"] = {
        "question": question,
        "answer": answer,
        "chunks": retrieved,
        "avg_score": avg_score
    }

    st.info("💡 Results saved! Go to **Compare All** to see side-by-side comparison.")

elif ask_button and not question:
    st.warning("Please enter a question!")