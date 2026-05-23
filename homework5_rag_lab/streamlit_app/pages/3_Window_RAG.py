"""
streamlit_app/pages/3_Window_RAG.py
Window Contextual RAG demo page.
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
window_builder = load_module("window", f"{base}/window.py")

# Page
st.title("🪟 Window Contextual RAG")
st.markdown(window_builder.describe())
st.markdown("---")

# Check if processed
if "window_ready" not in st.session_state:
    st.warning("⚠️ Please upload and process a PDF first!")
    st.page_link("pages/1_Upload.py", label="Go to Upload Page", icon="📄")
    st.stop()

# How it works
with st.expander("ℹ️ How Window RAG Works"):
    st.markdown("""
    ```
    1. Split document into fixed-size chunks
    2. For each chunk, attach neighboring chunks (window)
    3. Embed the CENTER chunk only
    4. At retrieval: return full window (before + center + after)
    5. LLM sees surrounding context!

    Fix: Adds neighboring text so LLM understands flow.
    ```
    """)

    st.markdown("""
    **Example window (window=1):**
    ```
    [Chunk N-1: previous paragraph]
    [MAIN CHUNK N: retrieved chunk]  ← what was searched
    [Chunk N+1: next paragraph]
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
    with st.spinner("Retrieving chunks with window context..."):

        # Retrieve top-k
        retrieved = retrieve_top_k(
            query=question,
            index=st.session_state.window_index,
            chunks=st.session_state.window_chunks,
            model=st.session_state.embedding_model,
            k=st.session_state.get("top_k", 3)
        )

        # Generate answer using window context
        context_parts = []
        for chunk in retrieved:
            context_parts.append(
                chunk.get("window_text", chunk["text"])
            )
        context = "\n\n---\n\n".join(context_parts)
        answer = generate_answer(question, context)

    # Show results
    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🪟 Retrieved Chunks (with Window)")
        st.caption("Each chunk includes its neighbors for better context!")

        for i, chunk in enumerate(retrieved):
            score = chunk.get("retrieval_score", 0)
            color = "green" if score > 0.5 else "orange" if score > 0.3 else "red"
            window_range = chunk.get("window_range", "N/A")

            st.markdown(f"**Chunk {i+1}** | Score: :{color}[{score:.3f}] | Window: {window_range}")

            with st.expander(f"View original chunk"):
                st.text(chunk.get("original_text", chunk["text"])[:300])

            with st.expander(f"View with window context"):
                st.text(chunk.get("window_text", chunk["text"])[:600])

    with col2:
        st.markdown("### 💬 Answer")
        st.markdown(f"> {answer}")

        avg_score = sum(
            c["retrieval_score"] for c in retrieved
        ) / len(retrieved)

        st.metric("Avg Retrieval Score", f"{avg_score:.3f}")

        st.markdown("**Window Advantage:**")
        st.success("✅ LLM sees surrounding text for better understanding")

    # Save to session for comparison
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = {}

    st.session_state.comparison_results["window"] = {
        "question": question,
        "answer": answer,
        "chunks": retrieved,
        "avg_score": avg_score
    }

    st.info("💡 Results saved! Go to **Compare All** to see side-by-side comparison.")

elif ask_button and not question:
    st.warning("Please enter a question!")