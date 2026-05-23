"""
streamlit_app/pages/4_Semantic_RAG.py
Semantic Contextual RAG demo page.
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
semantic_builder = load_module("semantic", f"{base}/semantic.py")

# Page
st.title("🧠 Semantic Contextual RAG")
st.markdown(semantic_builder.describe())
st.markdown("---")

# Check if processed
if "semantic_ready" not in st.session_state:
    st.warning("⚠️ Please upload and process a PDF first!")
    st.page_link("pages/1_Upload.py", label="Go to Upload Page", icon="📄")
    st.stop()

# How it works
with st.expander("ℹ️ How Semantic RAG Works"):
    st.markdown("""
    ```
    1. Split document into fixed-size chunks
    2. For each chunk → call LLM → generate topic summary
    3. Prepend summary to chunk before embedding
    4. Enriched chunk gets embedded and stored
    5. Query → find semantically enriched chunks → LLM answer

    Fix: Each chunk now explicitly states its topic!
    ```
    """)

    st.markdown("""
    **Example enriched chunk:**
    ```
    [TOPIC: This chunk discusses the mathematical formulation
    of the Contextual Hierarchical Graph used in MGranRAG,
    specifically the node and edge definitions.]

    Given a corpus of passages P, we construct a CHG G = (V, E)...
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
    with st.spinner("Retrieving semantically enriched chunks..."):

        # Retrieve
        retrieved = retrieve_top_k(
            query=question,
            index=st.session_state.semantic_index,
            chunks=st.session_state.semantic_chunks,
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
        st.markdown("### 🧠 Retrieved Chunks (with Semantic Context)")
        st.caption("Each chunk has an AI-generated topic label!")

        for i, chunk in enumerate(retrieved):
            score = chunk.get("retrieval_score", 0)
            color = "green" if score > 0.5 else "orange" if score > 0.3 else "red"
            summary = chunk.get("semantic_summary", "No summary")

            st.markdown(f"**Chunk {i+1}** | Score: :{color}[{score:.3f}]")
            st.info(f"🧠 Topic: {summary}")

            with st.expander("View original chunk"):
                st.text(chunk.get("original_text", chunk["text"])[:300])

            with st.expander("View with semantic context"):
                st.text(chunk["text"][:500])

    with col2:
        st.markdown("### 💬 Answer")
        st.markdown(f"> {answer}")

        avg_score = sum(
            c["retrieval_score"] for c in retrieved
        ) / len(retrieved)

        st.metric("Avg Retrieval Score", f"{avg_score:.3f}")

        st.markdown("**Semantic Advantage:**")
        st.success("✅ Chunks labeled with topic → better semantic matching")

    # Save to session for comparison
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = {}

    st.session_state.comparison_results["semantic"] = {
        "question": question,
        "answer": answer,
        "chunks": retrieved,
        "avg_score": avg_score
    }

    st.info("💡 Results saved! Go to **Compare All** to see side-by-side comparison.")

elif ask_button and not question:
    st.warning("Please enter a question!")