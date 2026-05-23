"""
streamlit_app/pages/5_Hierarchical_RAG.py
Hierarchical Contextual RAG demo page.
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
hierarchical_builder = load_module("hierarchical", f"{base}/hierarchical.py")

# Page
st.title("🏛️ Hierarchical Contextual RAG")
st.markdown(hierarchical_builder.describe())
st.markdown("---")

# Check if processed
if "hierarchical_ready" not in st.session_state:
    st.warning("⚠️ Please upload and process a PDF first!")
    st.page_link("pages/1_Upload.py", label="Go to Upload Page", icon="📄")
    st.stop()

# How it works
with st.expander("ℹ️ How Hierarchical RAG Works"):
    st.markdown("""
    ```
    1. Split document into section-aware chunks
    2. For each chunk → LLM generates hierarchy context:
       Paper → Section → Subsection → Content
    3. Prepend hierarchy path to chunk before embedding
    4. Query → find structure-aware chunks → LLM answer

    Fix: Each chunk knows EXACTLY where it is in the document!
    ```
    """)

    st.markdown("""
    **Example hierarchy context:**
    ```
    [LOCATION:
    Paper: Iterative Multi-Granular RAG with CHG
    Section: Preliminaries
    Context: From MGranRAG: Preliminaries section.
             This chunk defines the CHG graph structure.]

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
    with st.spinner("Retrieving hierarchically enriched chunks..."):

        # Retrieve
        retrieved = retrieve_top_k(
            query=question,
            index=st.session_state.hierarchical_index,
            chunks=st.session_state.hierarchical_chunks,
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
        st.markdown("### 🏛️ Retrieved Chunks (with Hierarchy)")
        st.caption("Each chunk knows its exact position in the document!")

        for i, chunk in enumerate(retrieved):
            score = chunk.get("retrieval_score", 0)
            color = "green" if score > 0.5 else "orange" if score > 0.3 else "red"
            section = chunk.get("section", "Unknown")
            hier_path = chunk.get("hierarchy_path", "No hierarchy")

            st.markdown(f"**Chunk {i+1}** | Score: :{color}[{score:.3f}] | Section: **{section}**")
            st.success(f"🏛️ {hier_path[:150]}...")

            with st.expander("View original chunk"):
                st.text(chunk.get("original_text", chunk["text"])[:300])

            with st.expander("View with hierarchy context"):
                st.text(chunk["text"][:600])

    with col2:
        st.markdown("### 💬 Answer")
        st.markdown(f"> {answer}")

        avg_score = sum(
            c["retrieval_score"] for c in retrieved
        ) / len(retrieved)

        st.metric("Avg Retrieval Score", f"{avg_score:.3f}")

        st.markdown("**Hierarchy Advantage:**")
        st.success("✅ Chunks have full document structure path")
        st.success("✅ Section-aware retrieval finds right content")

    # Save to session for comparison
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = {}

    st.session_state.comparison_results["hierarchical"] = {
        "question": question,
        "answer": answer,
        "chunks": retrieved,
        "avg_score": avg_score
    }

    st.info("💡 Results saved! Go to **Compare All** to see side-by-side comparison.")

elif ask_button and not question:
    st.warning("Please enter a question!")