"""
streamlit_app/components/chunk_viewer.py
Reusable component to display retrieved chunks.
"""

import streamlit as st


def show_chunk(chunk: dict, rank: int, rag_type: str = ""):
    """Display a single retrieved chunk as a card."""

    score = chunk.get("retrieval_score", 0)
    section = chunk.get("section", "Unknown")
    text = chunk.get("text", "")
    original = chunk.get("original_text", text)
    context_type = chunk.get("context_type", "traditional")

    # Color based on score
    if score > 0.5:
        border_color = "#00c853"  # green
    elif score > 0.3:
        border_color = "#ff6d00"  # orange
    else:
        border_color = "#d50000"  # red

    st.markdown(f"""
        <div style='
            border-left: 4px solid {border_color};
            padding: 10px 15px;
            margin: 8px 0;
            background-color: #1e1e1e;
            border-radius: 4px;
        '>
            <b>Chunk #{rank}</b> &nbsp;|&nbsp;
            Score: <b>{score:.3f}</b> &nbsp;|&nbsp;
            Section: <b>{section}</b>
        </div>
    """, unsafe_allow_html=True)

    # Show context type specific info
    if context_type == "window" and "window_range" in chunk:
        st.caption(f"🪟 Window range: chunks {chunk['window_range']}")

    if context_type == "semantic" and "semantic_summary" in chunk:
        st.caption(f"🧠 Topic: {chunk['semantic_summary']}")

    if context_type == "hierarchical" and "hierarchy_path" in chunk:
        st.caption(f"🏛️ {chunk['hierarchy_path'][:100]}...")

    # Show text
    with st.expander(f"📄 View chunk text", expanded=rank == 1):
        if context_type != "traditional" and original != text:
            st.markdown("**Original chunk:**")
            st.text(original[:400])
            st.markdown("**With context added:**")
            st.text(text[:400])
        else:
            st.text(text[:400])


def show_chunks_panel(
    chunks: list,
    title: str = "Retrieved Chunks",
    rag_type: str = ""
):
    """Display a panel of retrieved chunks."""
    st.markdown(f"#### {title}")

    if not chunks:
        st.warning("No chunks retrieved")
        return

    for i, chunk in enumerate(chunks):
        show_chunk(chunk, rank=i+1, rag_type=rag_type)