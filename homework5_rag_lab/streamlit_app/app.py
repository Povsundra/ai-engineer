"""
streamlit_app/app.py
Main entry point for RAG Comparison Lab demo.
Run: streamlit run app.py
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="RAG Comparison Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page
st.title("🔬 RAG Comparison Lab")
st.subheader("Compare 4 RAG approaches on a research paper")

st.markdown("---")

# Overview
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ## What is this?
    This lab compares **4 RAG approaches** on the same document
    and the same questions so you can clearly see the difference
    in retrieval quality and answer accuracy.
    """)

with col2:
    st.markdown("""
    ## 4 RAG Types
    | # | Type | Key Idea |
    |---|---|---|
    | 1 | Traditional | Raw chunks, no context |
    | 2 | Window | Add neighbor chunks |
    | 3 | Semantic | Add topic summary |
    | 4 | Hierarchical | Add section path |
    """)

st.markdown("---")

# How to use
st.markdown("## How to Use")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    **Step 1**
    
    📄 Go to **Upload & Process**
    
    Upload your PDF and process it
    to build all 4 RAG indexes.
    """)

with col2:
    st.info("""
    **Step 2**
    
    🔍 Go to each **RAG page**
    
    Ask questions and see how
    each approach retrieves chunks.
    """)

with col3:
    st.info("""
    **Step 3**
    
    📊 Go to **Compare All**
    
    See all 4 answers side by side
    and compare quality scores.
    """)

st.markdown("---")

# Status check
st.markdown("## System Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if "pdf_processed" in st.session_state and st.session_state.pdf_processed:
        st.success("✅ PDF Processed")
    else:
        st.warning("⏳ PDF Not Processed")

with col2:
    if "traditional_ready" in st.session_state and st.session_state.traditional_ready:
        st.success("✅ Traditional Ready")
    else:
        st.warning("⏳ Traditional Not Built")

with col3:
    if "window_ready" in st.session_state and st.session_state.window_ready:
        st.success("✅ Window Ready")
    else:
        st.warning("⏳ Window Not Built")

with col4:
    if "semantic_ready" in st.session_state and st.session_state.semantic_ready:
        st.success("✅ Semantic Ready")
    else:
        st.warning("⏳ Semantic Not Built")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
    RAG Comparison Lab | AI Engineering Final Project
    </div>
""", unsafe_allow_html=True)