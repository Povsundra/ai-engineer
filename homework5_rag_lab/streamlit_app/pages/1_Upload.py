"""
streamlit_app/pages/1_Upload.py
Upload and process PDF — builds all 4 RAG indexes.
Auto-loads existing indexes on startup if available.
Shows paper info and sample questions when ready.
"""

import streamlit as st
import os
import sys
import tempfile
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

from utils.pdf_extractor import extract_full_text, extract_sections, get_paper_metadata
from utils.chunker import fixed_size_chunk, section_aware_chunk
from utils.embedder import load_embedding_model, embed_chunks, build_faiss_index, save_index, load_index
from utils.llm import generate_semantic_context, generate_hierarchical_context
import importlib.util


def load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = os.path.join(os.path.dirname(__file__), '../../context_builders')
traditional = load_module("traditional", f"{base}/traditional.py")
window = load_module("window", f"{base}/window.py")
semantic = load_module("semantic", f"{base}/semantic.py")
hierarchical = load_module("hierarchical", f"{base}/hierarchical.py")

outputs_dir = "outputs"


def check_indexes_exist() -> bool:
    """Check if all 4 RAG indexes exist on disk."""
    required = [
        "traditional_index.faiss", "traditional_chunks.json",
        "window_index.faiss", "window_chunks.json",
        "semantic_index.faiss", "semantic_chunks.json",
        "hierarchical_index.faiss", "hierarchical_chunks.json",
    ]
    return all(os.path.exists(os.path.join(outputs_dir, f)) for f in required)


def load_paper_meta() -> dict:
    """Load saved paper metadata from disk."""
    meta_path = os.path.join(outputs_dir, "paper_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_chunk_stats() -> dict:
    """Load chunk statistics from saved files."""
    stats = {}
    for rag_type in ["traditional", "window", "semantic", "hierarchical"]:
        path = os.path.join(outputs_dir, f"{rag_type}_chunks.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            stats[rag_type] = len(chunks)
    return stats


def auto_load_indexes():
    """Auto-load all 4 RAG indexes from disk into session state."""
    try:
        if "embedding_model" not in st.session_state:
            with st.spinner("Loading embedding model..."):
                st.session_state.embedding_model = load_embedding_model()

        t_index, t_chunks = load_index(outputs_dir, "traditional")
        st.session_state.traditional_index = t_index
        st.session_state.traditional_chunks = t_chunks
        st.session_state.traditional_ready = True

        w_index, w_chunks = load_index(outputs_dir, "window")
        st.session_state.window_index = w_index
        st.session_state.window_chunks = w_chunks
        st.session_state.window_ready = True

        s_index, s_chunks = load_index(outputs_dir, "semantic")
        st.session_state.semantic_index = s_index
        st.session_state.semantic_chunks = s_chunks
        st.session_state.semantic_ready = True

        h_index, h_chunks = load_index(outputs_dir, "hierarchical")
        st.session_state.hierarchical_index = h_index
        st.session_state.hierarchical_chunks = h_chunks
        st.session_state.hierarchical_ready = True

        meta = load_paper_meta()
        st.session_state.paper_title = meta.get("title", "Unknown Paper")
        st.session_state.num_pages = meta.get("num_pages", 0)
        st.session_state.paper_filename = meta.get("filename", "unknown.pdf")
        st.session_state.pdf_processed = True
        return True

    except Exception as e:
        st.warning(f"Could not auto-load: {e}")
        return False


# ── Auto-load on startup ──
if "pdf_processed" not in st.session_state and check_indexes_exist():
    with st.spinner("🔄 Auto-loading saved indexes..."):
        auto_load_indexes()

# ── Page Title ──
st.title("📄 Upload & Process Document")
st.markdown("---")

# ── Show Paper Info if Loaded ──
if "pdf_processed" in st.session_state and st.session_state.pdf_processed:

    paper_title = st.session_state.get("paper_title", "Unknown")
    num_pages = st.session_state.get("num_pages", 0)
    filename = st.session_state.get("paper_filename", "unknown.pdf")
    chunk_stats = load_chunk_stats()

    # Paper info banner
    st.success("✅ Indexes loaded and ready!")

    st.markdown("### 📋 Current Paper")
    st.markdown(f"""
    <div style='
        background-color: #1e1e2e;
        border-left: 4px solid #4caf50;
        padding: 15px 20px;
        border-radius: 6px;
        margin-bottom: 10px;
    '>
        <div style='font-size: 1.1em; font-weight: bold; color: white;'>
            📄 {paper_title[:80]}...
        </div>
        <div style='color: #aaaaaa; margin-top: 5px;'>
            File: {filename} &nbsp;|&nbsp; Pages: {num_pages}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Index stats
    st.markdown("### 📊 Index Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📦 Traditional",
            f"{chunk_stats.get('traditional', 0)} chunks",
            "Ready ✅"
        )
    with col2:
        st.metric(
            "🪟 Window",
            f"{chunk_stats.get('window', 0)} chunks",
            "Ready ✅"
        )
    with col3:
        st.metric(
            "🧠 Semantic",
            f"{chunk_stats.get('semantic', 0)} chunks",
            "Ready ✅"
        )
    with col4:
        st.metric(
            "🏛️ Hierarchical",
            f"{chunk_stats.get('hierarchical', 0)} chunks",
            "Ready ✅"
        )

    st.markdown("---")

    # Sample questions to prepare
    st.markdown("### 💡 Suggested Questions to Try")
    st.caption("Click any question to copy it, then paste in the RAG pages!")

    questions = [
        {
            "q": "What problem does MGranRAG solve?",
            "tip": "Tests Abstract/Introduction retrieval",
            "best": "Hierarchical"
        },
        {
            "q": "How is the Contextual Hierarchical Graph constructed?",
            "tip": "Tests Methodology section retrieval",
            "best": "Hierarchical"
        },
        {
            "q": "What datasets were used in the experiments?",
            "tip": "Tests Experiments section retrieval",
            "best": "Semantic"
        },
        {
            "q": "How does MGranRAG compare to HippoRAG 2?",
            "tip": "Tests Results/comparison retrieval",
            "best": "Window"
        },
        {
            "q": "What are the limitations of MGranRAG?",
            "tip": "Tests Conclusion section retrieval",
            "best": "Hierarchical"
        },
    ]

    for i, item in enumerate(questions):
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            st.markdown(f"**Q{i+1}:** {item['q']}")

        with col2:
            st.caption(f"💡 {item['tip']}")

        with col3:
            st.caption(f"🏆 Best: {item['best']}")

        st.markdown("---")

    # Navigation hint
    st.markdown("### 🗺️ Where to Go Next")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info("**Step 1**\n\n📦 Go to\n**Traditional RAG**\nask a question")

    with col2:
        st.info("**Step 2**\n\n🪟 Go to\n**Window RAG**\nask same question")

    with col3:
        st.info("**Step 3**\n\n🧠 Go to\n**Semantic RAG**\nask same question")

    with col4:
        st.info("**Step 4**\n\n📊 Go to\n**Compare ALL**\nsee differences!")

    st.markdown("---")
    st.markdown("### 🔄 Upload Different Paper")
    st.caption("Want to try a different paper? Upload below — this will replace current indexes.")

# ── Upload Section ──
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf",
    help="Upload a research paper PDF"
)

# Settings
with st.expander("⚙️ Advanced Settings"):
    col1, col2, col3 = st.columns(3)
    with col1:
        chunk_size = st.slider("Chunk Size", 200, 1000, 500, 50)
    with col2:
        chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10)
    with col3:
        top_k = st.slider("Top K Results", 1, 5, 3)

    st.session_state.chunk_size = chunk_size
    st.session_state.chunk_overlap = chunk_overlap
    st.session_state.top_k = top_k

# Process button
if uploaded_file is not None:
    st.success(f"✅ File uploaded: {uploaded_file.name} ({uploaded_file.size/1024:.1f} KB)")

    if st.button("🚀 Process Document & Build All Indexes", type="primary"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            os.makedirs(outputs_dir, exist_ok=True)
            progress = st.progress(0)
            status = st.empty()

            # Extract
            status.text("📄 Extracting text...")
            progress.progress(10)
            meta = get_paper_metadata(tmp_path)
            full_text = extract_full_text(tmp_path)
            sections = extract_sections(tmp_path)

            st.session_state.paper_title = meta['title']
            st.session_state.num_pages = meta['num_pages']
            st.session_state.paper_filename = uploaded_file.name

            # Save metadata
            with open(os.path.join(outputs_dir, "paper_meta.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "title": meta['title'],
                    "num_pages": meta['num_pages'],
                    "filename": uploaded_file.name
                }, f)

            st.write(f"✅ {meta['num_pages']} pages, {len(full_text)} chars, {len(sections)} sections")

            # Chunk
            status.text("✂️ Chunking...")
            progress.progress(20)
            raw_chunks = fixed_size_chunk(full_text, chunk_size, chunk_overlap)
            section_chunks = section_aware_chunk(sections, chunk_size, chunk_overlap)
            st.write(f"✅ {len(raw_chunks)} fixed chunks, {len(section_chunks)} section chunks")

            # Embedding model
            status.text("🤖 Loading embedding model...")
            progress.progress(30)
            if "embedding_model" not in st.session_state:
                st.session_state.embedding_model = load_embedding_model()
            model = st.session_state.embedding_model
            st.write("✅ Embedding model ready!")

            # Traditional
            status.text("📦 Building Traditional RAG...")
            progress.progress(40)
            t_chunks = traditional.build(raw_chunks)
            t_emb = embed_chunks(t_chunks, model)
            t_idx = build_faiss_index(t_emb)
            save_index(t_idx, t_chunks, outputs_dir, "traditional")
            st.session_state.traditional_index = t_idx
            st.session_state.traditional_chunks = t_chunks
            st.session_state.traditional_ready = True
            st.write("✅ Traditional RAG ready!")

            # Window
            status.text("🪟 Building Window RAG...")
            progress.progress(55)
            w_chunks = window.build(raw_chunks, window=1)
            w_emb = embed_chunks(w_chunks, model)
            w_idx = build_faiss_index(w_emb)
            save_index(w_idx, w_chunks, outputs_dir, "window")
            st.session_state.window_index = w_idx
            st.session_state.window_chunks = w_chunks
            st.session_state.window_ready = True
            st.write("✅ Window RAG ready!")

            # Semantic
            status.text("🧠 Building Semantic RAG (LLM calls)...")
            progress.progress(65)
            with st.spinner("Generating semantic context... (1-2 min)"):
                s_chunks = semantic.build(raw_chunks, generate_semantic_context)
            s_emb = embed_chunks(s_chunks, model)
            s_idx = build_faiss_index(s_emb)
            save_index(s_idx, s_chunks, outputs_dir, "semantic")
            st.session_state.semantic_index = s_idx
            st.session_state.semantic_chunks = s_chunks
            st.session_state.semantic_ready = True
            st.write("✅ Semantic RAG ready!")

            # Hierarchical
            status.text("🏛️ Building Hierarchical RAG (LLM calls)...")
            progress.progress(80)
            with st.spinner("Generating hierarchical context... (1-2 min)"):
                h_chunks = hierarchical.build(
                    section_chunks,
                    generate_hierarchical_context,
                    meta['title']
                )
            h_emb = embed_chunks(h_chunks, model)
            h_idx = build_faiss_index(h_emb)
            save_index(h_idx, h_chunks, outputs_dir, "hierarchical")
            st.session_state.hierarchical_index = h_idx
            st.session_state.hierarchical_chunks = h_chunks
            st.session_state.hierarchical_ready = True
            st.write("✅ Hierarchical RAG ready!")

            progress.progress(100)
            status.text("✅ Done!")
            st.session_state.pdf_processed = True

            st.balloons()
            st.success("""
            🎉 All 4 RAG indexes ready and saved!
            Next time you open the app — loads automatically!
            """)

            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {e}")
            raise e
        finally:
            os.unlink(tmp_path)

elif "pdf_processed" not in st.session_state:
    st.info("👆 Upload a PDF to get started")

    st.markdown("---")
    st.markdown("### Or Use Demo Paper")
    if st.button("📄 Use MGranRAG Paper (demo)"):
        demo_path = os.path.join(
            os.path.dirname(__file__), "../../data/paper.pdf"
        )
        if os.path.exists(demo_path):
            st.success("✅ Demo paper found! Processing...")
            st.session_state.demo_pdf_path = demo_path
        else:
            st.error("Demo paper not found at data/paper.pdf")