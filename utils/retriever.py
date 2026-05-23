"""
utils/retriever.py
Retrieves most relevant chunks for a given query.
Used by all 4 RAG types.
"""

import numpy as np
import faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer


def embed_query(
    query: str,
    model: SentenceTransformer
) -> np.ndarray:
    """
    Convert query text into embedding vector.

    Args:
        query: user question string
        model: SentenceTransformer model

    Returns:
        numpy array of shape (1, embedding_dim)
    """
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)

    return query_embedding.astype(np.float32)


def retrieve_top_k(
    query: str,
    index: faiss.Index,
    chunks: List[Dict],
    model: SentenceTransformer,
    k: int = 3
) -> List[Dict]:
    """
    Retrieve top-k most relevant chunks for a query.
    Used by Traditional RAG.

    Args:
        query: user question
        index: FAISS index
        chunks: list of chunk dicts
        model: embedding model
        k: number of chunks to retrieve

    Returns:
        list of top-k chunks with scores added
    """
    # Embed query
    query_embedding = embed_query(query, model)

    # Search index
    scores, indices = index.search(query_embedding, k)

    # Build results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # FAISS returns -1 for empty slots
            continue

        chunk = chunks[idx].copy()
        chunk["retrieval_score"] = float(score)
        chunk["rank"] = len(results) + 1
        results.append(chunk)

    return results


def retrieve_with_window(
    query: str,
    index: faiss.Index,
    chunks: List[Dict],
    model: SentenceTransformer,
    k: int = 3,
    window: int = 1
) -> List[Dict]:
    """
    Retrieve top-k chunks AND their neighbors (window context).
    Used by Window RAG.

    Args:
        query: user question
        index: FAISS index
        chunks: list of chunk dicts
        model: embedding model
        k: number of chunks to retrieve
        window: how many neighbors on each side

    Returns:
        list of chunks with window context added
    """
    # First get top-k normally
    top_chunks = retrieve_top_k(query, index, chunks, model, k)

    # For each retrieved chunk, add neighbors
    results = []
    seen_ids = set()

    for chunk in top_chunks:
        chunk_id = chunk["chunk_id"]

        # Get neighbor indices
        start = max(0, chunk_id - window)
        end = min(len(chunks) - 1, chunk_id + window)

        # Build window text
        window_parts = []
        for i in range(start, end + 1):
            if i < len(chunks):
                window_parts.append(chunks[i]["text"])

        # Create windowed chunk
        windowed_chunk = chunk.copy()
        windowed_chunk["original_text"] = chunk["text"]
        windowed_chunk["window_text"] = "\n\n".join(window_parts)
        windowed_chunk["window_range"] = f"chunks {start}-{end}"

        # Use window text as the context sent to LLM
        windowed_chunk["text"] = windowed_chunk["window_text"]

        if chunk_id not in seen_ids:
            results.append(windowed_chunk)
            seen_ids.add(chunk_id)

    return results


def format_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into context string for LLM prompt.

    Args:
        chunks: list of retrieved chunk dicts

    Returns:
        formatted context string
    """
    context_parts = []

    for i, chunk in enumerate(chunks):
        section = chunk.get("section", "Unknown")
        score = chunk.get("retrieval_score", 0)
        text = chunk.get("text", "")

        context_parts.append(
            f"[Chunk {i+1}] "
            f"(Section: {section}, Score: {score:.3f})\n"
            f"{text}"
        )

    return "\n\n---\n\n".join(context_parts)


def display_results(
    query: str,
    results: List[Dict],
    show_full_text: bool = False
):
    """
    Pretty print retrieval results.
    Useful in notebooks and testing.

    Args:
        query: the question asked
        results: list of retrieved chunks
        show_full_text: show full text or preview only
    """
    print(f"\n🔍 Query: '{query}'")
    print(f"📦 Retrieved {len(results)} chunks:")
    print("=" * 60)

    for chunk in results:
        print(f"\n📄 Rank #{chunk['rank']}")
        print(f"   Chunk ID: {chunk['chunk_id']}")
        print(f"   Section:  {chunk.get('section', 'N/A')}")
        print(f"   Score:    {chunk['retrieval_score']:.3f}")

        text = chunk.get("text", "")
        if show_full_text:
            print(f"   Text:\n{text}")
        else:
            print(f"   Preview: {text[:200]}...")

        # Show window info if available
        if "window_range" in chunk:
            print(f"   Window:  {chunk['window_range']}")

        print("-" * 60)


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from utils.pdf_extractor import extract_full_text
    from utils.chunker import fixed_size_chunk
    from utils.embedder import (
        load_embedding_model,
        embed_chunks,
        build_faiss_index
    )

    print("=" * 50)
    print("Testing retriever.py")
    print("=" * 50)

    # Setup
    print("\n⚙️ Setup: Loading model and building index...")
    model = load_embedding_model()
    full_text = extract_full_text("data/paper.pdf")
    chunks = fixed_size_chunk(full_text)
    embeddings = embed_chunks(chunks, model)
    index = build_faiss_index(embeddings)

    # Test questions
    test_questions = [
        "What problem does MGranRAG solve?",
        "How is the Contextual Hierarchical Graph constructed?",
        "What datasets were used in the experiments?",
    ]

    # Test 1: Traditional retrieval
    print("\n" + "=" * 50)
    print("📌 Test 1: Traditional Retrieval (top-k)")
    print("=" * 50)

    for question in test_questions:
        results = retrieve_top_k(
            query=question,
            index=index,
            chunks=chunks,
            model=model,
            k=3
        )
        display_results(question, results)

    # Test 2: Window retrieval
    print("\n" + "=" * 50)
    print("🪟 Test 2: Window Retrieval (with neighbors)")
    print("=" * 50)

    results = retrieve_with_window(
        query=test_questions[0],
        index=index,
        chunks=chunks,
        model=model,
        k=3,
        window=1
    )
    display_results(test_questions[0], results)

    # Test 3: Format context
    print("\n" + "=" * 50)
    print("📝 Test 3: Format Context for LLM")
    print("=" * 50)

    results = retrieve_top_k(
        query=test_questions[0],
        index=index,
        chunks=chunks,
        model=model,
        k=3
    )
    context = format_context(results)
    print(f"Formatted context preview:")
    print(context[:500])
    print("...")

    print("\n✅ retriever.py working correctly!")