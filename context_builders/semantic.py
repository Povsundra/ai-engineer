"""
context_builders/semantic.py
Semantic Contextual RAG — adds LLM-generated topic summary.
Each chunk gets a semantic description prepended before embedding.
Requires LLM calls during indexing (one per chunk).
"""

from typing import List, Dict
from tqdm import tqdm


def build(
    chunks: List[Dict],
    llm_fn,
    batch_size: int = 5
) -> List[Dict]:
    """
    Semantic RAG — prepend LLM-generated topic summary to each chunk.
    The enriched text is used for BOTH embedding and LLM context.

    Args:
        chunks: raw chunks from chunker
        llm_fn: function to generate semantic context
                (utils.llm.generate_semantic_context)
        batch_size: process in batches to avoid rate limits

    Returns:
        chunks with semantic context prepended
    """
    print(f"🧠 Semantic RAG: generating context for {len(chunks)} chunks...")
    print(f"   This makes {len(chunks)} LLM API calls — may take a moment...")

    semantic_chunks = []

    for i, chunk in enumerate(tqdm(chunks, desc="Generating semantic context")):
        chunk_text = chunk["text"]

        try:
            # Generate semantic summary via LLM
            semantic_summary = llm_fn(chunk_text)

            # Build enriched chunk
            new_chunk = chunk.copy()
            new_chunk["original_text"] = chunk_text
            new_chunk["semantic_summary"] = semantic_summary
            new_chunk["context_type"] = "semantic"

            # Prepend summary to chunk — this is what gets embedded
            new_chunk["text"] = (
                f"[TOPIC: {semantic_summary}]\n\n"
                f"{chunk_text}"
            )

            # Context sent to LLM for answering
            new_chunk["context"] = new_chunk["text"]

        except Exception as e:
            print(f"\n⚠️ Error on chunk {i}: {e}")
            # Fallback — use original chunk
            new_chunk = chunk.copy()
            new_chunk["original_text"] = chunk_text
            new_chunk["semantic_summary"] = "Unknown topic"
            new_chunk["context_type"] = "semantic"
            new_chunk["context"] = chunk_text

        semantic_chunks.append(new_chunk)

    print(f"✅ Semantic RAG: {len(semantic_chunks)} chunks with semantic context")
    return semantic_chunks


def describe() -> str:
    """Return description of this RAG type."""
    return (
        "Semantic RAG prepends an LLM-generated topic summary to each chunk "
        "before embedding. This makes chunks more self-contained and improves "
        "retrieval accuracy for topic-based queries."
    )