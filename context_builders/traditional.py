"""
context_builders/traditional.py
Traditional RAG - no context added.
Chunks are used as-is from the chunker.
This is the BASELINE we compare against.
"""

from typing import List, Dict


def build(chunks: List[Dict]) -> List[Dict]:
    """
    Traditional RAG - return chunks unchanged.
    No context is added. This is the baseline.

    Args:
        chunks: raw chunks from chunker

    Returns:
        same chunks unchanged
    """
    print(f"[Traditional RAG] {len(chunks)} chunks (no context added)")
    return chunks


def describe() -> str:
    """Return description of this RAG type."""
    return (
        "Traditional RAG stores raw chunks with no additional context. "
        "Chunks may lack information about which document, section, or "
        "topic they belong to, leading to poor retrieval on complex questions."
    )