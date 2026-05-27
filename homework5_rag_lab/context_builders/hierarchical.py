"""
context_builders/hierarchical.py
Hierarchical Contextual RAG — adds document structure path.
Each chunk knows: Paper → Section → Subsection → Content
Requires LLM calls during indexing (one per chunk).
"""

from typing import List, Dict
from tqdm import tqdm


def build(
    chunks: List[Dict],
    llm_fn,
    paper_title: str = "Research Paper"
) -> List[Dict]:
    """
    Hierarchical RAG — prepend document hierarchy path to each chunk.
    Uses section info from section_aware_chunk() in chunker.py.

    Args:
        chunks: section-aware chunks from chunker
                (must have 'section' key)
        llm_fn: function to generate hierarchical context
                (utils.llm.generate_hierarchical_context)
        paper_title: title of the paper

    Returns:
        chunks with hierarchical context prepended
    """
    print(f"🏛️ Hierarchical RAG: generating context for {len(chunks)} chunks...")
    print(f"   Paper: {paper_title[:60]}")
    print(f"   This makes {len(chunks)} LLM API calls — may take a moment...")

    hierarchical_chunks = []

    for i, chunk in enumerate(tqdm(chunks, desc="Generating hierarchical context")):
        chunk_text = chunk["text"]
        section = chunk.get("section", "Unknown Section")

        try:
            # Generate hierarchical context via LLM
            hier_context = llm_fn(
                chunk_text=chunk_text,
                section=section,
                paper_title=paper_title
            )

            # Build hierarchy path
            hierarchy_path = (
                f"Paper: {paper_title}\n"
                f"Section: {section}\n"
                f"Context: {hier_context}"
            )

            # Build enriched chunk
            new_chunk = chunk.copy()
            new_chunk["original_text"] = chunk_text
            new_chunk["hierarchy_path"] = hierarchy_path
            new_chunk["hier_context"] = hier_context
            new_chunk["context_type"] = "hierarchical"

            # Prepend hierarchy to chunk — this is what gets embedded
            new_chunk["text"] = (
                f"[LOCATION: {hierarchy_path}]\n\n"
                f"{chunk_text}"
            )

            # Context sent to LLM for answering
            new_chunk["context"] = new_chunk["text"]

        except Exception as e:
            print(f"\n⚠️ Error on chunk {i}: {e}")
            # Fallback — use section info only
            new_chunk = chunk.copy()
            new_chunk["original_text"] = chunk_text
            new_chunk["hierarchy_path"] = (
                f"Paper: {paper_title} | Section: {section}"
            )
            new_chunk["context_type"] = "hierarchical"
            new_chunk["context"] = (
                f"[Paper: {paper_title} | Section: {section}]\n\n"
                f"{chunk_text}"
            )
            new_chunk["text"] = new_chunk["context"]

        hierarchical_chunks.append(new_chunk)

    print(f"✅ Hierarchical RAG: {len(hierarchical_chunks)} chunks with hierarchy context")
    return hierarchical_chunks


def describe() -> str:
    """Return description of this RAG type."""
    return (
        "Hierarchical RAG prepends the document structure path to each chunk. "
        "Each chunk knows which paper, section, and subsection it belongs to, "
        "making retrieval much more accurate for structure-dependent questions."
    )