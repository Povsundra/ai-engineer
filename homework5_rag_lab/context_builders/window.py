"""
context_builders/window.py
Window Contextual RAG — adds neighboring chunks as context.
Each chunk gets its surrounding chunks attached.
No LLM calls needed — fast and free!
"""

from typing import List, Dict


def build(
    chunks: List[Dict],
    window: int = 1
) -> List[Dict]:
    """
    Window RAG — attach neighboring chunks to each chunk.
    The embedding is done on the center chunk only,
    but the LLM receives the full window as context.

    Args:
        chunks: raw chunks from chunker
        window: how many neighbors on each side (default 1)

    Returns:
        chunks with window_text added
    """
    print(f"🪟 Window RAG: building window={window} for {len(chunks)} chunks...")

    windowed_chunks = []

    for i, chunk in enumerate(chunks):
        # Get neighbor indices
        start = max(0, i - window)
        end = min(len(chunks) - 1, i + window)

        # Collect window text
        window_parts = []
        for j in range(start, end + 1):
            if j == i:
                # Mark the main chunk
                window_parts.append(
                    f"[MAIN CHUNK]\n{chunks[j]['text']}\n[END MAIN CHUNK]"
                )
            else:
                window_parts.append(chunks[j]["text"])

        # Build windowed chunk
        new_chunk = chunk.copy()
        new_chunk["original_text"] = chunk["text"]
        new_chunk["window_text"] = "\n\n".join(window_parts)
        new_chunk["window_range"] = f"{start}-{end}"
        new_chunk["context_type"] = "window"

        # The text used for LLM context = full window
        new_chunk["context"] = new_chunk["window_text"]

        windowed_chunks.append(new_chunk)

    print(f"✅ Window RAG: {len(windowed_chunks)} chunks with window context")
    return windowed_chunks


def describe() -> str:
    """Return description of this RAG type."""
    return (
        "Window RAG attaches neighboring chunks to each retrieved chunk. "
        "When a chunk is retrieved, the chunks before and after it are also "
        "included, providing surrounding context without any LLM calls."
    )