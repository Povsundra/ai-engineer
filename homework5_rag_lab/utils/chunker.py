"""
utils/chunker.py
Splits extracted PDF text into chunks for RAG.
"""

import re
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter


def fixed_size_chunk(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Dict]:
    """
    Split text into fixed-size chunks with overlap.
    This is the standard chunking used by ALL RAG types.

    Args:
        text: full text to chunk
        chunk_size: max tokens per chunk
        chunk_overlap: overlap between chunks

    Returns:
        list of dicts: [{"chunk_id": 0, "text": "..."}]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    raw_chunks = splitter.split_text(text)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "chunk_id": i,
            "text": chunk_text.strip(),
            "char_count": len(chunk_text.strip())
        })

    print(f"✅ Created {len(chunks)} chunks")
    print(f"   Avg chunk size: {sum(c['char_count'] for c in chunks) // len(chunks)} chars")
    print(f"   Min chunk size: {min(c['char_count'] for c in chunks)} chars")
    print(f"   Max chunk size: {max(c['char_count'] for c in chunks)} chars")

    return chunks


def section_aware_chunk(
    sections: Dict[str, str],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Dict]:
    """
    Chunk text while keeping track of which section each chunk came from.
    Used by Hierarchical RAG to know section path.

    Args:
        sections: dict from pdf_extractor.extract_sections()
        chunk_size: max chars per chunk
        chunk_overlap: overlap between chunks

    Returns:
        list of dicts: [{
            "chunk_id": 0,
            "text": "...",
            "section": "Methodology",
            "char_count": 450
        }]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    chunk_id = 0

    # Skip these sections — not useful for QA
    skip_sections = {"References", "Acknowledgments", "Header"}

    for section_name, section_text in sections.items():

        if section_name in skip_sections:
            continue

        if not section_text.strip():
            continue

        # Split this section into chunks
        raw_chunks = splitter.split_text(section_text)

        for chunk_text in raw_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "section": section_name,
                "char_count": len(chunk_text)
            })
            chunk_id += 1

    print(f"✅ Created {len(chunks)} section-aware chunks")
    print(f"   Sections included:")
    
    # Show chunk count per section
    section_counts = {}
    for chunk in chunks:
        section_counts[chunk["section"]] = section_counts.get(chunk["section"], 0) + 1
    
    for section, count in section_counts.items():
        print(f"   → {section}: {count} chunks")

    return chunks


def sentence_chunk(text: str) -> List[Dict]:
    """
    Split text into individual sentences.
    Used for semantic chunking analysis.

    Args:
        text: full text

    Returns:
        list of dicts: [{"chunk_id": 0, "text": "..."}]
    """
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if len(sentence) < 20:  # skip very short sentences
            continue

        chunks.append({
            "chunk_id": i,
            "text": sentence,
            "char_count": len(sentence)
        })

    print(f"✅ Created {len(chunks)} sentence chunks")
    return chunks


def display_chunks(chunks: List[Dict], num_show: int = 3):
    """
    Display sample chunks for inspection.
    Useful in notebooks to see what chunks look like.
    """
    print(f"\n📦 Showing {num_show} sample chunks out of {len(chunks)}:")
    print("=" * 60)

    for i, chunk in enumerate(chunks[:num_show]):
        print(f"\nChunk #{chunk['chunk_id']}")
        print(f"Section: {chunk.get('section', 'N/A')}")
        print(f"Length: {chunk['char_count']} chars")
        print(f"Text preview:")
        print(f"{chunk['text'][:200]}...")
        print("-" * 60)


if __name__ == "__main__":
    # Quick test
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from utils.pdf_extractor import extract_full_text, extract_sections

    pdf_path = "data/paper.pdf"

    print("=" * 50)
    print("Testing chunker.py")
    print("=" * 50)

    # Test 1: fixed size chunking
    print("\n📦 Test 1: Fixed Size Chunking")
    full_text = extract_full_text(pdf_path)
    chunks = fixed_size_chunk(full_text)
    display_chunks(chunks, num_show=2)

    # Test 2: section aware chunking
    print("\n📑 Test 2: Section-Aware Chunking")
    sections = extract_sections(pdf_path)
    section_chunks = section_aware_chunk(sections)
    display_chunks(section_chunks, num_show=2)

    # Test 3: sentence chunking
    print("\n📝 Test 3: Sentence Chunking")
    sentence_chunks = sentence_chunk(full_text[:3000])
    display_chunks(sentence_chunks, num_show=2)

    print("\n✅ chunker.py working correctly!")