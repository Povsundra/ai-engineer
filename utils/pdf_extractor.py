"""
utils/pdf_extractor.py
Extracts text and structure from PDF papers.
"""

import fitz  # pymupdf
import re
from pathlib import Path


# ── Section headers commonly found in academic papers ──
SECTION_PATTERNS = [
    r"^abstract$",
    r"^introduction$",
    r"^related work$",
    r"^background$",
    r"^preliminaries$",
    r"^methodology$",
    r"^method$",
    r"^approach$",
    r"^experiments?$",
    r"^experimental settings?$",
    r"^results?$",
    r"^evaluation$",
    r"^ablation study$",
    r"^discussion$",
    r"^conclusion$",
    r"^future work$",
    r"^references$",
    r"^acknowledgments?$",
]


def is_section_header(text: str) -> bool:
    """Check if a line looks like a section header."""
    text = text.strip().lower()

    # Skip empty or very long lines
    if not text or len(text) > 80:
        return False

    # Check against known section patterns
    for pattern in SECTION_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return True

    # Check numbered sections like "1. Introduction" or "2 Methodology"
    if re.match(r"^\d+\.?\s+[A-Z][a-zA-Z\s]+$", text.strip()):
        return True

    return False


def extract_text_by_page(pdf_path: str) -> list[dict]:
    """
    Extract text from PDF page by page.
    
    Returns:
        list of dicts: [{"page": 1, "text": "..."}]
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # Clean up text
        text = text.strip()
        text = re.sub(r'\n{3,}', '\n\n', text)  # max 2 newlines

        pages.append({
            "page": page_num + 1,
            "text": text
        })

    doc.close()
    print(f"✅ Extracted {len(pages)} pages from {pdf_path.name}")
    return pages


def extract_full_text(pdf_path: str) -> str:
    """
    Extract all text from PDF as single string.
    
    Returns:
        str: full text of the PDF
    """
    pages = extract_text_by_page(pdf_path)
    full_text = "\n\n".join([p["text"] for p in pages])
    print(f"✅ Total characters extracted: {len(full_text)}")
    return full_text


def extract_sections(pdf_path: str) -> dict:
    """
    Extract text grouped by section.
    Detects section headers and groups content under them.

    Returns:
        dict: {
            "Abstract": "text...",
            "Introduction": "text...",
            "Methodology": "text...",
            ...
        }
    """
    pages = extract_text_by_page(pdf_path)
    full_text = "\n".join([p["text"] for p in pages])

    sections = {}
    current_section = "Header"  # text before first section
    current_content = []

    lines = full_text.split("\n")

    for line in lines:
        if is_section_header(line):
            # Save previous section
            if current_content:
                sections[current_section] = "\n".join(current_content).strip()

            # Start new section
            current_section = line.strip().title()
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_content:
        sections[current_section] = "\n".join(current_content).strip()

    # Remove empty sections
    sections = {k: v for k, v in sections.items() if v.strip()}

    print(f"✅ Detected {len(sections)} sections:")
    for section_name in sections.keys():
        print(f"   → {section_name}")

    return sections


def get_paper_metadata(pdf_path: str) -> dict:
    """
    Extract basic metadata from PDF.

    Returns:
        dict: title, authors, num_pages
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    # Get first page text for title/authors
    first_page = doc[0].get_text("text")
    lines = [l.strip() for l in first_page.split("\n") if l.strip()]

    metadata = {
        "filename": pdf_path.name,
        "num_pages": len(doc),
        "title": lines[0] if lines else "Unknown",
        "first_page_preview": first_page[:500]
    }

    doc.close()

    print(f"✅ Paper: {metadata['title'][:60]}...")
    print(f"✅ Pages: {metadata['num_pages']}")

    return metadata


if __name__ == "__main__":
    # Quick test when run directly
    import sys

    pdf_path = "data/paper.pdf"

    print("=" * 50)
    print("Testing pdf_extractor.py")
    print("=" * 50)

    # Test 1: metadata
    print("\n📋 Test 1: Metadata")
    meta = get_paper_metadata(pdf_path)

    # Test 2: pages
    print("\n📄 Test 2: Pages")
    pages = extract_text_by_page(pdf_path)
    print(f"First 200 chars of page 1:\n{pages[0]['text'][:200]}")

    # Test 3: sections
    print("\n📑 Test 3: Sections")
    sections = extract_sections(pdf_path)
    for name, content in sections.items():
        print(f"\n[{name}] — {len(content)} chars")
        print(f"Preview: {content[:100]}...")

    print("\n✅ pdf_extractor.py working correctly!")