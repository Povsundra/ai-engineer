"""
utils/llm.py
Handles all LLM calls via OpenRouter API.
Used for:
1. Generating answers from retrieved context
2. Generating context summaries for Semantic RAG
3. Generating hierarchical context for Hierarchical RAG
"""

import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_client() -> OpenAI:
    """
    Create OpenRouter client.
    Reads API key from .env file.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1"
    )

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found in .env file!\n"
            "Add this to your .env:\n"
            "OPENROUTER_API_KEY=your_key_here"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    return client


def generate_answer(
    question: str,
    context: str,
    model: str = None,
    max_tokens: int = 500
) -> str:
    """
    Generate answer from question + retrieved context.
    Core function used by ALL 4 RAG types.

    Args:
        question: user question
        context: formatted retrieved chunks
        model: LLM model name
        max_tokens: max response length

    Returns:
        str: generated answer
    """
    if model is None:
        model = os.getenv("LLM_MODEL", "anthropic/claude-haiku-3-5")

    client = get_client()

    prompt = f"""You are a helpful research assistant.
Answer the question based ONLY on the provided context.
If the context does not contain enough information, say so clearly.
Be concise and specific.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.1  # low temperature for factual answers
    )

    return response.choices[0].message.content.strip()


def generate_semantic_context(
    chunk_text: str,
    model: str = None,
    max_tokens: int = 100
) -> str:
    """
    Generate a short semantic summary for a chunk.
    Used by Semantic Contextual RAG.

    Args:
        chunk_text: raw chunk text
        model: LLM model name
        max_tokens: keep short — just a summary

    Returns:
        str: 1-2 sentence semantic context
    """
    if model is None:
        model = os.getenv("LLM_MODEL", "anthropic/claude-haiku-3-5")

    client = get_client()

    prompt = f"""In 1-2 sentences, describe the main topic and key concept of this text chunk.
Be specific and concise. Only output the description, nothing else.

Text chunk:
{chunk_text}

Description:"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.1
    )

    return response.choices[0].message.content.strip()


def generate_hierarchical_context(
    chunk_text: str,
    section: str,
    paper_title: str = "Research Paper",
    model: str = None,
    max_tokens: int = 150
) -> str:
    """
    Generate hierarchical context for a chunk.
    Used by Hierarchical Contextual RAG.

    Args:
        chunk_text: raw chunk text
        section: section name from pdf_extractor
        paper_title: title of the paper
        model: LLM model name
        max_tokens: keep short

    Returns:
        str: context with hierarchy path prepended
    """
    if model is None:
        model = os.getenv("LLM_MODEL", "anthropic/claude-haiku-3-5")

    client = get_client()

    prompt = f"""Given this chunk from a research paper, write a SHORT context 
that situates it within the document structure.
Include: paper title, section, and what this chunk is about.
Format: "From [paper]: [section] section. This chunk discusses [topic]."
Only output the context line, nothing else.

Paper title: {paper_title}
Section: {section}
Chunk: {chunk_text[:300]}

Context:"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.1
    )

    return response.choices[0].message.content.strip()


def generate_window_context(
    chunk_text: str,
    prev_chunk: str = "",
    next_chunk: str = "",
    model: str = None,
    max_tokens: int = 100
) -> str:
    """
    Generate context summary for window RAG.
    Summarizes what the surrounding chunks are about.

    Args:
        chunk_text: main chunk text
        prev_chunk: previous chunk text
        next_chunk: next chunk text
        model: LLM model name
        max_tokens: keep short

    Returns:
        str: brief context about the window
    """
    if model is None:
        model = os.getenv("LLM_MODEL", "anthropic/claude-haiku-3-5")

    client = get_client()

    context_info = []
    if prev_chunk:
        context_info.append(f"Previous context: {prev_chunk[:150]}")
    if next_chunk:
        context_info.append(f"Following context: {next_chunk[:150]}")

    surrounding = "\n".join(context_info)

    prompt = f"""In 1 sentence, describe what topic connects these text segments.
Only output the sentence, nothing else.

{surrounding}

Main text: {chunk_text[:200]}

Connecting topic:"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.1
    )

    return response.choices[0].message.content.strip()


def test_connection() -> bool:
    """
    Test if OpenRouter API connection works.
    Call this in setup to verify API key is valid.

    Returns:
        bool: True if connected, False if failed
    """
    try:
        client = get_client()
        model = os.getenv("LLM_MODEL", "anthropic/claude-haiku-3-5")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say 'API connected!' and nothing else."}
            ],
            max_tokens=20
        )

        result = response.choices[0].message.content.strip()
        print(f"✅ API Response: {result}")
        return True

    except Exception as e:
        print(f"❌ API Connection failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from utils.pdf_extractor import extract_full_text, get_paper_metadata
    from utils.chunker import fixed_size_chunk, section_aware_chunk
    from utils.embedder import load_embedding_model, embed_chunks, build_faiss_index
    from utils.retriever import retrieve_top_k, format_context

    print("=" * 50)
    print("Testing llm.py")
    print("=" * 50)

    # Test 1: API Connection
    print("\n🔌 Test 1: API Connection")
    connected = test_connection()
    if not connected:
        print("❌ Fix API key in .env file first!")
        sys.exit(1)

    # Test 2: Get paper info
    print("\n📄 Test 2: Setup RAG Pipeline")
    model = load_embedding_model()
    meta = get_paper_metadata("data/paper.pdf")
    paper_title = meta["title"]

    full_text = extract_full_text("data/paper.pdf")
    chunks = fixed_size_chunk(full_text)
    embeddings = embed_chunks(chunks, model)
    index = build_faiss_index(embeddings)

    # Test question
    question = "What is the main contribution of MGranRAG?"

    # Test 3: Generate Answer (Traditional RAG style)
    print("\n💬 Test 3: Generate Answer")
    results = retrieve_top_k(question, index, chunks, model, k=3)
    context = format_context(results)
    answer = generate_answer(question, context)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

    # Test 4: Generate Semantic Context
    print("\n🧠 Test 4: Generate Semantic Context")
    sample_chunk = chunks[5]["text"]
    semantic_ctx = generate_semantic_context(sample_chunk)
    print(f"Original chunk:\n{sample_chunk[:200]}...")
    print(f"\nSemantic context:\n{semantic_ctx}")

    # Test 5: Generate Hierarchical Context
    print("\n🏛️ Test 5: Generate Hierarchical Context")
    hier_ctx = generate_hierarchical_context(
        chunk_text=sample_chunk,
        section="Methodology",
        paper_title=paper_title
    )
    print(f"Hierarchical context:\n{hier_ctx}")

    # Test 6: Generate Window Context
    print("\n🪟 Test 6: Generate Window Context")
    window_ctx = generate_window_context(
        chunk_text=chunks[10]["text"],
        prev_chunk=chunks[9]["text"],
        next_chunk=chunks[11]["text"]
    )
    print(f"Window context:\n{window_ctx}")

    print("\n✅ llm.py working correctly!")