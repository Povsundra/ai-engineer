"""
utils/embedder.py
Converts text chunks into vectors and builds FAISS index.
Uses sentence-transformers (local, free, no API needed).
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ── Default embedding model ──
# Small, fast, good quality — perfect for demo
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Load sentence transformer model.
    Downloads automatically on first use (~80MB).

    Args:
        model_name: huggingface model name

    Returns:
        SentenceTransformer model
    """
    print(f"⏳ Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"✅ Embedding model loaded!")
    return model


def embed_chunks(
    chunks: List[Dict],
    model: SentenceTransformer = None,
    batch_size: int = 32
) -> np.ndarray:
    """
    Convert list of chunks into embedding vectors.

    Args:
        chunks: list of chunk dicts with 'text' key
        model: SentenceTransformer model (loads default if None)
        batch_size: how many chunks to embed at once

    Returns:
        numpy array of shape (num_chunks, embedding_dim)
    """
    if model is None:
        model = load_embedding_model()

    # Extract text from chunks
    texts = [chunk["text"] for chunk in chunks]

    print(f"⏳ Embedding {len(texts)} chunks...")

    # Embed with progress bar
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"✅ Embeddings created: shape {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS index from embeddings for fast similarity search.

    Args:
        embeddings: numpy array of shape (num_chunks, dim)

    Returns:
        faiss.Index ready for search
    """
    # Get embedding dimension
    dim = embeddings.shape[1]

    # Use simple flat index — exact search, good for small datasets
    index = faiss.IndexFlatIP(dim)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Add embeddings to index
    index.add(embeddings.astype(np.float32))

    print(f"✅ FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_index(
    index: faiss.Index,
    chunks: List[Dict],
    save_dir: str,
    name: str
):
    """
    Save FAISS index and chunks to disk.

    Args:
        index: FAISS index
        chunks: list of chunk dicts
        save_dir: folder to save in
        name: prefix name (e.g. 'traditional', 'window')
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save FAISS index
    index_path = os.path.join(save_dir, f"{name}_index.faiss")
    faiss.write_index(index, index_path)

    # Save chunks as JSON
    chunks_path = os.path.join(save_dir, f"{name}_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved index → {index_path}")
    print(f"✅ Saved chunks → {chunks_path}")


def load_index(
    save_dir: str,
    name: str
) -> tuple[faiss.Index, List[Dict]]:
    """
    Load FAISS index and chunks from disk.

    Args:
        save_dir: folder where files are saved
        name: prefix name used when saving

    Returns:
        tuple: (faiss.Index, list of chunks)
    """
    index_path = os.path.join(save_dir, f"{name}_index.faiss")
    chunks_path = os.path.join(save_dir, f"{name}_chunks.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks not found: {chunks_path}")

    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"✅ Loaded index: {index.ntotal} vectors")
    print(f"✅ Loaded chunks: {len(chunks)} chunks")

    return index, chunks


def build_and_save(
    chunks: List[Dict],
    save_dir: str,
    name: str,
    model: SentenceTransformer = None
) -> tuple[faiss.Index, np.ndarray]:
    """
    Full pipeline: chunks → embeddings → index → save.
    Convenience function used by all RAG types.

    Args:
        chunks: list of chunk dicts
        save_dir: where to save
        name: index name prefix
        model: embedding model

    Returns:
        tuple: (faiss.Index, embeddings)
    """
    print(f"\n🔨 Building index for: {name}")
    print("-" * 40)

    # Embed chunks
    embeddings = embed_chunks(chunks, model)

    # Build index
    index = build_faiss_index(embeddings)

    # Save to disk
    save_index(index, chunks, save_dir, name)

    return index, embeddings


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from utils.pdf_extractor import extract_full_text
    from utils.chunker import fixed_size_chunk

    print("=" * 50)
    print("Testing embedder.py")
    print("=" * 50)

    # Step 1: Load model
    print("\n🤖 Test 1: Load Embedding Model")
    model = load_embedding_model()

    # Step 2: Get chunks
    print("\n📦 Test 2: Get Chunks")
    full_text = extract_full_text("data/paper.pdf")
    chunks = fixed_size_chunk(full_text)

    # Step 3: Embed chunks
    print("\n🔢 Test 3: Embed Chunks")
    embeddings = embed_chunks(chunks, model)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Sample vector (first 5 dims): {embeddings[0][:5]}")

    # Step 4: Build index
    print("\n🗂️ Test 4: Build FAISS Index")
    index = build_faiss_index(embeddings)

    # Step 5: Save index
    print("\n💾 Test 5: Save Index")
    save_index(index, chunks, "outputs", "test")

    # Step 6: Load index back
    print("\n📂 Test 6: Load Index Back")
    loaded_index, loaded_chunks = load_index("outputs", "test")

    # Step 7: Quick search test
    print("\n🔍 Test 7: Quick Search Test")
    query = "What is the main contribution of this paper?"
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    distances, indices = loaded_index.search(
        query_embedding.astype(np.float32), k=3
    )

    print(f"Query: '{query}'")
    print(f"Top 3 results:")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        print(f"\n  Result {i+1} (score: {dist:.3f}):")
        print(f"  {loaded_chunks[idx]['text'][:150]}...")

    print("\n✅ embedder.py working correctly!")
    