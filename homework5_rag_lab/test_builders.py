
"""
test_builders.py
Test all context builders.
Run: python test_builders.py
"""

import sys
import os

# Add both paths
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./context_builders'))

from utils.pdf_extractor import extract_full_text, extract_sections, get_paper_metadata
from utils.chunker import fixed_size_chunk, section_aware_chunk
from utils.llm import generate_semantic_context, generate_hierarchical_context

# Import using importlib - most reliable way
import importlib.util

def load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load each builder directly from file path
traditional = load_module("traditional", "context_builders/traditional.py")
window = load_module("window", "context_builders/window.py")
semantic = load_module("semantic", "context_builders/semantic.py")
hierarchical = load_module("hierarchical", "context_builders/hierarchical.py")

print('=' * 50)
print('Testing all context builders')
print('=' * 50)

# Load data
full_text = extract_full_text('data/paper.pdf')
sections = extract_sections('data/paper.pdf')
meta = get_paper_metadata('data/paper.pdf')
paper_title = meta['title']

# Get chunks
raw_chunks = fixed_size_chunk(full_text, chunk_size=500)
section_chunks = section_aware_chunk(sections)

# Use only first 3 chunks for speed
test_chunks = raw_chunks[:3]
test_section_chunks = section_chunks[:3]

print()

# Test 1: Traditional
print('1 - Traditional RAG:')
t_chunks = traditional.build(test_chunks)
print(f'   Sample: {t_chunks[0]["text"][:100]}...')
print()

# Test 2: Window
print('2 - Window RAG:')
w_chunks = window.build(test_chunks, window=1)
print(f'   Sample: {w_chunks[1]["context"][:150]}...')
print()

# Test 3: Semantic
print('3 - Semantic RAG:')
s_chunks = semantic.build(test_chunks, generate_semantic_context)
print(f'   Summary: {s_chunks[0]["semantic_summary"]}')
print(f'   Sample: {s_chunks[0]["text"][:150]}...')
print()

# Test 4: Hierarchical
print('4 - Hierarchical RAG:')
h_chunks = hierarchical.build(
    test_section_chunks,
    generate_hierarchical_context,
    paper_title
)
print(f'   Path: {h_chunks[0]["hierarchy_path"]}')
print(f'   Sample: {h_chunks[0]["text"][:150]}...')
print()

print('All context builders working!')