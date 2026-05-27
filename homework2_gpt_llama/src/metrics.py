import ast
import re

def syntax_ok_python(code_text: str) -> bool:
    """
    Returns True if code parses as Python, False otherwise.
    Tries to extract a code block if present.
    """
    code = extract_code_block(code_text) or code_text
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

def extract_code_block(text: str) -> str | None:
    # Extract ```python ... ``` or ``` ... ```
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None

def repetition_score(text: str, n: int = 3) -> float:
    """
    Simple repetition score based on repeated n-grams.
    Higher = more repetitive.
    """
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < n:
        return 0.0
    ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return 1.0 - (unique / total) if total > 0 else 0.0

def vocab_richness(text: str) -> float:
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)
