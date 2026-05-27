import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def call_gpt(prompt: str, temperature: float, top_p: float) -> str:
    """
    OpenAI Chat Completions: supports temperature + top_p.
    (No top_k parameter in this API.)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=float(temperature),
        top_p=float(top_p),
    )
    return resp.choices[0].message.content or ""

def call_llama(prompt: str, temperature: float, top_p: float, top_k: int) -> str:
    """
    Ollama local generate endpoint supports temperature/top_p/top_k.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3")

    r = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
            },
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json().get("response", "")
