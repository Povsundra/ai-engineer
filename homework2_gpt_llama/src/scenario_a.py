from src.model_clients import call_gpt, call_llama
from src.metrics import syntax_ok_python
from src.logger import log_jsonl

PROMPT_A = "Write a Python function to calculate the Eigenvalues of a 4x4 matrix without using NumPy."

def run_scenario_a(model_name: str, temps: list[float], top_k: int = 50, top_p: float = 1.0, log_path: str = "logs/scenarioA.jsonl"):
    """
    Scenario A: temperature sweep T=0..2 step 0.2 with k=50, p=1.0
    Determine where python syntax fails.
    """
    results = []
    for t in temps:
        if model_name == "gpt":
            out = call_gpt(PROMPT_A, temperature=t, top_p=top_p)
        elif model_name == "llama":
            out = call_llama(PROMPT_A, temperature=t, top_p=top_p, top_k=top_k)
        else:
            raise ValueError("model_name must be 'gpt' or 'llama'")

        ok = syntax_ok_python(out)
        record = {
            "scenario": "A",
            "model": model_name,
            "prompt": PROMPT_A,
            "temperature": t,
            "top_p": top_p,
            "top_k": (None if model_name == "gpt" else top_k),
            "syntax_ok": ok,
            "output": out,
        }
        log_jsonl(log_path, record)
        results.append(record)
    return results

def first_failure_temperature(results: list[dict]) -> float | None:
    for r in results:
        if r["syntax_ok"] is False:
            return r["temperature"]
    return None
