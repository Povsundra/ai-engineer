from src.model_clients import call_gpt, call_llama
from src.metrics import repetition_score, vocab_richness
from src.logger import log_jsonl

PROMPT_B = "Summarize the history of Angkor Wat in the style of a 1920s hardboiled detective."

def run_scenario_b(model_name: str, temperature: float = 0.8, log_path: str = "logs/scenarioB.jsonl"):
    """
    Scenario B:
      - Llama: compare Top-K (k=5) vs Top-P (p=0.9)
      - GPT: compare restrictive Top-P (p=0.2) vs normal Top-P (p=0.9) since top_k isn't supported
    """
    outputs = []

    if model_name == "llama":
        # Run 1: Top-K fixed
        out1 = call_llama(PROMPT_B, temperature=temperature, top_p=1.0, top_k=5)
        # Run 2: Top-P nucleus
        out2 = call_llama(PROMPT_B, temperature=temperature, top_p=0.9, top_k=50)

        runs = [
            ("topk_5", {"top_k": 5, "top_p": 1.0}, out1),
            ("topp_0.9", {"top_k": 50, "top_p": 0.9}, out2),
        ]

    elif model_name == "gpt":
        # GPT doesn't support top_k, so we compare two top_p regimes
        out1 = call_gpt(PROMPT_B, temperature=temperature, top_p=0.2)  # restrictive
        out2 = call_gpt(PROMPT_B, temperature=temperature, top_p=0.9)  # more diverse

        runs = [
            ("topp_0.2", {"top_k": None, "top_p": 0.2}, out1),
            ("topp_0.9", {"top_k": None, "top_p": 0.9}, out2),
        ]
    else:
        raise ValueError("model_name must be 'gpt' or 'llama'")

    for tag, params, out in runs:
        rep = repetition_score(out, n=3)
        rich = vocab_richness(out)
        record = {
            "scenario": "B",
            "model": model_name,
            "prompt": PROMPT_B,
            "temperature": temperature,
            "top_p": params["top_p"],
            "top_k": params["top_k"],
            "variant": tag,
            "repetition_score_trigram": rep,
            "vocab_richness": rich,
            "output": out,
        }
        log_jsonl(log_path, record)
        outputs.append(record)

    return outputs
