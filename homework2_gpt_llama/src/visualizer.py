import math
import os
import matplotlib.pyplot as plt
from src.logger import ensure_dir

def softmax(logits, temperature: float):
    scaled = [x / temperature for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    return [e / s for e in exps]

def plot_temperature_flattening(save_path: str = "figures/temperature_flattening.png"):
    """
    Requirement: show probability distribution flattens as T increases.
    Uses sample logits (valid for demonstration).
    """
    logits = [5.1, 4.8, 2.1, 1.2, 0.3]
    labels = [f"tok{i}" for i in range(1, len(logits) + 1)]
    temps = [0.1, 0.7, 1.5]

    ensure_dir(os.path.dirname(save_path))

    for t in temps:
        probs = softmax(logits, t)
        plt.figure()
        plt.bar(labels, probs)
        plt.title(f"Softmax probabilities at Temperature T={t}")
        plt.ylabel("Probability")
        plt.xlabel("Token")
        plt.ylim(0, 1)
        plt.tight_layout()
        # Save separate files for each T
        out = save_path.replace(".png", f"_T{str(t).replace('.','_')}.png")
        plt.savefig(out)
        plt.close()

    return [save_path.replace(".png", f"_T{str(t).replace('.','_')}.png") for t in temps]
