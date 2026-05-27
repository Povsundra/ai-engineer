# Homework 2 — Decoding Experiments (Llama 3 Local + GPT-4o API)

## Project Overview
This project is a small **Streamlit** application for Homework 2 to explore how decoding parameters affect text generation.  
We run controlled experiments on **two different model families**:

- **Llama 3 (Local)** running on your machine via **Ollama**
- **GPT-4o (API)** accessed through the **OpenAI API**

The app supports:
- **Scenario A:** Temperature sweep (grid search) and syntax-checking generated Python code
- **Scenario B:** Sampling comparison (Top-K vs Top-P for Llama; Top-P regimes for GPT)
- **Visualizer:** Temperature effect plot showing probability distribution flattening

Outputs are automatically saved to:
- `logs/` (raw JSONL logs)
- `figures/` (generated plots)

---

## Models Used
### 1) Llama 3 (Local)
- Served by **Ollama**
- Endpoint: `http://localhost:11434`
- Supports `temperature`, `top_p`, `top_k`

### 2) GPT-4o (API)
- Served by **OpenAI API**
- Supports `temperature`, `top_p`
- Note: `top_k` is not available in OpenAI Chat Completions API (we document this in code/comments)

---

## Project Structure
```txt
homework2/
  app.py
  requirements.txt
  .env                # local only (ignored by git)
  .gitignore
  src/
    __init__.py
    model_clients.py
    scenario_a.py
    scenario_b.py
    metrics.py
    visualizer.py
    logger.py
  logs/               # auto-generated run logs (JSONL)
  figures/            # auto-generated plots (PNG)

``` 


# Requirements

- Python 3.10+ (3.12 also works)
- Internet connection (for GPT-4o API)
- Ollama installed (for Llama 3 local)

## Setup 
    1) Create & activate virtual environment
        python -m venv .venv
        .\.venv\Scripts\activate
    2) Install dependencies
        - pip install --upgrade pip
        - pip install -r requirements.txt
    3) Configure .env (project root)
        Create/edit .env:
    4) Start Llama 3 locally (Ollama)
    5) Run the Streamlit app 
        streamlit run app.py
    6) Outputs
        - Logs: logs/ (JSONL)
        - Plots: figures/ (PNG)

