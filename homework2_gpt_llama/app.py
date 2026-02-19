import streamlit as st
import pandas as pd

from src.scenario_a import run_scenario_a, first_failure_temperature
from src.scenario_b import run_scenario_b
from src.visualizer import plot_temperature_flattening

st.set_page_config(page_title="HW2 - Decoding Experiments", layout="wide")

st.title("HW2: Temperature, Top-P, Top-K (Llama local vs GPT-4o API)")

st.info(
    "Note: GPT-4o API supports temperature and top_p. "
    "Top_k is applied on Llama (Ollama). For Scenario B on GPT, we compare top_p=0.2 vs top_p=0.9."
)

tab1, tab2, tab3 = st.tabs(["Scenario A (Temperature Sweep)", "Scenario B (Top-K vs Top-P)", "Visualizer Plot"])

with tab1:
    st.header("Scenario A: Temperature Sweep (T=0.0 → 2.0 step 0.2)")
    model = st.selectbox("Choose model", ["gpt", "llama"])
    if st.button("Run Scenario A"):
        temps = [round(x * 0.2, 1) for x in range(0, 11)]  # 0.0..2.0
        log_file = f"logs/scenarioA_{model}.jsonl"
        results = run_scenario_a(model_name=model, temps=temps, top_k=50, top_p=1.0, log_path=log_file)

        fail_t = first_failure_temperature(results)
        st.success(f"Done. Logs saved to {log_file}")

        df = pd.DataFrame([{
            "T": r["temperature"],
            "syntax_ok": r["syntax_ok"]
        } for r in results])
        st.dataframe(df, use_container_width=True)

        if fail_t is None:
            st.write("✅ No syntax failure found in this sweep.")
        else:
            st.warning(f"⚠️ First syntax failure at T = {fail_t}")

with tab2:
    st.header("Scenario B: Style prompt (Top-K vs Top-P / Top-P regimes)")
    model = st.selectbox("Choose model for Scenario B", ["llama", "gpt"], key="model_b")
    if st.button("Run Scenario B"):
        log_file = f"logs/scenarioB_{model}.jsonl"
        outputs = run_scenario_b(model_name=model, temperature=0.8, log_path=log_file)
        st.success(f"Done. Logs saved to {log_file}")

        df = pd.DataFrame([{
            "variant": o["variant"],
            "top_p": o["top_p"],
            "top_k": o["top_k"],
            "repetition_score_trigram": round(o["repetition_score_trigram"], 3),
            "vocab_richness": round(o["vocab_richness"], 3),
        } for o in outputs])
        st.dataframe(df, use_container_width=True)

        for o in outputs:
            st.subheader(f"Output: {o['variant']}")
            st.code(o["output"])

with tab3:
    st.header("Probability Flattening Visualizer")
    if st.button("Generate plots"):
        paths = plot_temperature_flattening()
        st.success("Plots generated in figures/")
        for p in paths:
            st.image(p, caption=p)
