"""
streamlit_app/components/score_chart.py
Reusable chart components for RAG comparison.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


def show_score_bar(scores: dict, title: str = "Retrieval Scores"):
    """
    Show bar chart comparing scores across RAG types.

    Args:
        scores: dict of {rag_type: score}
        title: chart title
    """
    colors = {
        "traditional": "#757575",
        "window": "#1976d2",
        "semantic": "#388e3c",
        "hierarchical": "#f57c00"
    }

    rag_types = list(scores.keys())
    values = list(scores.values())
    bar_colors = [colors.get(r, "#ffffff") for r in rag_types]

    fig = go.Figure(go.Bar(
        x=rag_types,
        y=values,
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside"
    ))

    fig.update_layout(
        title=title,
        yaxis_title="Score",
        yaxis_range=[0, 1],
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)


def show_comparison_radar(
    all_scores: dict,
    questions: list
):
    """
    Show radar chart comparing all RAG types across questions.

    Args:
        all_scores: dict of {rag_type: [scores per question]}
        questions: list of question labels
    """
    colors = {
        "traditional": "#757575",
        "window": "#1976d2",
        "semantic": "#388e3c",
        "hierarchical": "#f57c00"
    }

    fig = go.Figure()

    short_questions = [f"Q{i+1}" for i in range(len(questions))]

    for rag_type, scores in all_scores.items():
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],  # close the polygon
            theta=short_questions + [short_questions[0]],
            name=rag_type.title(),
            line_color=colors.get(rag_type, "#ffffff"),
            fill="toself",
            opacity=0.3
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            ),
            bgcolor="#0e1117"
        ),
        paper_bgcolor="#0e1117",
        font_color="white",
        title="RAG Performance Across Questions",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def show_summary_table(results: dict):
    """
    Show summary table of all results.

    Args:
        results: dict of {rag_type: [result dicts]}
    """
    import pandas as pd

    rows = []
    for rag_type, rag_results in results.items():
        for i, result in enumerate(rag_results):
            avg_score = sum(
                c["score"] for c in result["retrieved_chunks"]
            ) / len(result["retrieved_chunks"])

            rows.append({
                "RAG Type": rag_type.title(),
                "Question": f"Q{i+1}",
                "Avg Score": f"{avg_score:.3f}",
                "Answer Preview": result["answer"][:100] + "..."
            })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)