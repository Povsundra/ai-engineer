"""
streamlit_app/components/answer_card.py
Reusable component to display RAG answers.
"""

import streamlit as st


def show_answer(
    question: str,
    answer: str,
    rag_type: str,
    color: str = "#1976d2"
):
    """Display a single answer card."""

    icons = {
        "traditional": "📦",
        "window": "🪟",
        "semantic": "🧠",
        "hierarchical": "🏛️"
    }

    icon = icons.get(rag_type, "💬")

    st.markdown(f"""
        <div style='
            border: 1px solid {color};
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            background-color: #1a1a2e;
        '>
            <div style='color: {color}; font-weight: bold; margin-bottom: 8px;'>
                {icon} {rag_type.title()} RAG
            </div>
            <div style='color: #cccccc; font-size: 0.9em; margin-bottom: 8px;'>
                Q: {question}
            </div>
            <div style='color: white;'>
                {answer}
            </div>
        </div>
    """, unsafe_allow_html=True)


def show_answer_comparison(
    question: str,
    answers: dict
):
    """
    Show answers from all RAG types side by side.

    Args:
        question: the question asked
        answers: dict of {rag_type: answer_text}
    """
    colors = {
        "traditional": "#757575",
        "window": "#1976d2",
        "semantic": "#388e3c",
        "hierarchical": "#f57c00"
    }

    st.markdown(f"**Q: {question}**")
    st.markdown("---")

    cols = st.columns(len(answers))

    for col, (rag_type, answer) in zip(cols, answers.items()):
        with col:
            color = colors.get(rag_type, "#ffffff")
            icons = {
                "traditional": "📦",
                "window": "🪟",
                "semantic": "🧠",
                "hierarchical": "🏛️"
            }
            icon = icons.get(rag_type, "💬")

            st.markdown(f"""
                <div style='
                    border-top: 3px solid {color};
                    padding: 10px;
                    background-color: #1a1a2e;
                    border-radius: 4px;
                    height: 100%;
                '>
                    <div style='color: {color}; font-weight: bold;'>
                        {icon} {rag_type.title()}
                    </div>
                    <div style='color: white; margin-top: 8px; font-size: 0.9em;'>
                        {answer if answer else "Not answered yet"}
                    </div>
                </div>
            """, unsafe_allow_html=True)