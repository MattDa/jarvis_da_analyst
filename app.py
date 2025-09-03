import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL.Image import register_extension
from openai import OpenAI

from agent import ReviewAgent

st.set_page_config(page_title="LLM Plotter", layout="wide")

st.title("LLM Data Visualizer")

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
prompt = st.text_input("Describe the visualization you want")

client = OpenAI(base_url="http://localhost:8000/v1", api_key=os.getenv("OPENAI_API_KEY", "llm"))
agent = ReviewAgent()


def generate_plot(df: pd.DataFrame, user_prompt: str):
    preview = df.head().to_csv(index=False)
    messages = [
        {
            "role": "system",
            "content": "You are a data visualization expert using Plotly in Python.",
        },
        {
            "role": "user",
            "content": (
                "Given the following data sample:\n" + preview +
                f"\nCreate Plotly code that satisfies the instruction: {user_prompt}. "
                "Assign the resulting Plotly figure to a variable named 'fig'."
                "Answer with just the code."
            ),
        },
    ]
    response = client.chat.completions.create(
        model="/models/Llama-3.1-8B-Instruct",
        messages=messages,
    )
    code = response.choices[0].message.content
    local_vars = {"df": df, "px": px, "go": go}
    code = code.replace("```", "") \
    .replace("python", "", 1)
    print(code)
    exec(code, local_vars)
    fig = local_vars.get("fig")
    return fig, code

if uploaded and prompt:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.dataframe(df.head())
        fig, code = generate_plot(df, prompt)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            review = agent.review_visual(prompt, fig.to_json())
            st.subheader("Agent Review")
            st.markdown(review.review)
            # st.subheader("Agent Chain of Thought")
            # st.markdown(review.thoughts)
    except Exception as e:
        st.error(f"Error: {e}")
        raise RuntimeError(e)

