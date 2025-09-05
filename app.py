import os
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent import ReviewAgent

st.set_page_config(page_title="LLM Plotter", layout="wide")

st.title("LLM Data Visualizer")


def ensure_consistent_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce each DataFrame column to a single dtype."""
    for col in df.columns:
        series = df[col]
        try:
            df[col] = pd.to_numeric(series, errors="raise")
            continue
        except (ValueError, TypeError):
            pass
        try:
            df[col] = pd.to_datetime(series, errors="raise")
            continue
        except (ValueError, TypeError):
            pass
        df[col] = series.astype(str)
    return df


def extract_code(text: str) -> str:
    pattern = r"```(?:python)?\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "system_message" not in st.session_state:
    st.session_state.system_message = None
if "last_fig" not in st.session_state:
    st.session_state.last_fig = None
if "last_review" not in st.session_state:
    st.session_state.last_review = None
if "last_thoughts" not in st.session_state:
    st.session_state.last_thoughts = None


uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded and st.session_state.df is None:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        df = ensure_consistent_dtypes(df)
        st.session_state.df = df
        preview = df.head().to_csv(index=False)
        st.session_state.system_message = SystemMessage(
            content=(
                "You are a data visualization expert using Plotly in Python. "
                f"Use the following data sample:\n{preview}\n" \
                "Return only Python code that creates a Plotly figure assigned to variable 'fig'."
            )
        )
        st.session_state.messages = []
    except Exception as e:
        st.error(f"Error: {e}")
        print(e)


df = st.session_state.df
if df is not None:
    st.dataframe(df.head())


chat = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key=os.getenv("OPENAI_API_KEY", "llm"),
    model="meta-llama/Llama-3.1-8B-Instruct",
)

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        if role == "assistant":
            st.code(msg.content, language="python")
        else:
            st.markdown(msg.content)


user_input = st.chat_input("Describe a visualization") if df is not None else None

if user_input:
    human_msg = HumanMessage(user_input)
    st.session_state.messages.append(human_msg)
    with st.chat_message("user"):
        st.markdown(user_input)

    conversation = [st.session_state.system_message] + st.session_state.messages
    ai_msg = chat.invoke(conversation)
    code = extract_code(ai_msg.content)
    st.session_state.messages.append(AIMessage(code))

    local_vars = {"df": df, "px": px, "go": go}
    try:
        exec(code, local_vars)
        fig = local_vars.get("fig")
        st.session_state.last_fig = fig
        if fig is not None:
            review = ReviewAgent().review_visual(user_input, fig.to_dict())
            st.session_state.last_review = review.review
            st.session_state.last_thoughts = review.thoughts
    except Exception as e:
        st.error(f"Error: {e}")
        print(e)


if st.session_state.last_fig is not None:
    st.plotly_chart(st.session_state.last_fig, use_container_width=True)
    if st.session_state.last_review:
        st.subheader("Agent Review")
        st.markdown(st.session_state.last_review)
        st.subheader("Agent Chain of Thought")
        st.markdown(st.session_state.last_thoughts)

