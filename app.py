import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


st.set_page_config(page_title="LLM Plotter", layout="wide")
st.title("LLM Data Visualizer")


def ensure_session_state() -> None:
    """Initialize Streamlit session state for chat and memory."""
    if "llm" not in st.session_state:
        st.session_state.llm = ChatOpenAI(model="gpt-5", temperature=0)
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "df" not in st.session_state:
        st.session_state.df = None


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


def generate_plot(
    llm: ChatOpenAI,
    memory: ConversationBufferMemory,
    df: pd.DataFrame,
    user_prompt: str,
):
    preview = df.head().to_csv(index=False)
    messages = [
        SystemMessage(content="You are a data visualization expert using Plotly in Python."),
        *memory.chat_memory.messages,
        HumanMessage(
            content=(
                "Given the following data sample:\n" + preview +
                f"\nCreate Plotly code that uses the whole dataset to satisfy the instruction: {user_prompt}. "
                "Use variable df for the dataset."
                "Assign the resulting Plotly figure to a variable named 'fig'."
                "Answer with just the code."
            )
        ),
    ]
    response = llm.invoke(messages)
    code = response.content
    local_vars = {"df": df, "px": px, "go": go}
    code = code.replace("```", "").replace("python", "", 1)
    exec(code, local_vars)
    fig = local_vars.get("fig")
    memory.chat_memory.add_user_message(user_prompt)
    memory.chat_memory.add_ai_message(code)
    return fig, code


ensure_session_state()
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
if uploaded:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        df = ensure_consistent_dtypes(df)
        st.session_state.df = df
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg.get("figure") is not None:
            st.plotly_chart(msg["figure"], use_container_width=True)
            st.code(msg["content"])
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Describe the visualization you want"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    df = st.session_state.get("df")
    if df is None:
        response = "Please upload a dataset first."
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
    else:
        try:
            fig, code = generate_plot(st.session_state.llm, st.session_state.memory, df, prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": code, "figure": fig})
            with st.chat_message("assistant"):
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                st.code(code)
        except Exception as e:
            error_msg = f"Error: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.markdown(error_msg)

