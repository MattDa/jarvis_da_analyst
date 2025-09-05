# LLM Data Visualizer

This Streamlit application lets you upload a CSV or Excel file and chat with a locally hosted `meta-llama/Llama-3.1-8B-Instruct` model (served by vLLM at `localhost:8000`) to create Plotly visuals. A ReAct agent reviews each visualization for accuracy and relevance and shares its reasoning.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure a vLLM server is running locally:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

## Running

```bash
streamlit run app.py
```

Upload your dataset, then converse with the model about the chart you want. Each response returns the Python code used to build the Plotly figure, shows the chart, and displays the agent's review and chain of thought.
