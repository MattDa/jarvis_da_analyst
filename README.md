# LLM Data Visualizer

This Streamlit application lets you upload a CSV or Excel file and describe a desired visualization. A locally hosted `meta-llama/Llama-3.1-8B-Instruct` model (served by vLLM at `localhost:8000`) generates Plotly code to build the chart. A ReAct agent then reviews the resulting visualization for accuracy and relevance to your prompt and displays its reasoning.

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

Upload your dataset, provide a prompt, and the app will generate and review a Plotly visualization.
