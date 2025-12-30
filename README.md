# LLM Data Visualizer
For use in air gapped, security heightened environments where access to modern agents are not available.

A Streamlit app for uploading CSV/Excel data and chatting with a locally hosted `meta-llama/Llama-3.1-8B-Instruct` model (served by vLLM on `localhost:8000`) to generate Plotly visualizations. Each chart is reviewed by a ReAct-style agent that provides accuracy/relevance feedback and displays its reasoning beneath the chart. The UI uses a dark theme.

Plotly is chosen over more mainstream options like Mermaid because it natively includes user-interactibility.

## Features

- Upload CSV/XLSX data and coerce columns to consistent dtypes.
- (Coming) Upload DOCX/PDF/TXT with textual representations of datatsets for analytics.
- In-memory chat thread (LangChain) with a local vLLM-backed Llama model.
- Plotly chart generation from user prompts.
- ReAct review agent that critiques the visualization.

## Requirements

- Python 3.10+
- vLLM server running locally with `meta-llama/Llama-3.1-8B-Instruct` on port 8000

## Local Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Start vLLM (in a separate terminal):

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

3. Run the app:

```bash
streamlit run app.py
```

## Docker Workflow

The application can be containerized and run against an existing vLLM container or host.

### Build

```bash
docker build -t llm-data-visualizer .
```

### Run

If vLLM is running on the host at port 8000:

```bash
docker run --rm -p 8501:8501 --add-host=host.docker.internal:host-gateway \
  -e OPENAI_BASE_URL=http://host.docker.internal:8000/v1 \
  llm-data-visualizer
```

If vLLM is in another container on a shared network (named `llm-net` here):

```bash
docker network create llm-net

docker run --rm -d --name vllm --network llm-net -p 8000:8000 \
  vllm/vllm-openai:latest --model meta-llama/Llama-3.1-8B-Instruct

docker run --rm -p 8501:8501 --network llm-net \
  -e OPENAI_BASE_URL=http://vllm:8000/v1 \
  llm-data-visualizer
```

Then open `http://localhost:8501` in your browser.

## Usage

1. Upload a CSV or Excel file.
2. Ask the model to create a chart (e.g., “Plot monthly revenue as a line chart”).
3. Review the generated Plotly visualization and the ReAct critique shown below it.

## Notes

- The app uses an in-memory chat history via `st.session_state`; no database is required.
- Ensure your vLLM server is reachable at the `OPENAI_BASE_URL` configured in your environment.
