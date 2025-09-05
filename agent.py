import json
import os
from dataclasses import dataclass
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


@dataclass
class ReviewResult:
    review: str
    thoughts: str


class ReviewAgent:
    """ReAct-style agent that critiques a Plotly visualization."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "llm")
        self.llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model)

    def _chat(self, messages: list) -> str:
        response = self.llm.invoke(messages)
        return response.content.strip()

    def review_visual(self, prompt: str, figure_spec: Dict[str, Any]) -> ReviewResult:
        fig_json = json.dumps(figure_spec)
        messages = [
            SystemMessage(
                content=(
                    "You are a ReAct agent that reviews data visualizations. "
                    "Use step-by-step reasoning to determine if the visualization is accurate and relevant to the user's prompt. "
                    "Respond in JSON with keys 'review' and 'thoughts'."
                )
            ),
            HumanMessage(
                content=(
                    f"User prompt: {prompt}\n" +
                    f"Visualization spec (Plotly JSON): {fig_json}"
                )
            ),
        ]
        content = self._chat(messages)
        try:
            data = json.loads(content)
            review = data.get("review", "")
            thoughts = data.get("thoughts", "")
        except json.JSONDecodeError:
            review = content
            thoughts = ""
        return ReviewResult(review=review, thoughts=thoughts)

