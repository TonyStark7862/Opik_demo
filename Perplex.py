import streamlit as st
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import LLMQualityPreset
from evidently.metrics import (
    TextLengthMetric, TextSentimentMetric, TextToxicityMetric,
    TextLanguageMetric, LLMGeneratedTextSimilarityMetric,
    LLMGeneratedTextRelevanceMetric,
)
from evidently.metrics.base_metric import LLMMetric
from evidently.metrics.llm.llm_base_metric import LLMMetricResult
from evidently.ui.workspace import Workspace

# Import your custom LLM function
from your_llm_module import abc_response

# Example local embedding model path
LOCAL_EMBEDDING_MODEL_PATH = "./local_models/all-MiniLM-L6-v2"

# Built-in metrics to demonstrate
BUILTIN_METRICS = [
    TextLengthMetric(),
    TextSentimentMetric(),
    TextToxicityMetric(),
    TextLanguageMetric(),
    LLMGeneratedTextSimilarityMetric(
        reference_column_name="reference",
        embedding_model_name_or_path=LOCAL_EMBEDDING_MODEL_PATH,
    ),
    LLMGeneratedTextRelevanceMetric(
        context_column_name="context",
        embedding_model_name_or_path=LOCAL_EMBEDDING_MODEL_PATH,
    ),
]

# Example custom metric: Checks if response contains a keyword
class ContainsKeywordMetric(LLMMetric):
    def __init__(self, keyword):
        self.keyword = keyword

    def calculate(self, data: pd.DataFrame) -> LLMMetricResult:
        results = data["llm_response"].apply(lambda x: self.keyword.lower() in x.lower())
        score = results.mean()
        return LLMMetricResult(
            name=f"Contains '{self.keyword}'",
            value=score,
            description=f"Fraction of responses containing '{self.keyword}'.",
        )

# Example custom metric: Penalize too short responses
class ShortResponsePenaltyMetric(LLMMetric):
    def __init__(self, min_length):
        self.min_length = min_length

    def calculate(self, data: pd.DataFrame) -> LLMMetricResult:
        results = data["llm_response"].apply(lambda x: len(x.split()) >= self.min_length)
        score = results.mean()
        return LLMMetricResult(
            name=f"Min {self.min_length} words",
            value=score,
            description=f"Fraction of responses with at least {self.min_length} words.",
        )

st.set_page_config(page_title="LLM Evaluation with Evidently AI", layout="wide")
st.title("Streamlit LLM App with Evidently AI Evaluation")

# Collect user prompt and context/reference for evaluation
prompt = st.text_area("Enter your prompt:")
context = st.text_area("Optional: Add context for relevance evaluation")
reference = st.text_area("Optional: Add reference answer for similarity evaluation")

if st.button("Get LLM Response"):
    response = abc_response(prompt)
    st.markdown("**LLM Response:**")
    st.write(response)

    # Prepare evaluation DataFrame
    eval_data = pd.DataFrame([{
        "prompt": prompt,
        "llm_response": response,
        "context": context,
        "reference": reference,
    }])

    # Built-in metrics
    st.subheader("Built-in LLM Metrics (Evidently AI)")
    report = Report(metrics=BUILTIN_METRICS)
    report.run(eval_data)
    st.components.v1.html(report.get_html(), height=600, scrolling=True)

    # Custom metrics
    st.subheader("Custom Metrics")
    custom_metrics = [
        ContainsKeywordMetric("important"),
        ShortResponsePenaltyMetric(10),
    ]
    for metric in custom_metrics:
        result = metric.calculate(eval_data)
        st.write(f"{result.name}: {result.value:.2f} ({result.description})")

    # LLM-as-judge (advanced)
    st.subheader("LLM-as-Judge Evaluation")
    judge_prompt = (
        f"Evaluate the following response for helpfulness and accuracy.\n"
        f"Prompt: {prompt}\n"
        f"Response: {response}\n"
        f"Context: {context}\n"
        f"Reference: {reference}\n"
        f"Score from 1 (poor) to 5 (excellent) and explain your reasoning."
    )
    judge_score = abc_response(judge_prompt)
    st.write("**LLM Judge Score and Explanation:**")
    st.write(judge_score)
