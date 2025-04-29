import streamlit as st
import pandas as pd

# Evidently v0.7+ imports
from evidently import Dataset, DataDefinition, Report
from evidently.descriptors import (
    Sentiment,
    TextLength,
    IncludesWords,
    LLMGeneratedRelevance,
    LLMGeneratedSimilarity,
)
from evidently.presets import TextEvals
from evidently.metrics import MeanValue, MaxValue, ValueRange

# Import your custom LLM function (do not define here)
from your_llm_module import abc_response

# Set your local embedding model path here
LOCAL_EMBEDDING_MODEL = "./local_models/all-MiniLM-L6-v2"

st.set_page_config(page_title="LLM Evaluation with Evidently AI", layout="wide")
st.title("Streamlit LLM App with Evidently AI Evaluation (v0.7+)")

# User input
prompt = st.text_area("Enter your prompt:")
context = st.text_area("Optional: Add context for relevance evaluation")
reference = st.text_area("Optional: Add reference answer for similarity evaluation")

if st.button("Get LLM Response"):
    response = abc_response(prompt)
    st.markdown("**LLM Response:**")
    st.write(response)

    # Prepare data for evaluation
    data = pd.DataFrame([{
        "prompt": prompt,
        "answer": response,
        "context": context,
        "reference": reference,
    }])

    # Define data definition and descriptors
    data_def = DataDefinition(
        text_columns=["prompt", "answer", "context", "reference"]
    )
    descriptors = [
        Sentiment("answer", alias="Sentiment"),
        TextLength("answer", alias="Length"),
        IncludesWords("answer", words_list=["important"], alias="HasImportant"),
        LLMGeneratedRelevance(
            answer_column="answer",
            context_column="context",
            alias="Relevance",
            embedding_model_name_or_path=LOCAL_EMBEDDING_MODEL
        ),
        LLMGeneratedSimilarity(
            answer_column="answer",
            reference_column="reference",
            alias="Similarity",
            embedding_model_name_or_path=LOCAL_EMBEDDING_MODEL
        ),
    ]

    dataset = Dataset.from_pandas(
        data,
        data_definition=data_def,
        descriptors=descriptors,
    )

    # Built-in and aggregate metrics
    report = Report([
        TextEvals(),  # Aggregates all standard text/LLM descriptors
        MeanValue(column="Sentiment"),
        MaxValue(column="Length"),
        ValueRange(column="Relevance"),
        ValueRange(column="Similarity"),
    ])
    report.run(dataset)

    # Show Evidently report in the UI
    st.subheader("Evidently LLM/Text Evaluation Report")
    st.components.v1.html(report.get_html(), height=600, scrolling=True)

    # Show descriptor values (row-level)
    st.subheader("Descriptor Values (Row-level)")
    st.dataframe(dataset.to_pandas())

    # Example: Custom metric logic (not a descriptor, just for demo)
    st.subheader("Custom Metrics Example")
    has_important = "important" in response.lower()
    st.write(f"Does the answer contain 'important'? {'Yes' if has_important else 'No'}")

    # LLM-as-Judge (advanced): Use your LLM to self-evaluate
    judge_prompt = (
        f"Evaluate the helpfulness and accuracy of this response.\n"
        f"Prompt: {prompt}\n"
        f"Response: {response}\n"
        f"Context: {context}\n"
        f"Reference: {reference}\n"
        f"Score from 1 (poor) to 5 (excellent) and explain your reasoning."
    )
    judge_result = abc_response(judge_prompt)
    st.subheader("LLM-as-Judge Evaluation")
    st.write(judge_result)
