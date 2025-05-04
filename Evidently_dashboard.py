import streamlit as st
import pandas as pd
import os
import time
from streamlit_autorefresh import st_autorefresh

# --- Evidently AI Imports ---
from evidently import ColumnMapping
from evidently.report import Report
from evidently.test_suite import TestSuite
# Import metrics and tests to build reports/suites manually
from evidently.metric_preset import DataSummary # To get column summaries
from evidently.metrics import ColumnSummaryMetric, ColumnValuePlot # Examples
from evidently.tests import * # Import all tests for flexibility

# --- Configuration ---
CSV_FILE = 'interactions.csv'
REFRESH_INTERVAL_SECONDS = 10 # Auto-refresh interval

# --- Helper Function to load data ---
@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS) # Cache data for short period
def load_data(file_path):
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame() # Return empty df if file is empty
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

# --- Main Dashboard Logic ---
st.set_page_config(layout="wide")
st.title("📊 LLM Observability Dashboard (Evidently AI)")

# Auto-refresh controller
refresh_count = st_autorefresh(interval=REFRESH_INTERVAL_SECONDS * 1000, key="dashboard_refresh")

# Load data
data_df = load_data(CSV_FILE)

if data_df.empty:
    st.warning(f"No interaction data found yet in '{CSV_FILE}'. Please interact with the chat app.")
else:
    st.success(f"Loaded {len(data_df)} interactions. Last refresh: {time.strftime('%H:%M:%S')}")
    st.dataframe(data_df.tail())

    # --- Define Column Mapping for Evidently ---
    # Dynamically create mapping based on columns present in the loaded CSV
    # This makes it robust to which evaluations were successful in app.py
    numerical_cols = data_df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = data_df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Remove columns we don't want treated purely as categorical for typical reports
    # (adjust based on specific needs)
    cols_to_remove_from_cat = ['session_id', 'interaction_id', 'timestamp', 'prompt', 'answer']
    # Identify custom eval columns that should be treated as categorical
    custom_label_cols = [col for col in data_df.columns if col.endswith('_label')]

    categorical_cols = [
        col for col in categorical_cols
        if col not in cols_to_remove_from_cat and col not in custom_label_cols
    ] + custom_label_cols # Ensure custom labels are categorical

    # Example: Manually specify if needed, but dynamic is safer
    column_mapping = ColumnMapping()
    # Let Evidently handle defaults for numerical/categorical, but you could map explicitly:
    # column_mapping.numerical_features = [...]
    # column_mapping.categorical_features = [...]


    # --- Generate Evidently Report ---
    st.header("Evidently Report")
    report_placeholder = st.empty()
    report_placeholder.info("Generating report...")

    try:
        # Build report manually with desired metrics
        report = Report(metrics=[
            DataSummary(), # Overall dataset stats
            ColumnSummaryMetric(column_name="latency_ms"),
            ColumnSummaryMetric(column_name="text_length_answer"),
            ColumnSummaryMetric(column_name="semantic_similarity_prompt_answer"),
            ColumnSummaryMetric(column_name="custom_sentiment_score"), # For custom score
            # Add summaries for custom labels (will show counts/frequencies)
            ColumnSummaryMetric(column_name="custom_toxicity_label"),
            ColumnSummaryMetric(column_name="custom_relevance_label"),
            ColumnSummaryMetric(column_name="custom_pii_label"),
            ColumnSummaryMetric(column_name="custom_decline_label"),
            ColumnSummaryMetric(column_name="custom_bias_label"),
            ColumnSummaryMetric(column_name="custom_compliance_label"),
            # Add more specific plots if desired
            # ColumnValuePlot(column_name="latency_ms"),
            # ColumnValuePlot(column_name="semantic_similarity_prompt_answer"),
        ])
        report.run(current_data=data_df, reference_data=None, column_mapping=column_mapping)

        # Save report to HTML
        report_path = "evidently_report.html"
        report.save_html(report_path)

        # Display HTML report in Streamlit
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        report_placeholder.empty() # Clear the 'generating' message
        with st.expander("View Full Evidently Report", expanded=True):
            st.components.v1.html(html_content, height=600, scrolling=True)

    except Exception as e:
        report_placeholder.error(f"Failed to generate Evidently Report: {e}")
        st.exception(e) # Show full traceback for debugging

    # --- Generate Evidently Test Suite ---
    st.header("Evidently Test Suite")
    test_suite_placeholder = st.empty()
    test_suite_placeholder.info("Generating test suite...")

    try:
        # Define tests based on available columns
        tests_to_run = []
        if "latency_ms" in data_df.columns:
            tests_to_run.append(TestMeanValue(column_name="latency_ms", lte=2000, is_critical=False)) # Warning if avg latency > 2s
        if "text_length_answer" in data_df.columns:
            tests_to_run.append(TestMeanValue(column_name="text_length_answer", gte=10)) # Must have some length
            tests_to_run.append(TestMeanValue(column_name="text_length_answer", lte=1000, is_critical=False)) # Warn if too long avg
        if "semantic_similarity_prompt_answer" in data_df.columns:
             # Check if column has non-NA values before testing
            if data_df["semantic_similarity_prompt_answer"].notna().any():
                 tests_to_run.append(TestMeanValue(column_name="semantic_similarity_prompt_answer", gte=0.3, is_critical=False)) # Warn if avg similarity is low
            else:
                 st.caption("Skipping Semantic Similarity tests as column contains only NA values.")

        if "custom_sentiment_score" in data_df.columns and data_df["custom_sentiment_score"].notna().any():
            tests_to_run.append(TestMeanValue(column_name="custom_sentiment_score", gte=0.0)) # Avg sentiment should be non-negative
        if "custom_toxicity_label" in data_df.columns:
            tests_to_run.append(TestCategoryShare(column_name="custom_toxicity_label", category="Toxic", lte=0.1)) # Less than 10% toxic
        if "custom_relevance_label" in data_df.columns:
            tests_to_run.append(TestCategoryShare(column_name="custom_relevance_label", category="Irrelevant", lte=0.2, is_critical=False)) # Warn if > 20% irrelevant
        if "custom_pii_label" in data_df.columns:
             tests_to_run.append(TestCategoryShare(column_name="custom_pii_label", category="PII Detected", eq=0)) # Should be 0 PII detected
        if "custom_decline_label" in data_df.columns:
            tests_to_run.append(TestCategoryShare(column_name="custom_decline_label", category="Declined", lte=0.15, is_critical=False)) # Warn if > 15% declined
        if "custom_bias_label" in data_df.columns:
            tests_to_run.append(TestCategoryShare(column_name="custom_bias_label", category="Bias Detected", lte=0.05)) # Less than 5% biased
        if "custom_compliance_label" in data_df.columns:
            tests_to_run.append(TestCategoryShare(column_name="custom_compliance_label", category="Not Compliant", eq=0)) # Should be 0 non-compliant


        if tests_to_run:
            test_suite = TestSuite(tests=tests_to_run)
            test_suite.run(current_data=data_df, reference_data=None, column_mapping=column_mapping)

            # Save test suite to HTML
            test_suite_path = "evidently_tests.html"
            test_suite.save_html(test_suite_path)

            # Display HTML test suite in Streamlit
            with open(test_suite_path, 'r', encoding='utf-8') as f:
                html_content_tests = f.read()
            test_suite_placeholder.empty() # Clear the 'generating' message
            with st.expander("View Full Evidently Test Suite Results", expanded=True):
                st.components.v1.html(html_content_tests, height=600, scrolling=True)
        else:
             test_suite_placeholder.warning("No applicable tests could be configured based on available data columns.")


    except Exception as e:
        test_suite_placeholder.error(f"Failed to generate Evidently Test Suite: {e}")
        st.exception(e) # Show full traceback for debugging

    # --- Optional: Add specific Streamlit plots/metrics ---
    st.header("Key Metrics Overview")
    col1, col2, col3 = st.columns(3)
    if "latency_ms" in data_df.columns:
        avg_latency = data_df["latency_ms"].mean()
        col1.metric("Avg Latency (ms)", f"{avg_latency:.2f}")
    if "custom_toxicity_label" in data_df.columns:
        toxic_count = (data_df["custom_toxicity_label"] == "Toxic").sum()
        col2.metric("Toxic Responses Count", f"{toxic_count}")
    if "semantic_similarity_prompt_answer" in data_df.columns:
        avg_similarity = data_df["semantic_similarity_prompt_answer"].mean() # Might be NaN if all failed
        col3.metric("Avg Semantic Similarity", f"{avg_similarity:.3f}" if pd.notna(avg_similarity) else "N/A")

    st.subheader("Latency Over Time")
    if "timestamp" in data_df.columns and "latency_ms" in data_df.columns:
        # Convert timestamp if it's not already datetime
        if not pd.api.types.is_datetime64_any_dtype(data_df['timestamp']):
             try:
                 data_df['timestamp_dt'] = pd.to_datetime(data_df['timestamp'])
                 st.line_chart(data_df.set_index('timestamp_dt')['latency_ms'])
             except Exception as e:
                 st.warning(f"Could not parse timestamp for chart: {e}")
        else:
             st.line_chart(data_df.set_index('timestamp')['latency_ms'])
