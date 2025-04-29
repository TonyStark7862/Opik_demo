import pandas as pd
import time
import os
import datetime
import schedule # For easy scheduling: pip install schedule
import random

# --- Evidently Imports ---
import evidently
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TextOverviewPreset
# Import metrics used in the Streamlit app to keep monitoring consistent
from evidently.metrics import (
    TextLengthMetric,
    SentimentMetric,
)
from evidently.metrics.llm import (
    LLMQualityMetric,
    LLMRelevanceMetric,
    LLMHelpfulnessMetric,
    LLMCoherenceMetric,
)
from evidently.ui.workspace import Workspace, WorkspaceOptions

# --- Custom Metrics Import ---
from custom_metrics import ResponseTimeMetric, KeywordPresenceMetric

# --- Placeholder for User's LLM API (needed for LLM-as-Judge) ---
# IMPORTANT: Replace this with your actual import or definition
# from llm_api import abc_response
def abc_response(prompt: str) -> str:
    """
    Placeholder for your custom LLM API function.
    Replace this with your actual implementation.
    Needed here if using LLM-as-Judge metrics.
    """
    # Simulate different responses
    time.sleep(0.1) # Faster simulation for monitor
    if "capital of france" in prompt.lower():
        return "The capital of France is Paris." if random.random() > 0.1 else "Paris." # Simulate variation
    else:
        return f"Monitored response to: '{prompt}'"

# --- Configuration ---
WORKSPACE_PATH = "evidently_workspace"
PROJECT_NAME = "My Streamlit LLM App" # MUST match the Streamlit app for combined view
REFERENCE_DATA_FILE = "reference_data.csv"
LOG_DATA_FILE = "llm_interactions_log.csv" # File where interactions *could* be logged

# --- Initialize Evidently Workspace & Project ---
# (Error handling omitted for brevity, assumes workspace/project exist)
WORKSPACE = Workspace(WORKSPACE_PATH)
PROJECT = WORKSPACE.get_project(PROJECT_NAME)
if not PROJECT:
    print(f"Error: Project '{PROJECT_NAME}' not found in workspace '{WORKSPACE_PATH}'.")
    print("Please run the Streamlit app first to create the project or check the name.")
    exit() # Or handle project creation if needed

# --- Load Reference Data ---
try:
    reference_df = pd.read_csv(REFERENCE_DATA_FILE)
    print(f"Loaded reference data from {REFERENCE_DATA_FILE}")
except Exception as e:
    print(f"Error loading reference data: {e}. Exiting.")
    exit()

# --- Monitoring Job Function ---
def run_monitoring_job():
    """Reads recent data, runs evaluations, and saves a snapshot."""
    print(f"\n[{datetime.datetime.now()}] Running monitoring job for project '{PROJECT_NAME}'...")

    # --- Simulate Reading Recent Data ---
    # In a real scenario, read data logged since the last run.
    # Example: Read the last N rows from a log file, or query a database.
    # Here, we just create some dummy data for demonstration.
    num_new_interactions = random.randint(2, 5)
    print(f"Simulating {num_new_interactions} new interactions...")
    prompts = [
        "What is the capital of France?",
        "Tell me a joke.",
        "Explain quantum physics simply.",
        "Who won the world cup in 2022?",
        "Summarize this text: [some long text]"
    ]
    current_data_list = []
    for i in range(num_new_interactions):
        prompt = random.choice(prompts)
        start_time = time.time()
        response = abc_response(prompt) # Use placeholder
        response_time = time.time() - start_time
        current_data_list.append({
            'prompt': prompt,
            'response': response,
            'response_time_sec': response_time,
            'timestamp': datetime.datetime.now() - datetime.timedelta(seconds=random.randint(1,60)),
        })
    current_df = pd.DataFrame(current_data_list)
    print(f"Generated {len(current_df)} new data points for evaluation.")

    if len(current_df) == 0:
        print("No new data to process.")
        return

    # --- Define Column Mapping ---
    column_mapping = ColumnMapping()
    column_mapping.text_features = ['prompt', 'response']

    # --- Define Metrics for Monitoring ---
    # Often similar to the interactive app, but maybe focused more on drift/aggregates
    monitoring_metrics = [
        DataDriftPreset(), # Check for drift compared to reference
        TextOverviewPreset(column_name="response"), # Basic text stats
        TextLengthMetric(column_name="response"),
        SentimentMetric(column_name="response"),
        ResponseTimeMetric(column_name='response_time_sec'),
        KeywordPresenceMetric(column_name='response', keywords=[' evidently', ' monitor', ' llm']),

        # Include LLM-as-Judge metrics if desired for monitoring batches
        LLMQualityMetric(eval_llm=abc_response, column_name="response", prompt_column="prompt"),
        LLMRelevanceMetric(eval_llm=abc_response, column_name="response", prompt_column="prompt"),
        # Add others as needed...
    ]

    # --- Run Evaluation ---
    print("Running Evidently evaluation...")
    monitor_report = Report(metrics=monitoring_metrics)
    monitor_report.run(
        current_data=current_df,
        reference_data=reference_df,
        column_mapping=column_mapping
    )
    print("Evaluation complete.")

    # --- Save Snapshot ---
    try:
        snapshot = monitor_report.as_snapshot()
        snapshot.metadata['app_name'] = PROJECT_NAME
        snapshot.metadata['run_type'] = 'periodic_monitor'
        snapshot.metadata['num_interactions'] = len(current_df)
        snapshot.tags = ["monitoring_run", "batch_evaluation"]

        PROJECT.add_snapshot(snapshot)
        PROJECT.save() # Save project metadata updates
        print(f"Snapshot saved successfully to Project '{PROJECT_NAME}'.")
        print(f"View updated dashboard: evidently ui --workspace {WORKSPACE_PATH}")
    except Exception as e:
        print(f"Error saving snapshot: {e}")

# --- Schedule the Job ---
print("Starting monitoring scheduler...")
# Schedule to run every 1 minute for demonstration
schedule.every(1).minutes.do(run_monitoring_job)
# Other options:
# schedule.every().hour.do(run_monitoring_job)
# schedule.every().day.at("10:30").do(run_monitoring_job)
# schedule.every(5).to(10).minutes.do(run_monitoring_job)

# Run the first job immediately
run_monitoring_job()

while True:
    schedule.run_pending()
    time.sleep(1)

