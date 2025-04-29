import streamlit as st
import pandas as pd
import time
import os
import datetime
import uuid # To generate unique IDs for interactions

# --- Evidently Imports ---
import evidently
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import TextOverviewPreset, ClassificationPreset # Example presets
from evidently.metrics import (
    ColumnSummaryMetric,
    TextLengthMetric,
    SentimentMetric,
    SentimentDistribution,
    TextDescriptorsDistribution,
    TextDescriptorsCorrelation,
    ColumnDriftMetric,
    DataDriftTable,
)
# LLM-specific metrics (might require additional setup/models)
from evidently.metrics.llm import (
    LLMQualityMetric,
    LLMRelevanceMetric,
    LLMHelpfulnessMetric,
    LLMCoherenceMetric,
    LLMCorrectnessMetric,
    LLMSimilarityMetric,
    LLMSafetyMetric,
    LLMPiiMetric,
    LLMSummarizationMetric,
    LLMRequirementsMetric,
    LLMTaskRequirement
)
from evidently.ui.workspace import Workspace, WorkspaceOptions
from evidently.pipeline.column_mapping import ColumnMapping as PipelineColumnMapping
from evidently.report import Report

# --- Custom Metrics Import ---
from custom_metrics import ResponseTimeMetric, KeywordPresenceMetric

# --- Placeholder for User's LLM API ---
# IMPORTANT: Replace this with your actual import or definition
# from llm_api import abc_response
def abc_response(prompt: str) -> str:
    """
    Placeholder for your custom LLM API function.
    Replace this with your actual implementation.
    It should take a string prompt and return a string response.
    """
    st.warning("Using placeholder `abc_response`. Replace with your actual LLM API call.")
    # Simulate some delay
    time.sleep(1.5)
    # Simulate different responses based on prompt
    if "hello" in prompt.lower():
        return "Hello there! How can I help you today?"
    elif "capital of france" in prompt.lower():
        return "The capital of France is Paris."
    elif "write a poem" in prompt.lower():
        return "Roses are red, violets are blue,\nEvidently helps monitor LLMs for you."
    else:
        return f"This is a simulated response to: '{prompt}'"

# --- Configuration ---
WORKSPACE_PATH = "evidently_workspace"
PROJECT_NAME = "My Streamlit LLM App" # Change this for different apps
REFERENCE_DATA_FILE = "reference_data.csv" # Example reference data

# --- Initialize Evidently Workspace ---
@st.cache_resource
def get_workspace(path):
    """Gets or creates the Evidently workspace."""
    if not os.path.exists(path):
        ws = Workspace.create(path)
        st.success(f"Created Evidently workspace at: {path}")
    else:
        ws = Workspace(path)
    return ws

@st.cache_resource
def get_project(_workspace, name):
    """Gets or creates an Evidently project within the workspace."""
    try:
        project = _workspace.create_project(name, exist_ok=True)
        project.description = f"Monitoring data for the LLM app: {name}"
        project.save()
        return project
    except Exception as e:
        st.error(f"Error creating/getting project '{name}': {e}")
        return None

WORKSPACE = get_workspace(WORKSPACE_PATH)
PROJECT = get_project(WORKSPACE, PROJECT_NAME)

# --- Load or Create Reference Data ---
@st.cache_data
def load_reference_data(file_path):
    """Loads reference data or creates a default if not found."""
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            st.error(f"Error loading reference data from {file_path}: {e}")
            # Fallback to default
    st.warning(f"Reference data file '{file_path}' not found. Creating default reference data.")
    # Create some sample reference data (prompts and ideal/reference answers)
    ref_data = pd.DataFrame({
        'prompt': [
            "What is the capital of France?",
            "Summarize the main idea: The quick brown fox jumps over the lazy dog.",
            "Are you helpful?"
        ],
        'reference_answer': [
            "Paris is the capital of France.",
            "The sentence describes a fast fox jumping over a slow dog.",
            "Yes, I strive to be helpful."
        ],
        # Add other relevant columns if needed for specific metrics
        'expected_sentiment': [0.0, 0.0, 0.5] # Example: Neutral, Neutral, Positive
    })
    try:
        ref_data.to_csv(file_path, index=False)
        st.success(f"Created default reference data at: {file_path}")
    except Exception as e:
        st.error(f"Could not save default reference data to {file_path}: {e}")
    return ref_data

reference_df = load_reference_data(REFERENCE_DATA_FILE)

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title(f"🧪 Evidently LLM Observability Demo ({PROJECT_NAME})")
st.markdown(f"Using workspace: `{WORKSPACE_PATH}` | Project: `{PROJECT_NAME}`")
st.info("Enter a prompt, get a response from the LLM, and see Evidently AI evaluations.")

# --- LLM Interaction Area ---
st.header("💬 Interact with LLM")
prompt_input = st.text_area("Enter your prompt:", height=100, key="prompt_input")

if st.button("Get Response & Evaluate", key="evaluate_button"):
    if not prompt_input:
        st.warning("Please enter a prompt.")
    else:
        st.subheader("LLM Interaction")
        with st.spinner("Calling LLM API and evaluating..."):
            # --- Call LLM and measure time ---
            start_time = time.time()
            try:
                llm_response = abc_response(prompt_input)
                response_time = time.time() - start_time
                st.markdown("**LLM Response:**")
                st.info(llm_response) # Use info box for response
                st.markdown(f"*(Response time: {response_time:.2f} seconds)*")

                # --- Prepare data for Evidently ---
                # Create a DataFrame for the current interaction
                current_interaction_data = pd.DataFrame({
                    'prompt': [prompt_input],
                    'response': [llm_response],
                    'response_time_sec': [response_time],
                    'timestamp': [datetime.datetime.now()],
                    'interaction_id': [str(uuid.uuid4())] # Unique ID for this interaction
                })

                # --- Define Column Mapping ---
                # Tells Evidently which columns contain what type of data
                column_mapping = ColumnMapping()
                column_mapping.text_features = ['prompt', 'response'] # Features to analyze as text
                # If using classification metrics or reference-based metrics:
                # column_mapping.target = 'expected_sentiment' # Example if evaluating sentiment against reference
                # column_mapping.prediction = 'actual_sentiment_score' # Example if a model predicts sentiment
                # column_mapping.id = 'interaction_id'
                # column_mapping.datetime = 'timestamp'

                # --- Build Evidently Report ---
                st.subheader("📊 Evidently Evaluation Report")

                # Define metrics to include
                report_metrics = [
                    # === Built-in Text Descriptors ===
                    TextLengthMetric(column_name="response", description="Length of the LLM response."),
                    SentimentMetric(column_name="response", description="Sentiment score of the response."),
                    # Requires extra packages (nltk, potentially models) - uncomment if installed
                    # PIIPresenceMetric(column_name="response"),
                    # ToxicityMetric(column_name="response"),
                    # LanguageMetric(column_name="response"),

                    # === Custom Metrics ===
                    ResponseTimeMetric(column_name='response_time_sec', description="Custom metric for response time."),
                    KeywordPresenceMetric(column_name='response', keywords=[' evidently', ' monitor', ' llm'], description="Checks for specific keywords."),

                    # === LLM Quality Metrics (using LLM-as-Judge) ===
                    # IMPORTANT: These require configuring an LLM for evaluation.
                    # We'll use the *same* custom LLM API as the judge here for simplicity.
                    # In practice, you might use a different, potentially stronger model.
                    # Requires defining the evaluation prompts or using defaults.
                    LLMQualityMetric(
                        eval_llm=abc_response, # Use your API as the judge
                        column_name="response",
                        prompt_column="prompt",
                        description="Overall quality judged by LLM.",
                        # You can customize the evaluation prompt/aspects
                        # aspect="Overall quality assessment."
                    ),
                    LLMRelevanceMetric(
                         eval_llm=abc_response,
                         column_name="response",
                         prompt_column="prompt",
                         description="Relevance to prompt judged by LLM."
                    ),
                    LLMHelpfulnessMetric(
                         eval_llm=abc_response,
                         column_name="response",
                         prompt_column="prompt",
                         description="Helpfulness judged by LLM."
                    ),
                    LLMCoherenceMetric(
                         eval_llm=abc_response,
                         column_name="response",
                         prompt_column="prompt",
                         description="Coherence judged by LLM."
                    ),
                    # Example requiring reference data:
                    # LLMCorrectnessMetric(
                    #      eval_llm=abc_response,
                    #      column_name="response",
                    #      prompt_column="prompt",
                    #      reference_column="reference_answer", # Needs corresponding reference answer
                    #      description="Correctness vs reference judged by LLM."
                    # ),

                    # === Data Drift/Overview (Comparing current to reference) ===
                    # These work best when you have more data points in 'current_interaction_data'
                    # or run them periodically on batches (see monitor.py)
                    # DataDriftTable(), # Compares distributions between current and reference
                    # TextOverviewPreset(column_name="response") # Provides overview stats
                ]

                # Create and run the report
                # Compare the single current interaction to the reference dataset
                llm_report = Report(metrics=report_metrics)
                llm_report.run(
                    current_data=current_interaction_data,
                    reference_data=reference_df, # Compare against the reference data
                    column_mapping=column_mapping
                )

                # Display the report within Streamlit
                # Use save_html and components.html to render
                report_path = f"temp_report_{current_interaction_data['interaction_id'].iloc[0]}.html"
                llm_report.save_html(report_path)
                with open(report_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=800, scrolling=True)
                os.remove(report_path) # Clean up temporary file

                # --- Save Snapshot to Workspace ---
                if PROJECT:
                    st.subheader("💾 Save to Evidently Workspace")
                    if st.button("Save Evaluation Snapshot", key="save_snapshot"):
                        try:
                            # Add metadata to the snapshot
                            snapshot = llm_report.as_snapshot()
                            snapshot.metadata['app_name'] = PROJECT_NAME
                            snapshot.metadata['interaction_id'] = current_interaction_data['interaction_id'].iloc[0]
                            snapshot.tags = ["streamlit_run", "llm_evaluation"]

                            PROJECT.add_snapshot(snapshot)
                            PROJECT.save() # Save project metadata updates
                            st.success(f"Snapshot saved successfully to Project '{PROJECT_NAME}'!")
                            st.caption("You can now view this evaluation alongside others in the Evidently UI.")
                            st.code(f"Run: evidently ui --workspace {WORKSPACE_PATH}")
                        except Exception as e:
                            st.error(f"Error saving snapshot: {e}")
                else:
                    st.error("Cannot save snapshot: Evidently Project not initialized correctly.")

            except Exception as e:
                st.error(f"An error occurred during LLM call or evaluation: {e}")
                st.exception(e) # Show traceback for debugging

# --- Display Reference Data ---
st.sidebar.subheader("Reference Data")
st.sidebar.dataframe(reference_df, height=200)
st.sidebar.caption(f"Loaded from: `{REFERENCE_DATA_FILE}`")

# --- Instructions ---
st.sidebar.subheader("How to Use")
st.sidebar.markdown(f"""
1.  **Install:** `pip install -r requirements.txt`
2.  **Provide LLM API:** Replace the placeholder `abc_response` function in `app.py` with your actual LLM API logic.
3.  **Run Streamlit:** `streamlit run app.py`
4.  **Run Evidently UI (separate terminal):** `evidently ui --workspace {WORKSPACE_PATH}`
5.  Enter prompts, evaluate, and optionally save snapshots. View aggregated results in the Evidently UI (usually at `http://localhost:8000`).
""")
st.sidebar.subheader("Managing Multiple Apps")
st.sidebar.markdown(f"""
* To monitor a different LLM app, change the `PROJECT_NAME` variable in `app.py` and `monitor.py`.
* Each unique `PROJECT_NAME` will create a separate project within the `{WORKSPACE_PATH}` workspace in the Evidently UI.
""")

