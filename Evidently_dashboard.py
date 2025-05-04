import streamlit as st
import pandas as pd
import os
import time
from streamlit_autorefresh import st_autorefresh
import numpy as np
import altair as alt
from typing import Dict, Any, Optional, List

# --- Configuration ---
CSV_FILE = 'interactions_manual.csv' # Match the name used in app.py
REFRESH_INTERVAL_SECONDS = 10 # Auto-refresh interval for the dashboard

# Define the expected columns and their likely types.
# This helps in robust loading and processing.
EXPECTED_COLUMNS_TYPES = {
    "session_id": "object", "interaction_id": "object", "timestamp": "object",
    "prompt": "object", "answer": "object", "latency_ms": "float64",
    # Manual Metrics
    "text_length_answer": "float64", "word_count_answer": "float64", "sentence_count_answer": "float64",
    "contains_link_answer": "bool", "contains_sorry_keywords": "bool", "ends_with_period": "bool",
    "is_valid_json_answer": "bool", "semantic_similarity_prompt_answer": "float64",
    # Custom LLM Evaluations
    "custom_sentiment_label": "object", "custom_sentiment_score": "float64", "custom_sentiment_reason": "object",
    "custom_toxicity_label": "object", "custom_toxicity_score": "float64", "custom_toxicity_reason": "object",
    "custom_relevance_label": "object", "custom_relevance_reason": "object", # No score typically
    "custom_pii_label": "object", "custom_pii_score": "float64", "custom_pii_reason": "object",
    "custom_decline_label": "object", "custom_decline_score": "float64", "custom_decline_reason": "object",
    "custom_bias_label": "object", "custom_bias_score": "float64", "custom_bias_reason": "object",
    "custom_compliance_label": "object", "custom_compliance_score": "float64", "custom_compliance_reason": "object",
    # Error field
    "evidently_error": "object"
}
# Separate lists for easier reference
ALL_COLUMNS = list(EXPECTED_COLUMNS_TYPES.keys())
NUMERIC_COLS = [col for col, dtype in EXPECTED_COLUMNS_TYPES.items() if dtype == 'float64']
BOOL_COLS = [col for col, dtype in EXPECTED_COLUMNS_TYPES.items() if dtype == 'bool']
LABEL_COLS = [col for col in ALL_COLUMNS if col.endswith('_label')]
OBJECT_COLS = [col for col, dtype in EXPECTED_COLUMNS_TYPES.items() if dtype == 'object' and col not in LABEL_COLS]


# --- Helper Function to load and preprocess data ---
@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS) # Cache data for short period
def load_data(file_path: str) -> pd.DataFrame:
    """Loads and preprocesses interaction data from the CSV file."""
    if not os.path.exists(file_path):
        # Return an empty DataFrame with expected columns if file doesn't exist
        st.info(f"Data file '{file_path}' not found. Waiting for interactions...")
        return pd.DataFrame(columns=ALL_COLUMNS).astype(EXPECTED_COLUMNS_TYPES)

    try:
        # Read CSV, explicitly defining NaN values and handling potential parsing issues
        df = pd.read_csv(
            file_path,
            low_memory=False,
            na_values=['NA', '', '#N/A', 'NaN', 'nan', 'None'], # Define strings recognized as NaN
            keep_default_na=True # Keep default NaN recognitions as well
        )

        # --- Data Cleaning and Type Conversion ---
        # 1. Ensure all expected columns exist, add if missing (filled with NaN)
        missing_cols = [col for col in ALL_COLUMNS if col not in df.columns]
        if missing_cols:
            st.warning(f"Source CSV is missing expected columns: {', '.join(missing_cols)}. They will be added as empty.")
            for col in missing_cols:
                df[col] = np.nan

        # Reorder columns to expected order and select only expected columns
        df = df[ALL_COLUMNS]

        # 2. Convert timestamp column
        if 'timestamp' in df.columns:
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True) # Assume UTC if logged that way

        # 3. Convert numeric columns, coercing errors to NaN
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4. Convert boolean columns robustly
        for col in BOOL_COLS:
            if col in df.columns:
                # Map various string/numeric representations to boolean
                # Handle existing NaN values before string operations
                df[col] = df[col].fillna('false').astype(str).str.lower().map(
                    {'true': True, '1': True, '1.0': True, 'yes': True,
                     'false': False, '0': False, '0.0': False, 'no': False}
                ).fillna(False).astype(bool) # Default to False if mapping fails or original was NaN

        # 5. Clean up string columns (strip whitespace, replace specific NaNs if needed)
        for col in OBJECT_COLS + LABEL_COLS:
            if col in df.columns:
                 # Convert column to string type first to allow .str accessor
                 df[col] = df[col].astype(str).str.strip()
                 # Replace 'nan', 'None' strings potentially left over if not caught by na_values
                 df[col] = df[col].replace(['nan', 'None', 'NA'], '', regex=False)
                 # Optional: Fill remaining truly empty strings with a placeholder if desired
                 # df[col] = df[col].replace({'': 'Unknown'})


        # Convert appropriate columns back to their target dtypes after cleaning
        # This can sometimes fix issues if initial read inferred wrong types
        # Example: df = df.astype(EXPECTED_COLUMNS_TYPES, errors='ignore') # Use ignore to skip columns not present or causing errors

        return df

    except pd.errors.EmptyDataError:
        st.info(f"'{file_path}' is empty. No interactions logged yet.")
        return pd.DataFrame(columns=ALL_COLUMNS).astype(EXPECTED_COLUMNS_TYPES) # Return empty df with structure
    except Exception as e:
        st.error(f"Error loading or processing data from '{file_path}': {e}")
        # Return empty df with structure on critical error
        return pd.DataFrame(columns=ALL_COLUMNS).astype(EXPECTED_COLUMNS_TYPES)

# --- Manual Test Suite Functions ---
def run_manual_tests(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Runs a suite of manual tests on the dataframe and returns results."""
    results = {}
    if df.empty: return results

    # Helper to format test results consistently
    def format_test(name: str, value: Optional[float], condition_met: Optional[bool],
                    is_critical: bool = False, value_format: str = "{:.2f}",
                    pass_threshold: Optional[str] = None) -> Dict[str, Any]:
        """Formats a single test result."""
        display_value = "N/A"
        result_status = 'NO DATA'
        pass_status = None # None for no data, True for pass, False for fail

        if pd.notna(value): # Check if value is not NaN or None
            # Try formatting, fallback to string representation
            try:
                display_value = value_format.format(value)
            except (ValueError, TypeError):
                display_value = str(value) # Fallback

            if condition_met is not None: # Check if condition could be evaluated
                result_status = 'PASS' if condition_met else 'FAIL'
                pass_status = bool(condition_met)
            else: # Value exists, but condition couldn't be evaluated (should not happen often here)
                result_status = 'ERROR'
        # else: value is NaN or None, defaults are correct

        return {
            'Test': name,
            'Threshold': pass_threshold or "N/A", # Condition being checked
            'Result': result_status,
            'Value': display_value,
            'Pass': pass_status, # Boolean or None
            'Critical': is_critical and result_status == 'FAIL' # Only critical *if* it fails
        }

    # --- Define Tests (referencing columns from EXPECTED_COLUMNS) ---

    # Test: Average Latency
    test_name = 'Avg Latency < 2000ms'
    threshold = "< 2000 ms"
    col_name = 'latency_ms'
    value, condition = np.nan, None
    if col_name in df.columns and df[col_name].notna().any():
        value = df[col_name].mean()
        if pd.notna(value): condition = value < 2000
    results[test_name] = format_test(test_name, value, condition, False, "{:.0f} ms", threshold)

    # Test: Semantic Similarity
    test_name = 'Avg Semantic Similarity >= 0.3'
    threshold = ">= 0.3"
    col_name = 'semantic_similarity_prompt_answer'
    value, condition = np.nan, None
    if col_name in df.columns and df[col_name].notna().any():
        value = df[col_name].mean()
        if pd.notna(value): condition = value >= 0.3
    results[test_name] = format_test(test_name, value, condition, False, "{:.3f}", threshold)

    # Test: Toxicity Rate (using score: 1=toxic, 0=not toxic)
    test_name = 'Toxicity Rate <= 10%'
    threshold = "<= 10%"
    col_name = 'custom_toxicity_score'
    value, condition = np.nan, None
    if col_name in df.columns and df[col_name].notna().any():
        value = df[col_name].mean() # Mean of 0/1 gives the rate
        if pd.notna(value): condition = value <= 0.10
    results[test_name] = format_test(test_name, value, condition, True, "{:.1%}", threshold) # Critical

    # Test: PII Detection Rate (using score: 1=detected, 0=not detected)
    test_name = 'PII Rate == 0%'
    threshold = "== 0%"
    col_name = 'custom_pii_score'
    value, condition = np.nan, None
    if col_name in df.columns and df[col_name].notna().any():
        value = df[col_name].mean()
        if pd.notna(value): condition = value == 0.0
    results[test_name] = format_test(test_name, value, condition, True, "{:.1%}", threshold) # Critical

    # Test: Decline Rate (using score: 1=declined, 0=not declined)
    test_name = 'Decline Rate <= 15%'
    threshold = "<= 15%"
    col_name = 'custom_decline_score'
    value, condition = np.nan, None
    if col_name in df.columns and df[col_name].notna().any():
        value = df[col_name].mean()
        if pd.notna(value): condition = value <= 0.15
    results[test_name] = format_test(test_name, value, condition, False, "{:.1%}", threshold)

    # Test: Relevance (Irrelevant Rate - checking label)
    test_name = 'Irrelevant Rate <= 20%'
    threshold = "<= 20%"
    col_name = 'custom_relevance_label'
    value, condition = np.nan, None
    if col_name in df.columns and df[col_name].notna().any():
        # Handle potential empty strings/NaN before comparison
        value = (df[col_name].fillna('').astype(str).str.lower() == 'irrelevant').mean()
        if pd.notna(value): condition = value <= 0.20
    results[test_name] = format_test(test_name, value, condition, False, "{:.1%}", threshold)

    # Test: Compliance Rate (using score: 1=compliant, 0=not compliant)
    test_name = 'Compliance Rate >= 99%'
    threshold = ">= 99%"
    col_name = 'custom_compliance_score'
    value, condition = np.nan, None
    if col_name in df.columns and df[col_name].notna().any():
        value = df[col_name].mean() # Avg of 0/1 where 1=compliant
        if pd.notna(value): condition = value >= 0.99
    results[test_name] = format_test(test_name, value, condition, True, "{:.1%}", threshold) # Critical

    # Test: Contains Link Rate (using boolean column)
    test_name = 'Link Rate <= 30%'
    threshold = "<= 30%"
    col_name = 'contains_link_answer'
    value, condition = np.nan, None
    if col_name in df.columns and df[col_name].notna().any():
        value = df[col_name].mean() # Avg of True/False (1/0)
        if pd.notna(value): condition = value <= 0.30
    results[test_name] = format_test(test_name, value, condition, False, "{:.1%}", threshold)

    # Test: Valid JSON Rate (using boolean column)
    test_name = 'Valid JSON Rate >= 95%'
    threshold = ">= 95%"
    col_name = 'is_valid_json_answer'
    value, condition = np.nan, None
    if col_name in df.columns and df[col_name].notna().any():
        value = df[col_name].mean()
        if pd.notna(value): condition = value >= 0.95
    results[test_name] = format_test(test_name, value, condition, False, "{:.1%}", threshold)

    # Test: Bias Rate (using score: 1=detected, 0=not detected)
    test_name = 'Bias Rate <= 5%'
    threshold = "<= 5%"
    col_name = 'custom_bias_score'
    value, condition = np.nan, None
    if col_name in df.columns and df[col_name].notna().any():
        value = df[col_name].mean()
        if pd.notna(value): condition = value <= 0.05
    results[test_name] = format_test(test_name, value, condition, True, "{:.1%}", threshold) # Critical

    return results

# --- Helper to display metrics ---
def display_metric(df: pd.DataFrame, col_name: str, title: str, fmt: str = "{:.2f}") -> None:
    """Safely calculates and displays a metric using st.metric."""
    value_display = "N/A"
    if col_name in df.columns and df[col_name].notna().any():
        value = df[col_name].mean()
        if pd.notna(value):
            try:
                value_display = fmt.format(value)
            except (ValueError, TypeError):
                value_display = str(value) # Fallback if format fails

    st.metric(title, value_display)

# --- Main Dashboard Logic ---
st.set_page_config(layout="wide", page_title="LLM Observability Dashboard")
st.title("📊 LLM Observability Dashboard (Manual Implementation)")
st.caption(f"Monitoring interactions logged in `{CSV_FILE}`. Refreshes every {REFRESH_INTERVAL_SECONDS} seconds.")

# Auto-refresh controller
refresh_count = st_autorefresh(interval=REFRESH_INTERVAL_SECONDS * 1000, key="dashboard_refresh")

# Load data
data_df = load_data(CSV_FILE)

if data_df.empty and not os.path.exists(CSV_FILE):
    st.warning(f"Data file '{CSV_FILE}' not found. Please start the chat app (`app.py`) and interact with it to generate data.")
elif data_df.empty:
     st.warning(f"Data file '{CSV_FILE}' is empty or could not be loaded properly. Please interact with the chat app (`app.py`).")
else:
    st.success(f"Loaded {len(data_df)} interactions. Last refresh: {time.strftime('%H:%M:%S')}")

    tab1, tab2, tab3 = st.tabs(["📋 Test Suite", "📈 Metrics & Trends", "📄 Raw Data"])

    # === Tab 1: Test Suite Results ===
    with tab1:
        st.subheader("📊 Model Quality & Safety Test Suite")
        test_results_dict = run_manual_tests(data_df.copy()) # Pass copy

        if not test_results_dict:
            st.info("No tests could be run. Check if the required columns exist in the CSV and contain data.")
        else:
            # Calculate summary stats
            passes = sum(1 for r in test_results_dict.values() if r['Pass'] is True)
            fails_critical = sum(1 for r in test_results_dict.values() if r['Result'] == 'FAIL' and r['Critical'])
            fails_warning = sum(1 for r in test_results_dict.values() if r['Result'] == 'FAIL' and not r['Critical'])
            no_data = sum(1 for r in test_results_dict.values() if r['Result'] == 'NO DATA')
            total_tests = len(test_results_dict)

            # Display summary metrics
            st.markdown("#### Test Summary")
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            with col_m1: st.metric("Total Tests", total_tests)
            with col_m2: st.metric("✅ Passed", passes)
            with col_m3: st.metric("❌ Failed (Critical)", fails_critical, delta_color="inverse")
            with col_m4: st.metric("⚠️ Failed (Warning)", fails_warning, delta_color="inverse")
            with col_m5: st.metric("⚪ No Data", no_data)


            # --- Display Test Results Table ---
            st.markdown("#### Detailed Test Results")
            results_display_df = pd.DataFrame(test_results_dict.values())

            # Function to apply row styling based on test result
            def style_test_row(row: pd.Series) -> List[str]:
                """Applies background color style based on test result."""
                style = ''
                if row['Result'] == 'FAIL':
                    style = 'background-color: #FFCCCB;' if row['Critical'] else 'background-color: #FFFFE0;'
                elif row['Result'] == 'PASS':
                    style = 'background-color: #90EE90;' # Light Green
                elif row['Result'] == 'NO DATA':
                    style = 'background-color: #E5E4E2;' # Light Grey
                return [style] * len(row) # Apply style to all cells in the row

            # Select and configure columns for display
            display_cols = ['Test', 'Threshold', 'Result', 'Value', 'Critical']
            st.dataframe(
                results_display_df[display_cols].style.apply(style_test_row, axis=1),
                use_container_width=True,
                hide_index=True,
                column_config={ # Customize column display
                     "Test": st.column_config.TextColumn("Test Case", help="The specific quality check performed."),
                     "Threshold": st.column_config.TextColumn("Condition", help="The condition for passing the test."),
                     "Result": st.column_config.TextColumn("Status", help="PASS, FAIL, or NO DATA."),
                     "Value": st.column_config.TextColumn("Metric Value", help="The calculated metric value."),
                     "Critical": st.column_config.CheckboxColumn("Critical?", help="Is failure critical?", default=False),
                }
            )

    # === Tab 2: Metrics & Trends ===
    with tab2:
        st.subheader("📈 Key Performance Indicators (KPIs)")
        col1, col2, col3, col4 = st.columns(4)

        with col1: display_metric(data_df, 'latency_ms', "Avg Latency", "{:.0f} ms")
        with col2: display_metric(data_df, 'custom_sentiment_score', "Avg Sentiment Score", "{:.3f}")
        with col3: display_metric(data_df, 'semantic_similarity_prompt_answer', "Avg Sem. Similarity", "{:.3f}")
        with col4: display_metric(data_df, 'custom_toxicity_score', "Avg Toxicity Rate", "{:.1%}") # Rate based on 0/1 score

        st.divider()

        # --- Plotting Section ---
        st.subheader("📉 Metrics Over Time")
        # Ensure timestamp data is available and valid for plotting
        if 'timestamp_dt' in data_df.columns and data_df['timestamp_dt'].notna().any():
             df_plot = data_df.dropna(subset=['timestamp_dt']).sort_values('timestamp_dt')

             if not df_plot.empty:
                 # Select numeric columns suitable for time series plotting
                 # Exclude scores that are just 0/1 flags unless visualizing the flag itself
                 plotable_numeric_cols = [
                     'latency_ms', 'text_length_answer', 'word_count_answer',
                     'semantic_similarity_prompt_answer', 'custom_sentiment_score',
                     # Can also plot rates over time by resampling, but let's stick to individual points for now
                     # 'custom_toxicity_score', 'custom_pii_score', 'custom_decline_score', etc.
                 ]
                 # Filter out columns that don't exist or are all NA in the plot data
                 available_plot_cols = [col for col in plotable_numeric_cols if col in df_plot.columns and df_plot[col].notna().any()]

                 if available_plot_cols:
                     selected_metric = st.selectbox(
                         "Select Metric to Plot vs. Time:",
                         available_plot_cols,
                         index=0,
                         help="Choose a metric to see its trend over interaction time."
                     )

                     if selected_metric:
                         # Create Altair Chart
                         base = alt.Chart(df_plot).encode(
                             x=alt.X('timestamp_dt:T', title='Timestamp (UTC)')
                         )

                         # Tooltip configuration
                         tooltip_list = [
                             alt.Tooltip('timestamp_dt:T', title='Time', format='%Y-%m-%d %H:%M:%S'),
                             alt.Tooltip(selected_metric, title=selected_metric.replace('_', ' ').title(), format='.3f'), # Format metric value
                             alt.Tooltip('prompt:N', title='Prompt'), # Use :N for Nominal (text)
                             alt.Tooltip('answer:N', title='Answer')
                         ]

                         # Line chart with points
                         line = base.mark_line(point=True, strokeWidth=1.5).encode(
                             y=alt.Y(selected_metric, title=selected_metric.replace('_', ' ').title(), scale=alt.Scale(zero=False)), # Don't force Y axis to start at 0
                             tooltip=tooltip_list
                         ).properties(
                             title=f'{selected_metric.replace("_", " ").title()} Over Time'
                         )

                         # Add interactive zooming and panning
                         chart = line.interactive()

                         st.altair_chart(chart, use_container_width=True)
                 else:
                     st.info("No suitable numeric metrics found with data to plot trends.")

                 # --- Label Distribution Plots ---
                 st.divider()
                 st.subheader("📊 Label Distributions")
                 available_label_cols = [col for col in LABEL_COLS if col in data_df.columns and data_df[col].astype(str).str.strip().replace('', np.nan).notna().any()]

                 if available_label_cols:
                     selected_label = st.selectbox(
                         "Select Label Column for Distribution:",
                         available_label_cols,
                         index=0,
                         help="View the frequency of different labels assigned by LLM evaluations."
                        )
                     if selected_label:
                         st.write(f"**Distribution for: {selected_label.replace('_', ' ').title()}**")
                         # Clean labels before counting: fill NaN/empty with 'Unknown', then count
                         counts = data_df[selected_label].astype(str).str.strip().replace('', 'Unknown').fillna('Unknown').value_counts()
                         if not counts.empty:
                             # Use Altair for better bar chart customization
                             count_df = counts.reset_index()
                             count_df.columns = ['Label', 'Count']
                             bar_chart = alt.Chart(count_df).mark_bar().encode(
                                 x=alt.X('Label', sort='-y'), # Sort bars by count descending
                                 y=alt.Y('Count'),
                                 tooltip=['Label', 'Count']
                             ).properties(
                                title=f'{selected_label.replace("_", " ").title()} Counts'
                             )
                             st.altair_chart(bar_chart, use_container_width=True)
                         else:
                             st.caption(f"No data available for the label '{selected_label}'.")

                 else:
                      st.info("No columns ending in '_label' found with data for distribution plots.")

             else:
                st.info("No interaction data with valid timestamps available for plotting trends.")
        else:
            st.info("Timestamp column ('timestamp_dt') missing or contains no valid dates. Cannot plot time series.")


    # === Tab 3: Raw Data ===
    with tab3:
        st.subheader("📄 Raw Interaction Log")
        st.caption(f"Displaying the raw data from `{CSV_FILE}`. Use the filters to explore.")

        # Optional: Add filters for the raw data view
        # Example: Filter by session_id
        # unique_sessions = data_df['session_id'].dropna().unique()
        # selected_session = st.selectbox("Filter by Session ID:", options=["All"] + list(unique_sessions))
        # if selected_session != "All":
        #     filtered_df = data_df[data_df['session_id'] == selected_session]
        # else:
        #     filtered_df = data_df
        # st.dataframe(filtered_df, use_container_width=True)

        # Display the full (potentially filtered) dataframe
        st.dataframe(data_df, use_container_width=True)
