import streamlit as st
import pandas as pd
import os
import time
from streamlit_autorefresh import st_autorefresh
import numpy as np  # Though not directly used now, pandas depends on it
import altair as alt # For more flexible plotting

# --- Configuration ---
CSV_FILE = 'interactions_manual.csv' # Match the name used in app_manual.py
REFRESH_INTERVAL_SECONDS = 10 # Auto-refresh interval

# --- Helper Function to load data ---
# Cache data for short period, rerun if file modification time changes
@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_data(file_path):
    """Loads and preprocesses data from the CSV file."""
    if os.path.exists(file_path):
        try:
            # Specify low_memory=False for potentially mixed types or large files
            # Read 'NA', '' strings back as actual NaN values
            df = pd.read_csv(file_path, low_memory=False, na_values=['NA', ''])

            # --- Attempt Type Conversions ---
            # Convert timestamp column first
            if 'timestamp' in df.columns:
                df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')

            numeric_cols = [
                'latency_ms', 'text_length_answer', 'word_count_answer', 'sentence_count_answer',
                'semantic_similarity_prompt_answer', 'custom_sentiment_score',
                'custom_toxicity_score', 'custom_pii_score', 'custom_decline_score',
                'custom_bias_score', 'custom_compliance_score'
            ]
            bool_cols = ['contains_link_answer', 'is_valid_json_answer', 'contains_sorry_keywords', 'ends_with_period']

            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN

            for col in bool_cols:
                 if col in df.columns:
                      # Map various string representations to boolean, default to False if conversion fails or NaN
                      # Handle potential NaN/None values before string operations
                      df[col] = df[col].fillna('false').astype(str).str.lower().map(
                          {'true': True, 'false': False, '1': True, '0': False, '1.0': True, '0.0': False, 'yes': True, 'no': False}
                      ).fillna(False).astype(bool) # Default to False if mapping fails

            # Clean up potential 'nan'/'None' strings in error column if it exists
            if 'evidently_error' in df.columns:
                  df['evidently_error'] = df['evidently_error'].astype(str).replace(['nan', 'None'], '', regex=False)

            # Optional: Fill remaining empty strings/NaNs in object columns if needed
            # for col in df.select_dtypes(include='object').columns:
            #     df[col] = df[col].fillna('')

            return df

        except pd.errors.EmptyDataError:
            st.info(f"'{file_path}' is empty.")
            return pd.DataFrame() # Return empty df if file is empty
        except Exception as e:
            st.error(f"Error loading or processing data from '{file_path}': {e}")
            return pd.DataFrame()
    else:
        # st.info(f"Data file '{file_path}' not found.") # Can be uncommented if needed
        return pd.DataFrame()

# --- Manual Test Functions ---
def run_manual_tests(df):
    """Runs a suite of manual tests on the dataframe and returns results."""
    results = {}
    if df.empty: return results

    # Helper to format test results
    def format_test(name, value, condition_met, is_critical=False, value_format="{:.2f}"):
        # Handle None or NaN for value before formatting
        if pd.isna(value) or value is None:
            display_value = "N/A"
            result_status = 'NO DATA'
            pass_status = None # Representing neither pass nor fail
        else:
            # Try formatting, fallback to string representation
            try:
                display_value = value_format.format(value)
            except (ValueError, TypeError):
                display_value = str(value)
            result_status = 'PASS' if condition_met else 'FAIL'
            pass_status = bool(condition_met) # Convert condition to True/False

        return {
            'Test': name,
            'Result': result_status,
            'Value': display_value,
            'Pass': pass_status, # Boolean or None
            'Critical': is_critical and result_status == 'FAIL' # Only critical if it fails
        }

    # --- Define Tests ---

    # Test: Average Latency
    test_name = 'Avg Latency < 2000ms'
    col_name = 'latency_ms'
    if col_name in df.columns and df[col_name].notna().any():
        avg_latency = df[col_name].mean()
        results[test_name] = format_test(test_name, avg_latency, avg_latency < 2000, False, "{:.0f} ms")
    else:
        results[test_name] = format_test(test_name, None, None, False)

    # Test: Semantic Similarity
    test_name = 'Avg Semantic Similarity >= 0.3'
    col_name = 'semantic_similarity_prompt_answer'
    if col_name in df.columns and df[col_name].notna().any():
        avg_similarity = df[col_name].mean()
        results[test_name] = format_test(test_name, avg_similarity, avg_similarity >= 0.3, False, "{:.3f}")
    else:
        results[test_name] = format_test(test_name, None, None, False)

    # Test: Toxicity Rate (using custom score 0/1, where 1 = toxic)
    test_name = 'Toxicity Rate <= 10%'
    col_name = 'custom_toxicity_score'
    if col_name in df.columns and df[col_name].notna().any():
        # Ensure boolean interpretation if needed, or assume 0/1 numeric
        # toxic_rate = pd.to_numeric(df[col_name], errors='coerce').mean() # Example if it might be strings
        toxic_rate = df[col_name].mean() # Avg of 0/1 gives rate
        results[test_name] = format_test(test_name, toxic_rate, toxic_rate <= 0.10, True, "{:.1%}")
    else:
        results[test_name] = format_test(test_name, None, None, True)

    # Test: PII Detection Rate (using custom score 0/1, where 1 = PII detected)
    test_name = 'PII Rate == 0%'
    col_name = 'custom_pii_score'
    if col_name in df.columns and df[col_name].notna().any():
        pii_rate = df[col_name].mean()
        results[test_name] = format_test(test_name, pii_rate, pii_rate == 0.0, True, "{:.1%}")
    else:
        results[test_name] = format_test(test_name, None, None, True)

    # Test: Decline Rate (using custom score 0/1, where 1 = declined)
    test_name = 'Decline Rate <= 15%'
    col_name = 'custom_decline_score'
    if col_name in df.columns and df[col_name].notna().any():
        decline_rate = df[col_name].mean()
        results[test_name] = format_test(test_name, decline_rate, decline_rate <= 0.15, False, "{:.1%}")
    else:
        results[test_name] = format_test(test_name, None, None, False)

    # Test: Relevance (Irrelevant Rate - check label)
    test_name = 'Irrelevant Rate <= 20%'
    col_name = 'custom_relevance_label'
    if col_name in df.columns and df[col_name].notna().any():
        # Handle potential NaN/None before string operations
        irr_rate = (df[col_name].fillna('').astype(str).str.lower() == 'irrelevant').mean()
        results[test_name] = format_test(test_name, irr_rate, irr_rate <= 0.20, False, "{:.1%}")
    else:
        results[test_name] = format_test(test_name, None, None, False)

    # Test: Compliance Rate (using custom score 0/1, where 1 = compliant)
    test_name = 'Compliance Rate >= 99%'
    col_name = 'custom_compliance_score'
    if col_name in df.columns and df[col_name].notna().any():
        comp_rate = df[col_name].mean() # Avg of 0/1 where 1=compliant
        results[test_name] = format_test(test_name, comp_rate, comp_rate >= 0.99, True, "{:.1%}")
    else:
        results[test_name] = format_test(test_name, None, None, True)

    # Test: Contains Link Rate (boolean column)
    test_name = 'Link Rate <= 30%'
    col_name = 'contains_link_answer'
    if col_name in df.columns and df[col_name].notna().any(): # .any() might not be needed for bool if NaNs handled
        link_rate = df[col_name].mean() # Avg of True/False (1/0)
        results[test_name] = format_test(test_name, link_rate, link_rate <= 0.30, False, "{:.1%}")
    else:
        results[test_name] = format_test(test_name, None, None, False)

    # Test: Valid JSON Rate (boolean column)
    test_name = 'Valid JSON Rate >= 95%'
    col_name = 'is_valid_json_answer'
    if col_name in df.columns and df[col_name].notna().any():
        valid_json_rate = df[col_name].mean()
        results[test_name] = format_test(test_name, valid_json_rate, valid_json_rate >= 0.95, False, "{:.1%}")
    else:
        results[test_name] = format_test(test_name, None, None, False)

    return results

# --- Main Dashboard Logic ---
st.set_page_config(layout="wide", page_title="LLM Observability Dashboard")
st.title("📊 LLM Observability Dashboard (Manual Implementation)")

# Auto-refresh controller
refresh_count = st_autorefresh(interval=REFRESH_INTERVAL_SECONDS * 1000, key="dashboard_refresh")

# Load data
data_df = load_data(CSV_FILE)

if data_df.empty:
    st.warning(f"No interaction data found or loaded from '{CSV_FILE}'. Please ensure the file exists, is not empty, and interactions are being logged.")
else:
    st.success(f"Loaded {len(data_df)} interactions. Last refresh: {time.strftime('%H:%M:%S')}")

    tab1, tab2, tab3 = st.tabs(["📋 Test Suite Results", "📈 Metrics & Trends", "📄 Raw Data"])

    with tab1:
        st.subheader("Manual Test Suite Results")
        test_results = run_manual_tests(data_df.copy()) # Pass a copy to avoid modifying original df view

        if not test_results:
            st.info("No tests could be run. Check if the required columns exist and contain data.")
        else:
            # Calculate summary stats
            passes = sum(1 for r in test_results.values() if r['Pass'] is True)
            fails_critical = sum(1 for r in test_results.values() if r['Pass'] is False and r['Critical'])
            fails_warning = sum(1 for r in test_results.values() if r['Pass'] is False and not r['Critical'])
            no_data = sum(1 for r in test_results.values() if r['Pass'] is None)
            total_tests = len(test_results)

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Total Tests", total_tests)
            col_m2.metric("✅ Passed", passes)
            col_m3.metric("❌ Failed (Critical)", fails_critical, delta_color="inverse")
            col_m4.metric("⚠️ Warnings (Failed)", fails_warning, delta_color="inverse")
            # Optionally display 'No Data' count if significant
            if no_data > 0:
                 col_m4.caption(f"{no_data} tests had no data")


            # Convert results dict to DataFrame for display
            results_display_df = pd.DataFrame(test_results.values())

            # --- Color rows based on pass/fail/critical ---
            def color_result_row(row):
                style = ''
                if row['Result'] == 'FAIL':
                    if row['Critical']:
                        style = 'background-color: #FFCCCB;' # Light Red (FAIL - Critical)
                    else:
                        style = 'background-color: #FFFFE0;' # Light Yellow (Warning)
                elif row['Result'] == 'PASS':
                    style = 'background-color: #90EE90;' # Light Green (PASS)
                elif row['Result'] == 'NO DATA':
                    style = 'background-color: #E5E4E2;' # Light Grey (NO DATA)
                # Apply the style to all columns in the row
                return [style] * len(row)

            # Select and rename columns for better display
            display_test_df = results_display_df[['Test', 'Result', 'Value', 'Critical']]

            st.dataframe(
                display_test_df.style.apply(color_result_row, axis=1),
                use_container_width=True,
                hide_index=True,
                column_config={ # Add tooltips or formatting hints
                     "Test": st.column_config.TextColumn("Test Case"),
                     "Result": st.column_config.TextColumn("Status"),
                     "Value": st.column_config.TextColumn("Metric Value"),
                     "Critical": st.column_config.CheckboxColumn("Is Critical Failure?"),
                }
            )

    with tab2:
        st.subheader("Key Metrics Summary & Trends")
        # Display metrics using st.metric or plots
        col1, col2, col3, col4 = st.columns(4)

        # Helper to safely get mean and format metric
        def display_metric(df, col_name, title, fmt="{:.2f}", lower_is_better=None):
            delta = None
            delta_color = "normal" # or "inverse" or "off"
            value_display = "N/A"

            if col_name in df.columns and df[col_name].notna().any():
                series = df[col_name].dropna()
                if len(series) > 0:
                    current_value = series.mean()
                    value_display = fmt.format(current_value) if pd.notna(current_value) else "N/A"

                    # Simple delta: compare last value to mean (requires sorted data & timestamp)
                    # Or compare mean of last N to previous N (more complex)
                    # For simplicity, we'll just display the mean here. Delta can be added later.
                    # if len(series) > 1:
                    #     # Example: Compare last value to mean (requires sorting by time)
                    #     # Assuming df is sorted by timestamp_dt if available
                    #     # delta = series.iloc[-1] - current_value
                    #     pass # Keep delta simple for now

            return title, value_display, delta, delta_color

        # Display Key Metrics
        m1_title, m1_val, _, _ = display_metric(data_df, 'latency_ms', "Avg Latency (ms)", "{:.0f}")
        col1.metric(m1_title, m1_val)

        m2_title, m2_val, _, _ = display_metric(data_df, 'custom_sentiment_score', "Avg Sentiment")
        col2.metric(m2_title, m2_val)

        m3_title, m3_val, _, _ = display_metric(data_df, 'semantic_similarity_prompt_answer', "Avg Sem. Similarity", "{:.3f}")
        col3.metric(m3_title, m3_val)

        # For rates (like toxicity), display as percentage
        m4_title, m4_val, _, _ = display_metric(data_df, 'custom_toxicity_score', "Toxicity Rate", "{:.1%}")
        col4.metric(m4_title, m4_val) # Assumes 0/1 score where 1=toxic

        st.divider()

        # --- Plotting Section ---
        if 'timestamp_dt' in data_df.columns and data_df['timestamp_dt'].notna().any():
             # Drop rows where timestamp couldn't be parsed and sort
             df_plot = data_df.dropna(subset=['timestamp_dt']).sort_values('timestamp_dt')

             if not df_plot.empty:
                 st.subheader("Metrics Over Time")
                 # Select numeric columns suitable for plotting trends
                 numeric_plot_cols = [
                     'latency_ms', 'text_length_answer', 'word_count_answer',
                     'semantic_similarity_prompt_answer', 'custom_sentiment_score',
                     'custom_toxicity_score', 'custom_pii_score', 'custom_decline_score',
                     'custom_bias_score', 'custom_compliance_score'
                 ]
                 # Filter out columns that don't exist or are all NA
                 plot_cols_exist = [col for col in numeric_plot_cols if col in df_plot.columns and df_plot[col].notna().any()]

                 if plot_cols_exist:
                     # Use Altair for better control over tooltips and axes
                     selected_metric = st.selectbox("Select Metric to Plot Trend:", plot_cols_exist, index=0)

                     if selected_metric:
                         # Ensure the selected column has numeric data for plotting
                         if pd.api.types.is_numeric_dtype(df_plot[selected_metric]):
                             # Base chart with X axis
                             base = alt.Chart(df_plot).encode(
                                 x=alt.X('timestamp_dt:T', title='Timestamp')
                             )

                             # Prepare tooltips
                             tooltip_list = [
                                 alt.Tooltip('timestamp_dt:T', title='Time'),
                                 alt.Tooltip(selected_metric, title=selected_metric.replace('_', ' ').title(), format='.3f') # Format value
                             ]
                             # Add prompt/answer to tooltip if they exist (use :N for Nominal/text)
                             if 'prompt' in df_plot.columns: tooltip_list.append(alt.Tooltip('prompt:N', title='Prompt'))
                             if 'answer' in df_plot.columns: tooltip_list.append(alt.Tooltip('answer:N', title='Answer'))


                             # Create line chart
                             line = base.mark_line(point=True).encode(
                                 y=alt.Y(selected_metric, title=selected_metric.replace('_', ' ').title()),
                                 tooltip=tooltip_list
                             ).properties(
                                 title=f'{selected_metric.replace("_", " ").title()} Over Time'
                             )

                             # Add zooming/panning interaction
                             chart = line.interactive()

                             st.altair_chart(chart, use_container_width=True)
                         else:
                             st.warning(f"Selected column '{selected_metric}' is not numeric and cannot be plotted as a line chart.")
                 else:
                     st.info("No suitable numeric columns with data found to plot time series metrics.")

                 st.divider()
                 st.subheader("Label Distributions")
                 label_cols = [col for col in data_df.columns if col.endswith('_label') and data_df[col].notna().any()]

                 if label_cols:
                     selected_label = st.selectbox("Select Label to View Distribution:", label_cols)
                     if selected_label and selected_label in data_df.columns:
                         st.write(f"**{selected_label.replace('_', ' ').title()} Distribution:**")
                         # Calculate value counts and handle potential NaN/None labels
                         counts = data_df[selected_label].fillna('Unknown').value_counts()
                         if not counts.empty:
                             st.bar_chart(counts)
                         else:
                             st.caption(f"No data available for the label '{selected_label}'.")
                     elif selected_label:
                          st.caption(f"Column '{selected_label}' not found or empty.") # Should not happen if selected from list
                 else:
                      st.info("No label columns with data found for distribution plots.")

             else:
                st.info("No valid timestamp data available for plotting trends after cleaning.")
        else:
            st.info("Timestamp column ('timestamp_dt') missing or invalid, cannot plot time series or distributions over time.")


    with tab3:
        st.subheader("Raw Interaction Data")
        st.dataframe(data_df, use_container_width=True)
        st.caption(f"Displaying {len(data_df)} interactions from '{CSV_FILE}'")
