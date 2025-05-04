import streamlit as st
import pandas as pd
import os
import time
from streamlit_autorefresh import st_autorefresh
import numpy as np
import altair as alt # For more flexible plotting

# --- Configuration ---
CSV_FILE = 'interactions_manual.csv' # Match the name used in app_manual.py
REFRESH_INTERVAL_SECONDS = 10 # Auto-refresh interval

# --- Helper Function to load data ---
@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS) # Cache data for short period
def load_data(file_path):
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, low_memory=False, na_values=['NA', '']) # Read 'NA' back as NaN
            # Convert timestamp column
            if 'timestamp' in df.columns:
                df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # Convert potential numeric columns (handle errors -> NaN)
            numeric_cols = [
                'latency_ms', 'text_length_answer', 'word_count_answer', 'sentence_count_answer',
                'semantic_similarity_prompt_answer', 'custom_sentiment_score',
                'custom_toxicity_score', 'custom_pii_score', 'custom_decline_score',
                'custom_bias_score', 'custom_compliance_score'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            # Convert potential boolean columns
            bool_cols = ['contains_link_answer', 'is_valid_json_answer', 'contains_sorry_keywords', 'ends_with_period']
            for col in bool_cols:
                 if col in df.columns:
                      # Map various string representations to boolean, default to False if conversion fails
                      df[col] = df[col].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False, '1.0': True, '0.0': False}).fillna(False).astype(bool)

            return df
        except pd.errors.EmptyDataError:
            st.info(f"'{file_path}' is empty.")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading data from '{file_path}': {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

# --- Manual Test Functions ---
def run_manual_tests(df):
    results = {}
    if df.empty: return results

    # Helper to format test results
    def format_test(name, value, condition_met, is_critical=False, value_format="{:.2f}"):
         return {
              'Test': name,
              'Result': 'PASS' if condition_met else ('FAIL' if condition_met is False else 'NO DATA'),
              'Value': value_format.format(value) if isinstance(value, (int, float)) and condition_met is not None else str(value),
              'Pass': condition_met,
              'Critical': is_critical
         }

    # Test: Average Latency
    if 'latency_ms' in df.columns and df['latency_ms'].notna().any():
        avg_latency = df['latency_ms'].mean()
        results['Avg Latency < 2000ms'] = format_test('Avg Latency < 2000ms', avg_latency, avg_latency < 2000, False, "{:.0f} ms")

    # Test: Semantic Similarity
    if 'semantic_similarity_prompt_answer' in df.columns and df['semantic_similarity_prompt_answer'].notna().any():
        avg_similarity = df['semantic_similarity_prompt_answer'].mean()
        results = format_test('Avg Semantic Similarity >= 0.3', avg_similarity, avg_similarity >= 0.3, False, "{:.3f}")

    # Test: Toxicity Rate (using custom score 0/1)
    if 'custom_toxicity_score' in df.columns and df['custom_toxicity_score'].notna().any():
        toxic_rate = df['custom_toxicity_score'].mean() # Avg of 0/1 gives rate
        results = format_test('Toxicity Rate <= 10%', toxic_rate, toxic_rate <= 0.10, True, "{:.1%}")

    # Test: PII Detection Rate (using custom score 0/1)
    if 'custom_pii_score' in df.columns and df['custom_pii_score'].notna().any():
        pii_rate = df['custom_pii_score'].mean()
        results = format_test('PII Rate == 0%', pii_rate, pii_rate == 0.0, True, "{:.1%}")

    # Test: Decline Rate (using custom score 0/1)
    if 'custom_decline_score' in df.columns and df['custom_decline_score'].notna().any():
        decline_rate = df['custom_decline_score'].mean()
        results = format_test('Decline Rate <= 15%', decline_rate, decline_rate <= 0.15, False, "{:.1%}")

    # Test: Relevance (Irrelevant Rate - check label)
    if 'custom_relevance_label' in df.columns:
        irr_rate = (df['custom_relevance_label'].astype(str).str.lower() == 'irrelevant').mean()
        results = format_test('Irrelevant Rate <= 20%', irr_rate, irr_rate <= 0.20, False, "{:.1%}")

    # Test: Compliance Rate (using custom score 0/1)
    if 'custom_compliance_score' in df.columns and df['custom_compliance_score'].notna().any():
        comp_rate = df['custom_compliance_score'].mean() # Avg of 0/1 where 1=compliant
        results = format_test('Compliance Rate >= 99%', comp_rate, comp_rate >= 0.99, True, "{:.1%}")

    # Test: Contains Link Rate
    if 'contains_link_answer' in df.columns:
         link_rate = df['contains_link_answer'].mean() # Avg of True/False (1/0)
         results = format_test('Link Rate <= 30%', link_rate, link_rate <= 0.30, False, "{:.1%}")

    # Test: Valid JSON Rate
    if 'is_valid_json_answer' in df.columns:
         valid_json_rate = df['is_valid_json_answer'].mean()
         results = format_test('Valid JSON Rate >= 95%', valid_json_rate, valid_json_rate >= 0.95, False, "{:.1%}")


    return results

# --- Main Dashboard Logic ---
st.set_page_config(layout="wide")
st.title("📊 LLM Observability Dashboard (Manual Implementation - No Evidently)")

# Auto-refresh controller
refresh_count = st_autorefresh(interval=REFRESH_INTERVAL_SECONDS * 1000, key="dashboard_refresh")

# Load data
data_df = load_data(CSV_FILE)

if data_df.empty:
    st.warning(f"No interaction data found yet in '{CSV_FILE}'. Please interact with the chat app.")
else:
    st.success(f"Loaded {len(data_df)} interactions. Last refresh: {time.strftime('%H:%M:%S')}")

    tab1, tab2, tab3 = st.tabs()

    with tab1:
        st.subheader("Manual Test Suite Results")
        test_results = run_manual_tests(data_df)
        if not test_results:
            st.info("No tests could be run on the current data.")
        else:
            # Calculate summary stats
            passes = sum(1 for r in test_results.values() if r['Pass'] is True)
            fails_critical = sum(1 for r in test_results.values() if r['Pass'] is False and r['Critical'])
            fails_warning = sum(1 for r in test_results.values() if r['Pass'] is False and not r['Critical'])
            no_data = sum(1 for r in test_results.values() if r['Pass'] is None)
            total_tests = len(test_results)

            st.metric("Total Tests Run", total_tests)
            st.metric("Passed", passes)
            st.metric("Failed (Critical)", fails_critical)
            st.metric("Warnings (Failed Non-Critical)", fails_warning)
            # st.metric("No Data / Skipped", no_data) # Optional

            results_display_df = pd.DataFrame(test_results.values())

            # Color rows based on pass/fail
            def color_result(row):
                color = 'background-color: #D3D3D3' # Default Grey (NO DATA)
                if row == 'FAIL':
                    color = 'background-color: #FFCCCB' # Light Red (FAIL - Critical)
                    if row['Critical'] == False:
                         color = 'background-color: #FFFFE0' # Light Yellow (Warning)
                elif row == 'PASS':
                    color = 'background-color: #90EE90' # Light Green (PASS)
                return [color] * len(row)

            st.dataframe(
                results_display_df].style.apply(color_result, axis=1),
                use_container_width=True,
                hide_index=True
            )

    with tab2:
        st.subheader("Key Metrics Summary & Trends")
        # Display metrics using st.metric or plots
        col1, col2, col3, col4 = st.columns(4)

        # Helper to safely get mean and format metric
        def display_metric(df, col_name, title, fmt="{:.2f}"):
             if col_name in df.columns and df[col_name].notna().any():
                 value = df[col_name].mean()
                 return title, fmt.format(value) if pd.notna(value) else "N/A"
             return title, "N/A"

        m1_title, m1_val = display_metric(data_df, 'latency_ms', "Avg Latency (ms)", "{:.0f}")
        col1.metric(m1_title, m1_val)

        m2_title, m2_val = display_metric(data_df, 'custom_sentiment_score', "Avg Sentiment")
        col2.metric(m2_title, m2_val)

        m3_title, m3_val = display_metric(data_df, 'semantic_similarity_prompt_answer', "Avg Sem. Similarity", "{:.3f}")
        col3.metric(m3_title, m3_val)

        m4_title, m4_val = display_metric(data_df, 'custom_toxicity_score', "Toxicity Rate", "{:.1%}")
        col4.metric(m4_title, m4_val)

        st.divider()

        # Plot some time series data if timestamp exists
        if 'timestamp_dt' in data_df.columns and data_df['timestamp_dt'].notna().any():
             df_plot = data_df.dropna(subset=['timestamp_dt']).sort_values('timestamp_dt')

             st.subheader("Metrics Over Time")
             # Select numeric columns suitable for plotting trends
             numeric_plot_cols = [
                 'latency_ms', 'text_length_answer', 'word_count_answer',
                 'semantic_similarity_prompt_answer', 'custom_sentiment_score',
                 'custom_toxicity_score', 'custom_pii_score', 'custom_decline_score',
                 'custom_bias_score', 'custom_compliance_score'
             ]
             plot_cols_exist = [col for col in numeric_plot_cols if col in df_plot.columns and df_plot[col].notna().any()]

             if plot_cols_exist:
                 # Use Altair for better control over tooltips and axes
                 base = alt.Chart(df_plot).encode(x='timestamp_dt:T')
                 charts =
                 for col in plot_cols_exist:
                      chart = base.mark_line(point=True).encode(
                           y=alt.Y(col, axis=alt.Axis(title=col.replace('_', ' ').title())),
                           tooltip=
                      ).properties(
                           title=f'{col.replace("_", " ").title()} Trend'
                      )
                      charts.append(chart)

                 # Combine charts vertically or allow selection
                 # st.altair_chart(alt.vconcat(*charts), use_container_width=True) # Might be too tall
                 selected_metric = st.selectbox("Select Metric to Plot Trend:", plot_cols_exist)
                 if selected_metric:
                      selected_chart = base.mark_line(point=True).encode(
                           y=alt.Y(selected_metric, axis=alt.Axis(title=selected_metric.replace('_', ' ').title())),
                           tooltip=
                      ).properties(
                           title=f'{selected_metric.replace("_", " ").title()} Trend'
                      ).interactive() # Add zooming/panning
                      st.altair_chart(selected_chart, use_container_width=True)

             else:
                 st.info("Not enough numeric data or columns available to plot time series metrics.")

             st.subheader("Label Distributions")
             label_cols = [col for col in data_df.columns if col.endswith('_label')]
             if label_cols:
                 selected_label = st.selectbox("Select Label to View Distribution:", label_cols)
                 if selected_label and selected_label in data_df.columns:
                      st.write(f"**{selected_label.replace('_', ' ').title()} Distribution:**")
                      # Calculate value counts and handle potential NaN/None labels
                      counts = data_df[selected_label].fillna('Unknown').value_counts()
                      if not counts.empty:
                           st.bar_chart(counts)
                      else:
                           st.caption("No data available for this label.")
             else:
                  st.info("No label columns found for distribution plots.")

        else:
            st.info("Timestamp column missing or invalid, cannot plot time series.")


    with tab3:
        st.subheader("Raw Interaction Data")
        # Display all columns, maybe allow filtering later
        st.dataframe(data_df, use_container_width=True)
