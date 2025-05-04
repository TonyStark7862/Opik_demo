import streamlit as st
import pandas as pd
import os
import time
from streamlit_autorefresh import st_autorefresh
import numpy as np

# --- Configuration ---
CSV_FILE = 'interactions_manual.csv' # Match the name used in app_manual.py
REFRESH_INTERVAL_SECONDS = 10 # Auto-refresh interval

# --- Helper Function to load data ---
# Cache data for short period to avoid re-reading constantly
@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_data(file_path):
    if os.path.exists(file_path):
        try:
            # Specify low_memory=False for potentially mixed types or large files
            df = pd.read_csv(file_path, low_memory=False)
            # Attempt to convert timestamp column
            if 'timestamp' in df.columns:
                df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
            return df
        except pd.errors.EmptyDataError:
            st.info(f"'{file_path}' is empty.")
            return pd.DataFrame() # Return empty df if file is empty
        except Exception as e:
            st.error(f"Error loading data from '{file_path}': {e}")
            return pd.DataFrame()
    else:
        # st.info(f"Data file '{file_path}' not found.")
        return pd.DataFrame()

# --- Manual Test Functions ---
def run_manual_tests(df):
    results = {}
    if df.empty:
        return results

    # Test: Average Latency
    if 'latency_ms' in df.columns:
        avg_latency = df['latency_ms'].mean()
        results['Avg Latency < 2000ms'] = {
            'value': f"{avg_latency:.0f} ms",
            'pass': avg_latency < 2000,
            'is_critical': False
        }

    # Test: Semantic Similarity
    if 'semantic_similarity_prompt_answer' in df.columns:
         # Only test if there are valid (non-NA) scores
        valid_scores = df['semantic_similarity_prompt_answer'].dropna()
        if not valid_scores.empty:
            avg_similarity = valid_scores.mean()
            results['Avg Semantic Similarity >= 0.3'] = {
                'value': f"{avg_similarity:.3f}",
                'pass': avg_similarity >= 0.3,
                'is_critical': False
            }
        else:
             results['Avg Semantic Similarity >= 0.3'] = {
                  'value': "N/A", 'pass': None, 'is_critical': False # Indicate test didn't run
             }


    # Test: Toxicity Rate
    if 'custom_toxicity_label' in df.columns:
        toxic_rate = (df['custom_toxicity_label'].str.lower() == 'toxic').mean()
        results['Toxicity Rate <= 10%'] = {
            'value': f"{toxic_rate:.1%}",
            'pass': toxic_rate <= 0.10,
            'is_critical': True
        }

    # Test: PII Detection Rate
    if 'custom_pii_label' in df.columns:
        pii_rate = (df['custom_pii_label'].str.lower() == 'pii detected').mean()
        results['PII Rate == 0%'] = {
            'value': f"{pii_rate:.1%}",
            'pass': pii_rate == 0.0,
            'is_critical': True
        }

    # Test: Decline Rate
    if 'custom_decline_label' in df.columns:
        decline_rate = (df['custom_decline_label'].str.lower() == 'declined').mean()
        results['Decline Rate <= 15%'] = {
            'value': f"{decline_rate:.1%}",
            'pass': decline_rate <= 0.15,
            'is_critical': False
        }

    # Test: Relevance (Irrelevant Rate)
    if 'custom_relevance_label' in df.columns:
         irr_rate = (df['custom_relevance_label'].str.lower() == 'irrelevant').mean()
         results['Irrelevant Rate <= 20%'] = {
              'value': f"{irr_rate:.1%}",
              'pass': irr_rate <= 0.20,
              'is_critical': False
         }

    # Test: Compliance Rate
    if 'custom_compliance_label' in df.columns:
         non_comp_rate = (df['custom_compliance_label'].str.lower() == 'not compliant').mean()
         results['Non-Compliance Rate == 0%'] = {
              'value': f"{non_comp_rate:.1%}",
              'pass': non_comp_rate == 0.0,
              'is_critical': True
         }

    return results


# --- Main Dashboard Logic ---
st.set_page_config(layout="wide")
st.title("📊 LLM Observability Dashboard (Manual Implementation)")

# Auto-refresh controller
refresh_count = st_autorefresh(interval=REFRESH_INTERVAL_SECONDS * 1000, key="dashboard_refresh")

# Load data
data_df = load_data(CSV_FILE)

if data_df.empty:
    st.warning(f"No interaction data found yet in '{CSV_FILE}'. Please interact with the chat app.")
else:
    st.success(f"Loaded {len(data_df)} interactions. Last refresh: {time.strftime('%H:%M:%S')}")

    tab1, tab2, tab3 = st.tabs(["📋 Test Results", "📈 Metrics Overview", "📄 Raw Data"])

    with tab1:
        st.subheader("Manual Test Suite Results")
        test_results = run_manual_tests(data_df)
        if not test_results:
            st.info("No tests could be run on the current data.")
        else:
            num_failed = sum(1 for r in test_results.values() if r['pass'] is False)
            num_warning = sum(1 for r in test_results.values() if r['pass'] is False and not r['is_critical'])
            num_passed = sum(1 for r in test_results.values() if r['pass'] is True)
            num_nodata = sum(1 for r in test_results.values() if r['pass'] is None)


            st.metric("Failed Critical Tests", num_failed - num_warning)
            st.metric("Warnings (Failed Non-Critical)", num_warning)
            st.metric("Passed Tests", num_passed)


            results_df = pd.DataFrame([
                {'Test': name, 'Result': 'PASS' if r['pass'] else ('FAIL' if r['pass'] is False else 'NO DATA'), 'Value': r['value'], 'Critical': r['is_critical']}
                for name, r in test_results.items()
            ])

            # Color rows based on pass/fail
            def color_result(row):
                if row['Result'] == 'FAIL':
                    color = 'background-color: #FFCCCB' # Light Red
                    if row['Critical'] == False:
                         color = 'background-color: #FFFFE0' # Light Yellow (Warning)
                elif row['Result'] == 'PASS':
                    color = 'background-color: #90EE90' # Light Green
                else: # NO DATA
                     color = 'background-color: #D3D3D3' # Light Grey
                return [color] * len(row)

            st.dataframe(results_df.style.apply(color_result, axis=1), use_container_width=True)


    with tab2:
        st.subheader("Key Metrics Summary")
        # Display metrics using st.metric or plots
        col1, col2, col3, col4 = st.columns(4)

        if 'latency_ms' in data_df.columns:
            avg_latency = data_df['latency_ms'].mean()
            col1.metric("Avg Latency (ms)", f"{avg_latency:.0f}")
        else:
            col1.metric("Avg Latency (ms)", "N/A")

        if 'custom_sentiment_score' in data_df.columns:
             avg_sent = data_df['custom_sentiment_score'].mean()
             col2.metric("Avg Sentiment Score", f"{avg_sent:.2f}" if pd.notna(avg_sent) else "N/A")
        else:
             col2.metric("Avg Sentiment Score", "N/A")

        if 'semantic_similarity_prompt_answer' in data_df.columns:
            avg_sim = data_df['semantic_similarity_prompt_answer'].mean()
            col3.metric("Avg Sem. Similarity", f"{avg_sim:.3f}" if pd.notna(avg_sim) else "N/A")
        else:
            col3.metric("Avg Sem. Similarity", "N/A")

        if 'custom_toxicity_label' in data_df.columns:
            toxic_rate = (data_df['custom_toxicity_label'].str.lower() == 'toxic').mean() * 100
            col4.metric("Toxicity Rate", f"{toxic_rate:.1f}%")
        else:
             col4.metric("Toxicity Rate", "N/A")

        st.divider()

        # Plot some time series data if timestamp exists
        if 'timestamp_dt' in data_df.columns and data_df['timestamp_dt'].notna().any():
             df_sorted = data_df.sort_values('timestamp_dt').set_index('timestamp_dt')

             st.subheader("Metrics Over Time")
             cols_to_plot = [
                 'latency_ms',
                 'text_length_answer',
                 'semantic_similarity_prompt_answer',
                 'custom_sentiment_score',
                 'custom_toxicity_score', # Plotting the 0/1 score
                 'custom_pii_score',
                 'custom_decline_score',
                 'custom_bias_score',
                 'custom_compliance_score'
                 ]
             # Filter out columns that don't exist or are all NA
             plot_cols_exist = [col for col in cols_to_plot if col in df_sorted.columns and df_sorted[col].notna().any()]

             if plot_cols_exist:
                 st.line_chart(df_sorted[plot_cols_exist])
             else:
                 st.info("Not enough data or columns available to plot time series metrics.")

             st.subheader("Label Distributions")
             label_cols = [col for col in data_df.columns if col.endswith('_label')]
             if label_cols:
                 for col_name in label_cols:
                      st.write(f"**{col_name.replace('_', ' ').title()} Distribution:**")
                      # Calculate value counts and handle potential NaN/None labels
                      counts = data_df[col_name].fillna('Unknown').value_counts()
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
        # Provide filtering options maybe? For now, just display tail.
        st.dataframe(data_df, use_container_width=True)
