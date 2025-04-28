import streamlit as st
import datetime
import pandas as pd # Optional: for better log display
import time # To simulate duration/latency
import uuid # To generate unique IDs like traces

# === Step 1: Define your custom demo API function ===
# Replace this with your actual LLM or logic
def abc_response(prompt: str) -> str:
    """Generates a response based on the input prompt."""
    # Example: Simple echo for demonstration
    # Simulate some work time
    time.sleep(0.5)
    return f"Echo from abc_response: {prompt}"

# === Step 2: Initialize session state for logging ===
# This creates a list to store log entries if it doesn't exist yet
if 'interaction_log_opik_style' not in st.session_state:
    st.session_state.interaction_log_opik_style = []

# === Step 3: Define a function to log interaction locally (Opik Style) ===
def log_locally_opik_style(prompt: str, response: str, start_time: float, end_time: float):
    """Logs interaction data locally, mimicking Opik structure."""
    trace_id = str(uuid.uuid4()) # Generate a unique ID for this interaction trace
    span_id = str(uuid.uuid4()) # Generate a unique ID for this specific step (span)
    timestamp = datetime.datetime.now().isoformat()
    duration_ms = (end_time - start_time) * 1000

    log_entry = {
        "trace_id": trace_id,
        "span_id": span_id,
        "name": "LLM Interaction", # Name for this step/span
        "timestamp": timestamp,
        "duration_ms": round(duration_ms, 2),
        "status": "OK", # Could be 'ERROR' if abc_response failed
        "inputs": {"prompt": prompt},
        "outputs": {"response": response},
        "metadata": {
            "app": "streamlit-local-opik-log-demo",
            "version": "1.1",
            "prompt_length": len(prompt),
            "response_length": len(response),
            # Add any other relevant metadata
        },
        # --- Potential Future Opik-like fields ---
        # "attributes": {"model": "abc_model", "temperature": 0.7},
        # "events": [{"name": "Token calculation", "timestamp": ...}],
        # "feedback": {"score": None, "comment": None},
    }
    st.session_state.interaction_log_opik_style.append(log_entry)
    st.success(f"Interaction logged locally! (Trace ID: {trace_id})")

# === Step 4: Build Streamlit UI ===
st.set_page_config(layout="wide")
st.title("Streamlit LLM App with Opik-Style Local Logging")
st.markdown("Enter a prompt, get a response using `abc_response`, and view the locally logged interaction details below.")

st.markdown("---")

# Input Area
col1, col2 = st.columns([3, 1])

with col1:
    user_prompt = st.text_area("Enter your prompt:", height=100, key="prompt_input_opik")

with col2:
    st.write("")
    st.write("")
    send_button = st.button("Send Prompt", use_container_width=True, key="send_button_opik")


if send_button:
    if not user_prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        # --- Interaction ---
        st.markdown("### Current Interaction")
        response = None
        start_time = time.time()
        try:
            with st.spinner("Generating response..."):
                # Get response from your custom API
                response = abc_response(user_prompt)
            end_time = time.time()

            # Display current response
            st.text_area("Response:", value=response, height=150, disabled=True, key="response_output_opik")

            # Log interaction locally using our Opik-style function
            log_locally_opik_style(user_prompt, response, start_time, end_time)

        except Exception as e:
            end_time = time.time()
            st.error(f"Error during LLM call or logging: {e}")
            # Optionally log the error as well
            # log_error_locally(...) # You could create a similar function for errors

        # Clear the input box after processing (optional)
        # st.session_state.prompt_input_opik = ""

st.markdown("---")

# === Step 5: Display Local Interaction Log ===
st.header("Interaction Log (Opik-Style, Current Session)")

if not st.session_state.interaction_log_opik_style:
    st.info("No interactions logged yet in this session.")
else:
    # Option 1: Display as a DataFrame (requires pandas)
    try:
        # Flatten the nested dictionaries for better display in DataFrame
        flat_log = []
        for entry in st.session_state.interaction_log_opik_style:
            flat_entry = {
                "Timestamp": entry["timestamp"],
                "Duration (ms)": entry["duration_ms"],
                "Status": entry["status"],
                "Trace ID": entry["trace_id"],
                "Span ID": entry["span_id"],
                "Prompt": entry["inputs"]["prompt"],
                "Response": entry["outputs"]["response"],
                "Prompt Length": entry["metadata"].get("prompt_length", ""),
                "Response Length": entry["metadata"].get("response_length", ""),
                "App": entry["metadata"].get("app", ""),
            }
            flat_log.append(flat_entry)

        log_df = pd.DataFrame(flat_log)
        # Display newest first
        st.dataframe(log_df.iloc[::-1], use_container_width=True)

        # Allow downloading the log as CSV
        @st.cache_data # Cache the conversion
        def convert_df_to_csv(df):
           # IMPORTANT: Cache the conversion to prevent computation on every rerun
           return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(log_df.iloc[::-1])
        st.download_button(
           label="Download Log as CSV",
           data=csv,
           file_name='streamlit_opik_style_log.csv',
           mime='text/csv',
        )

    except ImportError:
        st.warning("Pandas not installed. Displaying logs as raw list.")
        # Display newest first
        st.json(st.session_state.interaction_log_opik_style[::-1]) # Fallback

    # You could also add expanders here for a more detailed view if needed
