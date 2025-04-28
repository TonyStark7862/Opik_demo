import streamlit as st
import opik
import os

# === Step 1: Configure Opik SDK to point to your local Opik server ===
# This tells the SDK to send logs to your local Opik server running at localhost:5173
os.environ["OPIK_BASE_URL"] = "http://localhost:5173/api"
os.environ["OPIK_URL_OVERRIDE"] = "http://localhost:5173/api"

# Force configure Opik to use local server URL
opik.configure(use_local=True, force=True)

# === Step 2: Define your custom demo API function ===
def abc_response(prompt: str) -> str:
    # Replace this with your actual LLM or logic
    return f"Echo from abc_response: {prompt}"

# === Step 3: Define a function to log prompt and response to Opik ===
def log_to_opik(prompt: str, response: str):
    with opik.start_trace(name="StreamlitDemoTrace") as trace:
        trace.log_input(prompt)
        trace.log_output(response)
        trace.log_metadata({"app": "streamlit-demo", "version": "1.0"})

# === Step 4: Build Streamlit UI ===
st.title("Streamlit + Local Opik Demo (Dashboard at localhost:5173)")

user_prompt = st.text_input("Enter your prompt:")

if st.button("Send"):
    if not user_prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        # Get response from your custom API
        response = abc_response(user_prompt)
        st.markdown("### Response:")
        st.write(response)

        # Log interaction to local Opik server
        try:
            log_to_opik(user_prompt, response)
            st.success("Logged interaction to local Opik dashboard!")
        except Exception as e:
            st.error(f"Failed to log to Opik: {e}")

st.markdown("---")
st.info("Make sure your local Opik server is running at [http://localhost:5173](http://localhost:5173) to see the dashboard.")

