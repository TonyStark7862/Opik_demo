#!/bin/bash

# Script to run the Evidently LLM Observability Demo Locally

echo "Starting the LLM Observability Demo..."

# --- Configuration ---
WORKSPACE_DIR="evidently_workspace"
APP_FILE="app.py"
MONITOR_FILE="monitor.py"
REQUIREMENTS_FILE="requirements.txt"
EVIDENTLY_PORT=8000 # Default port for Evidently UI
STREAMLIT_PORT=8501 # Default port for Streamlit

# --- Check Prerequisites ---
echo "[1/5] Checking prerequisites..."
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 is not installed or not in PATH. Please install Python 3."
    exit 1
fi

if ! command -v pip &> /dev/null
then
    # Try pip3
    if ! command -v pip3 &> /dev/null
    then
      echo "Error: pip (or pip3) is not installed or not in PATH. Please install pip."
      exit 1
    else
      PIP_COMMAND="pip3"
    fi
else
    PIP_COMMAND="pip"
fi
echo "Prerequisites check passed (Python 3, pip)."

# --- Install Dependencies ---
echo "[2/5] Installing dependencies from $REQUIREMENTS_FILE..."
if [ -f "$REQUIREMENTS_FILE" ]; then
    $PIP_COMMAND install -r $REQUIREMENTS_FILE
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies. Please check $REQUIREMENTS_FILE and your internet connection."
        exit 1
    fi
    echo "Dependencies installed successfully."
else
    echo "Error: $REQUIREMENTS_FILE not found in the current directory."
    exit 1
fi

# --- Function to clean up background processes ---
cleanup() {
    echo "\nStopping background processes..."
    # Check if PIDs exist before killing
    if [ ! -z "$EVIDENTLY_PID" ]; then
        echo "Stopping Evidently UI (PID: $EVIDENTLY_PID)..."
        kill $EVIDENTLY_PID
    fi
    if [ ! -z "$MONITOR_PID" ]; then
        echo "Stopping Monitor Script (PID: $MONITOR_PID)..."
        kill $MONITOR_PID
    fi
    echo "Cleanup complete."
    exit 0
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# --- Start Evidently UI in Background ---
echo "[3/5] Starting Evidently UI service in background..."
echo "   (Will run on http://localhost:$EVIDENTLY_PORT)"
evidently ui --workspace $WORKSPACE_DIR --port $EVIDENTLY_PORT &
EVIDENTLY_PID=$! # Capture the Process ID
echo "   Evidently UI PID: $EVIDENTLY_PID"
sleep 3 # Give it a moment to start

# --- Start Monitoring Script in Background ---
echo "[4/5] Starting Periodic Monitoring script ($MONITOR_FILE) in background..."
python3 $MONITOR_FILE &
MONITOR_PID=$! # Capture the Process ID
echo "   Monitor Script PID: $MONITOR_PID"
sleep 1

# --- Start Streamlit App in Foreground ---
echo "[5/5] Starting Streamlit application ($APP_FILE)..."
echo "   (Access at http://localhost:$STREAMLIT_PORT or the URL Streamlit provides)"
echo "   Press CTRL+C in this terminal to stop the Streamlit app AND background processes."

streamlit run $APP_FILE --server.port $STREAMLIT_PORT --server.headless true

# --- Script End ---
# If Streamlit exits normally (e.g., browser tab closed), cleanup might not be triggered by trap.
# Call cleanup explicitly here for normal exit.
echo "\nStreamlit app stopped."
cleanup

