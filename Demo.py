# Save as opik_demo.py

from opik import Dataset, evaluate
from opik.metrics import answer_relevance, hallucination
import random

# Step 1: Create dataset
mock_data = [
    {"input": "Explain quantum computing", "expected_output": "Quantum computing uses qubits...", "context": ["quantum_physics.pdf"]},
    {"input": "Define machine learning", "expected_output": "ML involves algorithms...", "context": ["ai_basics.txt"]}
]
dataset = Dataset.create(name="mock_evaluation", data=mock_data)

# Step 2: Define pipeline
@evaluate(dataset="mock_evaluation")
def mock_llm_pipeline(input_text: str):
    context = random.choice(["retrieved_context_1", "retrieved_context_2"])
    response = f"Mock response for {input_text}"
    return {
        "output": response,
        "context": context,
        "latency": random.uniform(0.5, 2.0)
    }

# Step 3: Run evaluation
results = mock_llm_pipeline.run(
    metrics=[answer_relevance(), hallucination()],
    experiment_name="mock_experiment_v1"
)

print("Evaluation complete. Visit http://localhost:8501 to view results.")
