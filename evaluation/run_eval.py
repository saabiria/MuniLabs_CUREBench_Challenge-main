import json
from agents.clinical_agent import ClinicalAgent

def run_evaluation():

    agent = ClinicalAgent()

    with open("datasets/sample_questions.json") as f:
        dataset = json.load(f)

    results = []

    for item in dataset:

        question = item["question"]

        output = agent.solve(question)

        results.append(output)

    with open("results.json", "w") as f:

        json.dump(results, f, indent=4)

    print("Evaluation complete. Results saved to results.json")