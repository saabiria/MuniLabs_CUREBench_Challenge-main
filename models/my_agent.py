from agents.clinical_agent import ClinicalAgent


class MyAgent:

    def __init__(self):
        self.agent = ClinicalAgent()

    def generate(self, question):
        result = self.agent.solve(question)

        return {
            "prediction": result["final_answer"],
            "reasoning_trace": result.get("reasoning_trace", [])
        }