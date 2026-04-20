class PlannerAgent:
    """
    Breaks a question into reasoning steps for the ClinicalAgent.
    """

    def plan(self, question):
        # Basic prototype: map question type to steps
        steps = [
            "Identify relevant drugs and patient conditions",
            "Retrieve relevant evidence",
            "Check drug interactions and contraindications",
            "Generate step-by-step reasoning",
            "Produce final answer."
        ]
        # You can make this dynamic later using ML classification of question type
        return steps
