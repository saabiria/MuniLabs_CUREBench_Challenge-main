from tools.drug_lookup import lookup_drug
from tools.interaction_checker import check_interaction

class ToolManager:

    def __init__(self):
        self.tools = {
            "lookup_drug": lookup_drug,
            "check_interaction": check_interaction
        }

    def run(self, tool_name, *args):
        if tool_name in self.tools:
            return self.tools[tool_name](*args)
        return "Tool not found"