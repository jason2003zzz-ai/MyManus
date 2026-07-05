import os

from app.tool import BaseTool


class AskHuman(BaseTool):
    """Add a tool to ask human for help."""

    name: str = "ask_human"
    description: str = "Use this tool to ask human for help."
    parameters: str = {
        "type": "object",
        "properties": {
            "inquire": {
                "type": "string",
                "description": "The question you want to ask human.",
            }
        },
        "required": ["inquire"],
    }

    async def execute(self, inquire: str) -> str:
        if os.environ.get("OPENMANUS_WEB_MODE") == "1":
            return (
                "Human input is not available in MyManus Web mode. "
                "Do not wait for terminal input; continue with the available evidence, "
                "or explain the specific blocker in the final answer."
            )
        return input(f"""Bot: {inquire}\n\nYou: """).strip()
