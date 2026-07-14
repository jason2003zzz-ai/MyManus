from app.exceptions import ToolError
from app.tool.base import BaseTool


_TERMINATE_DESCRIPTION = """Terminate the interaction when the request is met OR if the assistant cannot proceed further with the task.
When you have finished all the tasks, call this tool to end the work.
Always include a concise, user-facing final_answer because it is shown as the final response."""


class Terminate(BaseTool):
    name: str = "terminate"
    description: str = _TERMINATE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "The finish status of the interaction.",
                "enum": ["success", "failure"],
            },
            "final_answer": {
                "type": "string",
                "description": "The final user-facing answer. Do not include raw tool logs.",
            },
            "reason": {
                "type": "string",
                "description": "Brief reason for ending the run. Required for failure.",
            },
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Evidence receipt IDs emitted by successful tools that directly "
                    "prove the completion criteria. Use [] only for pure answer tasks."
                ),
            },
        },
        "required": ["status", "final_answer", "evidence_ids"],
    }

    async def execute(
        self,
        status: str,
        final_answer: str = "",
        reason: str = "",
        evidence_ids: list[str] | None = None,
    ) -> str:
        """Finish the current execution"""
        if status not in {"success", "failure"}:
            raise ToolError("Terminate status must be exactly `success` or `failure`.")
        if not isinstance(final_answer, str) or not final_answer.strip():
            raise ToolError("Terminate requires a non-empty user-facing final_answer.")
        if not isinstance(evidence_ids, list):
            raise ToolError("Terminate requires evidence_ids as an array.")
        if status == "failure" and not reason.strip():
            raise ToolError("terminate(failure) requires a non-empty reason.")

        result = f"The interaction has been completed with status: {status}"
        if reason:
            result += f"\nReason: {reason}"
        if final_answer:
            result += f"\nFinal answer: {final_answer}"
        result += f"\nEvidence receipts: {', '.join(evidence_ids) or 'none'}"
        return result
