from typing import List

from pydantic import Field

from app.agent.manus import Manus


ROLE_INSTRUCTIONS = {
    "general": (
        "You are the general implementation specialist in a coordinated team. "
        "Use local tools, email tools, and file-generation tools when relevant, but "
        "do not perform web search or Playwright browser automation. Complete only "
        "the assigned work package and return a concise handoff."
    ),
    "browser": (
        "You are the web research and browser specialist in a coordinated team. "
        "Use read-only web search for discovery and Microsoft Playwright MCP for "
        "navigation, reading, screenshots, and interaction. Do not perform unrelated "
        "local coding or data analysis. Search results and snippets are discovery only; "
        "open or fetch the strongest first-party source and inspect the relevant original "
        "page, navigation, or site search before answering. Never conclude that data is "
        "absent from search snippets alone; a negative conclusion requires a direct "
        "source read plus another relevant query or page path. Do not take screenshots "
        "unless visual evidence is required. "
        "Once the requested facts and source URLs are collected, call terminate "
        "immediately and return a concise structured handoff."
    ),
}


class ScopedManus(Manus):
    """A Manus worker with a hard tool boundary for its team role."""

    team_role: str = "general"
    allowed_tool_names: set[str] = Field(default_factory=set)
    allowed_tool_prefixes: List[str] = Field(default_factory=list)
    denied_tool_prefixes: List[str] = Field(default_factory=list)

    @classmethod
    async def create_for_role(cls, role: str, **kwargs) -> "ScopedManus":
        if role not in ROLE_INSTRUCTIONS:
            raise ValueError(f"Unsupported ScopedManus role: {role}")

        profile = {
            "general": {
                "allowed_tool_names": set(),
                "allowed_tool_prefixes": [],
                "denied_tool_prefixes": ["mcp_playwright_", "mcp_stepsearch_"],
            },
            "browser": {
                "allowed_tool_names": {"terminate", "ask_human"},
                "allowed_tool_prefixes": ["mcp_playwright_", "mcp_stepsearch_"],
                "denied_tool_prefixes": [],
            },
        }[role]
        instance = await cls.create(team_role=role, **profile, **kwargs)
        instance.name = f"MyManus/{role}"
        instance.system_prompt = (
            f"{instance.system_prompt}\n\nTEAM WORKER BOUNDARY:\n"
            f"{ROLE_INSTRUCTIONS[role]}"
        )
        return instance

    def _profile_allows(self, tool_name: str) -> bool:
        if any(tool_name.startswith(prefix) for prefix in self.denied_tool_prefixes):
            return False
        if self.team_role == "general":
            return True
        return tool_name in self.allowed_tool_names or any(
            tool_name.startswith(prefix) for prefix in self.allowed_tool_prefixes
        )

    @staticmethod
    def _tool_param_name(tool_param: dict) -> str:
        function = tool_param.get("function") or {}
        return str(function.get("name") or tool_param.get("name") or "")

    def get_tool_params_for_step(self) -> List[dict]:
        if self.team_role == "browser":
            return [
                tool.to_param()
                for tool in self.available_tools.tools
                if self._profile_allows(tool.name)
            ]
        return [
            tool_param
            for tool_param in super().get_tool_params_for_step()
            if self._profile_allows(self._tool_param_name(tool_param))
        ]

    async def execute_tool(self, command):
        tool_name = command.function.name if command and command.function else ""
        if not self._profile_allows(tool_name):
            return (
                f"Error: Team role '{self.team_role}' is not allowed to use "
                f"tool '{tool_name}'. Return this work to the coordinator."
            )
        return await super().execute_tool(command)
