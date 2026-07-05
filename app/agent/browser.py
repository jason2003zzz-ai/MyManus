from app.agent.mcp import MCPAgent


class BrowserAgent(MCPAgent):
    """Browser agent powered by an MCP browser server such as Playwright MCP."""

    name: str = "browser"
    description: str = "A browser agent that controls a browser through MCP tools"
