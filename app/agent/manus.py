import json
import re
from typing import Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import Field, model_validator

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
from app.tool.mcp import MCPClients, MCPClientTool
from app.tool.office import CreateExcelWorkbook, CreateWordDocument
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor


SNAPSHOT_REF_RE = re.compile(r"\[ref=([^\]]+)\]")
SNAPSHOT_BOX_RE = re.compile(r"\[box=([^\]]+)\]")

SNAPSHOT_REF_TARGET_RE = re.compile(r"^f\d+e\d+$")

ACTION_REF_KEYWORDS = (
    "评论区在等你",
    "写评论",
    "发表评论",
    "发评论",
    "评论框",
    "输入评论",
    "comment-input",
    "reply-box",
    "reply-input",
    "textarea",
    "contenteditable",
)

COMMENT_TASK_KEYWORDS = ("评论", "comment", "reply")

SELECTOR_GUESS_PATTERNS = (
    "textarea",
    "contenteditable",
    "placeholder",
    "comment",
    "reply",
    "[",
    ".",
    "#",
)

SEARCH_QUERY_PARAM_NAMES = {"keyword", "q", "query", "wd", "search_query"}
CJK_TEXT_RE = re.compile(r"[\u4e00-\u9fff]+")
QUOTED_SEARCH_INTENT_RE = re.compile(
    r"(?:搜索|搜|查找|寻找|定位|find|search(?:\s+for)?)"
    r"\s*(?:UP主|用户|账号|博主|作者|关键词|关键字)?"
    r"\s*[\"“'「『]([^\"”'」』]{1,80})[\"”'」』]",
    re.IGNORECASE,
)
USER_ENTITY_INTENT_RE = re.compile(
    r"(?:UP主|用户|账号|博主|作者)\s*([A-Za-z0-9_\-\u4e00-\u9fff]{2,40})"
)


class Manus(ToolCallAgent):
    """A versatile general-purpose agent with support for both local and MCP tools."""

    name: str = "MyManus"
    description: str = "A versatile agent that can solve various tasks using multiple tools including MCP-based tools"

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 30000
    max_steps: int = 20

    # MCP clients for remote tool access
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            StrReplaceEditor(),
            CreateWordDocument(),
            CreateExcelWorkbook(),
            AskHuman(),
            Terminate(),
        )
    )

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    # Track connected MCP servers
    connected_servers: Dict[str, str] = Field(
        default_factory=dict
    )  # server_id -> url/command
    _initialized: bool = False

    latest_ref_hints: list[str] = Field(default_factory=list)
    @model_validator(mode="after")
    def initialize_helper(self) -> "Manus":
        """Initialize basic components synchronously."""
        return self

    @classmethod
    async def create(cls, **kwargs) -> "Manus":
        """Factory method to create and properly initialize a Manus instance."""
        instance = cls(**kwargs)
        await instance.initialize_mcp_servers()
        instance._initialized = True
        return instance

    async def initialize_mcp_servers(self) -> None:
        """Initialize connections to configured MCP servers."""
        for server_id, server_config in config.mcp_config.servers.items():
            try:
                if server_config.type == "sse":
                    if server_config.url:
                        await self.connect_mcp_server(server_config.url, server_id)
                        logger.info(
                            f"Connected to MCP server {server_id} at {server_config.url}"
                        )
                elif server_config.type == "stdio":
                    if server_config.command:
                        await self.connect_mcp_server(
                            server_config.command,
                            server_id,
                            use_stdio=True,
                            stdio_args=server_config.args,
                            stdio_env=server_config.env,
                        )
                        logger.info(
                            f"Connected to MCP server {server_id} using command {server_config.command}"
                        )
                elif server_config.type in {"http", "streamableHttp"}:
                    if server_config.url:
                        default_llm = config.llm.get("default")
                        headers = server_config.resolved_headers(
                            getattr(default_llm, "api_key", None)
                        )
                        await self.connect_mcp_server(
                            server_config.url,
                            server_id,
                            use_http=True,
                            headers=headers,
                        )
                        logger.info(
                            f"Connected to MCP server {server_id} at {server_config.url}"
                        )
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {server_id}: {e}")

    async def connect_mcp_server(
        self,
        server_url: str,
        server_id: str = "",
        use_stdio: bool = False,
        use_http: bool = False,
        stdio_args: List[str] = None,
        stdio_env: Dict[str, str] = None,
        headers: Dict[str, str] = None,
    ) -> None:
        """Connect to an MCP server and add its tools."""
        if use_stdio:
            await self.mcp_clients.connect_stdio(
                server_url, stdio_args or [], server_id, env=stdio_env
            )
            self.connected_servers[server_id or server_url] = server_url
        elif use_http:
            await self.mcp_clients.connect_streamable_http(
                server_url, server_id, headers=headers
            )
            self.connected_servers[server_id or server_url] = server_url
        else:
            await self.mcp_clients.connect_sse(server_url, server_id)
            self.connected_servers[server_id or server_url] = server_url

        # Update available tools with only the new tools from this server
        new_tools = [
            tool for tool in self.mcp_clients.tools if tool.server_id == server_id
        ]
        self.available_tools.add_tools(*new_tools)

    async def disconnect_mcp_server(self, server_id: str = "") -> None:
        """Disconnect from an MCP server and remove its tools."""
        await self.mcp_clients.disconnect(server_id)
        if server_id:
            self.connected_servers.pop(server_id, None)
        else:
            self.connected_servers.clear()

        # Rebuild available tools without the disconnected server's tools
        base_tools = [
            tool
            for tool in self.available_tools.tools
            if not isinstance(tool, MCPClientTool)
        ]
        self.available_tools = ToolCollection(*base_tools)
        self.available_tools.add_tools(*self.mcp_clients.tools)

    async def cleanup(self):
        """Clean up Manus agent resources."""
        # Disconnect from all MCP servers only if we were initialized
        if self._initialized:
            await self.disconnect_mcp_server()
            self._initialized = False

    def get_tool_params_for_step(self) -> List[dict]:
        """Expose tools selected by the task-level capability router."""
        allowed_names = self.task_controller.allowed_tool_names(
            tool.name for tool in self.available_tools.tools
        )
        return [
            tool.to_param()
            for tool in self.available_tools.tools
            if tool.name in allowed_names
        ]

    def _user_text(self) -> str:
        parts = []
        for message in self.memory.messages:
            role = getattr(message.role, "value", message.role)
            if role == "user" and message.content:
                parts.append(message.content)
        return "\n".join(parts)

    @staticmethod
    def _is_snapshot_ref(target: object) -> bool:
        return isinstance(target, str) and bool(SNAPSHOT_REF_TARGET_RE.fullmatch(target))

    def _task_needs_comment_interaction(self) -> bool:
        text = self._user_text().lower()
        return any(keyword in text for keyword in COMMENT_TASK_KEYWORDS)

    def _ref_hints_text(self) -> str:
        if not self.latest_ref_hints:
            return ""
        return "\n".join(f"- {hint}" for hint in self.latest_ref_hints[:6])

    def _looks_like_selector_guess(self, target: object) -> bool:
        if not isinstance(target, str):
            return False
        if self._is_snapshot_ref(target):
            return False
        lowered = target.lower()
        return any(pattern in lowered for pattern in SELECTOR_GUESS_PATTERNS)

    def _selector_ref_guard_message(self, target: object) -> str:
        return (
            "Error: Tool router blocked a guessed selector target "
            f"`{target}` because the latest snapshot contains likely exact refs for "
            "this interaction. Use the exact snapshot ref as `target` instead of "
            "guessing CSS selectors such as textarea/comment/reply.\n"
            "Latest likely refs:\n"
            f"{self._ref_hints_text()}"
        )

    def _guard_ref_first_interaction(self, tool_name: str, args: dict) -> Optional[str]:
        if not self.latest_ref_hints or not self._task_needs_comment_interaction():
            return None

        if tool_name in {
            "mcp_playwright_browser_click",
            "mcp_playwright_browser_type",
        }:
            target = args.get("target")
            if self._looks_like_selector_guess(target):
                return self._selector_ref_guard_message(target)

        if tool_name == "mcp_playwright_browser_fill_form":
            for field in args.get("fields") or []:
                target = field.get("target") if isinstance(field, dict) else None
                if self._looks_like_selector_guess(target):
                    return self._selector_ref_guard_message(target)

        return None

    def _guard_playwright_command(self, tool_name: str, args: dict) -> Optional[str]:
        if tool_name == "mcp_playwright_browser_run_code_unsafe":
            script = str(args.get("code") or "")
            if re.search(r"locator\([^)]*\[ref=['\"]", script):
                return (
                    "Error: Snapshot refs are Playwright MCP virtual identifiers, not "
                    "DOM attributes. Do not use CSS selectors such as `[ref=...]` "
                    "inside browser_run_code_unsafe. Pass the pure ref to the normal "
                    "browser tool, or use a real DOM locator."
                )
            if "editor.type(code" in script and "\\n" in script:
                return (
                    "Error: Do not keyboard-type multiline source code into a browser "
                    "code editor; auto-indentation can corrupt the program. Use the "
                    "editor API such as CodeMirror.setValue(code), Monaco model "
                    "setValue(code), or upload the source file."
                )
        return self._guard_ref_first_interaction(tool_name, args)

    @staticmethod
    def _normalize_snapshot_targets(tool_name: str, args: dict) -> dict:
        if tool_name not in {
            "mcp_playwright_browser_click",
            "mcp_playwright_browser_type",
            "mcp_playwright_browser_fill_form",
        }:
            return args

        normalized = dict(args)

        def pure_ref(target: object) -> object:
            if not isinstance(target, str):
                return target
            match = SNAPSHOT_REF_RE.search(target)
            return match.group(1) if match else target

        if "target" in normalized:
            normalized["target"] = pure_ref(normalized["target"])
        if tool_name == "mcp_playwright_browser_fill_form":
            normalized["fields"] = [
                {**field, "target": pure_ref(field.get("target"))}
                if isinstance(field, dict)
                else field
                for field in normalized.get("fields") or []
            ]
        return normalized

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(CJK_TEXT_RE.search(text))

    @staticmethod
    def _normalize_query_text(text: str) -> str:
        return re.sub(r"\s+", "", str(text or "")).strip().lower()

    @staticmethod
    def _cjk_ngrams(text: str, size: int = 2) -> set[str]:
        grams: set[str] = set()
        for block in CJK_TEXT_RE.findall(text):
            if len(block) < size:
                continue
            grams.update(block[index : index + size] for index in range(len(block) - size + 1))
        return grams

    def _query_matches_context(self, query: str, context: str) -> bool:
        normalized_query = self._normalize_query_text(query)
        normalized_context = self._normalize_query_text(context)
        if not normalized_query or not normalized_context:
            return True
        if normalized_query in normalized_context or normalized_context in normalized_query:
            return True

        query_grams = self._cjk_ngrams(query)
        if query_grams and query_grams & self._cjk_ngrams(context):
            return True

        return False

    def _last_assistant_text(self) -> str:
        for message in reversed(self.memory.messages):
            role = getattr(message.role, "value", message.role)
            if role == "assistant" and message.content:
                return message.content
        return ""

    def _extract_intended_search_query(self) -> Optional[str]:
        assistant_text = self._last_assistant_text()
        for source in (assistant_text, self._user_text()):
            if not source:
                continue
            quoted_match = QUOTED_SEARCH_INTENT_RE.search(source)
            if quoted_match:
                return quoted_match.group(1).strip()
            entity_match = USER_ENTITY_INTENT_RE.search(source)
            if entity_match:
                return entity_match.group(1).strip()
        return None

    def _sanitize_search_navigation(
        self, tool_name: str, args: dict
    ) -> tuple[dict, Optional[str]]:
        if tool_name != "mcp_playwright_browser_navigate":
            return args, None

        raw_url = args.get("url")
        if not isinstance(raw_url, str) or not raw_url:
            return args, None

        try:
            parsed = urlsplit(raw_url)
        except ValueError:
            return args, None

        params = parse_qsl(parsed.query, keep_blank_values=True)
        if not params:
            return args, None

        intended_query = self._extract_intended_search_query()
        context = "\n".join(part for part in [self._user_text(), self._last_assistant_text()] if part)
        corrected_params: list[tuple[str, str]] = []
        changed = False

        for key, value in params:
            if key.lower() not in SEARCH_QUERY_PARAM_NAMES or not value:
                corrected_params.append((key, value))
                continue

            if intended_query and self._contains_cjk(value) and not self._query_matches_context(
                value, intended_query
            ):
                logger.warning(
                    "Corrected search URL parameter "
                    f"{key!r} from {value!r} to {intended_query!r} "
                    "to avoid bad manual URL encoding."
                )
                corrected_params.append((key, intended_query))
                changed = True
                continue

            if (
                not intended_query
                and self._contains_cjk(value)
                and context
                and not self._query_matches_context(value, context)
            ):
                return (
                    args,
                    "Error: Tool router blocked a likely wrong search URL. "
                    f"The `{key}` parameter decodes to `{value}`, but that text does not "
                    "match the current task or the agent's stated search intent. Do not "
                    "manually invent percent-encoded Chinese search URLs; use the page "
                    "search box, or build the URL with encodeURIComponent from the exact "
                    "user-provided text.",
                )

            corrected_params.append((key, value))

        if not changed:
            return args, None

        sanitized = dict(args)
        sanitized["url"] = urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urlencode(corrected_params, doseq=True),
                parsed.fragment,
            )
        )
        return sanitized, None

    @staticmethod
    def _short_text(text: str, limit: int = 120) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        return compact[:limit]

    def _extract_snapshot_ref_hints(self, snapshot_text: str) -> list[str]:
        if not self._task_needs_comment_interaction():
            return []

        hints: list[str] = []
        lines = snapshot_text.splitlines()
        for index, line in enumerate(lines):
            ref_match = SNAPSHOT_REF_RE.search(line)
            if not ref_match:
                continue

            block = "\n".join(lines[index : index + 5])
            lowered = block.lower()
            if not any(keyword.lower() in lowered for keyword in ACTION_REF_KEYWORDS):
                continue

            ref = ref_match.group(1)
            box_match = SNAPSHOT_BOX_RE.search(block)
            cleaned = re.sub(r"\[[^\]]+\]", "", block)
            cleaned = re.sub(r"^\s*-\s*", "", cleaned, flags=re.MULTILINE)
            text = self._short_text(cleaned)
            box = f" box={box_match.group(1)}" if box_match else ""
            hints.append(f'target="{ref}"{box} text="{text}"')

            if len(hints) >= 6:
                break

        return hints

    def _augment_result_after_tool(self, tool_name: str, result: str) -> str:
        hints: list[str] = []

        if tool_name == "mcp_playwright_browser_snapshot":
            ref_hints = self._extract_snapshot_ref_hints(result)
            if ref_hints:
                self.latest_ref_hints = ref_hints
                hints.append(
                    "Tool router hint: likely exact snapshot refs for this interaction:\n"
                    f"{self._ref_hints_text()}\n"
                    "Use these exact `target` refs with browser_click/browser_type "
                    "before trying CSS selectors."
                )

        if not hints:
            return result
        return "\n\n".join(hints) + "\n\n" + result

    async def execute_tool(self, command):
        tool_name = command.function.name if command and command.function else ""
        try:
            args = json.loads(command.function.arguments or "{}")
        except Exception:
            args = {}

        if tool_name == "mcp_playwright_browser_snapshot" and (
            self.task_controller.requires_luogu_accept
        ):
            # Inline snapshots are immediately compacted and avoid a redundant
            # file-write/read round trip during problem solving.
            args.pop("filename", None)
            if self.task_controller.submit_panel_seen:
                requested_depth = args.get("depth")
                args["depth"] = max(
                    requested_depth if isinstance(requested_depth, int) else 0,
                    12,
                )
                args["boxes"] = True

        args = self._normalize_snapshot_targets(tool_name, args)
        args, search_guard_message = self._sanitize_search_navigation(tool_name, args)
        if command and command.function:
            command.function.arguments = json.dumps(args, ensure_ascii=False)
        if search_guard_message:
            return search_guard_message

        guard_message = self._guard_playwright_command(tool_name, args)
        if guard_message:
            return guard_message

        result = await super().execute_tool(command)
        result = self._augment_result_after_tool(tool_name, result)
        return result

    async def think(self) -> bool:
        """Process current state and decide next actions with appropriate context."""
        if not self._initialized:
            await self.initialize_mcp_servers()
            self._initialized = True

        return await super().think()
