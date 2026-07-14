import hashlib
import json
import re
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from app.prompt.luogu import LUOGU_SUBMISSION_GUIDE


LUOGU_MARKERS = ("luogu.com.cn", "洛谷")
SUBMISSION_MARKERS = (
    "提交",
    "交题",
    "交上",
    "交一下",
    "做题",
    "做出来",
    "完成",
    "通过",
    "ac这道",
    "ac 这道",
    "solve",
    "submit",
)
BROWSER_TOOL_PREFIX = "mcp_playwright_browser_"
LUOGU_BROWSER_TOOL_SUFFIXES = {
    "navigate",
    "navigate_back",
    "snapshot",
    "take_screenshot",
    "click",
    "type",
    "fill_form",
    "press_key",
    "select_option",
    "file_upload",
    "evaluate",
    "run_code_unsafe",
    "wait_for",
    "tabs",
    "handle_dialog",
    "mouse_click_xy",
    "mouse_wheel",
}
DOM_INTERACTION_SUFFIXES = (
    "click",
    "type",
    "fill_form",
    "select_option",
    "evaluate",
    "run_code_unsafe",
    "press_key",
)
VIEWPORT_MOVE_SUFFIXES = ("mouse_wheel", "mouse_move_xy")
FAILURE_MARKERS = (
    "error:",
    "timeouterror",
    "timeout exceeded",
    "does not match any elements",
    "not found",
    "not visible",
    "not editable",
    "not enabled",
    "target page, context or browser has been closed",
    "execution timeout",
    "success': false",
    'success": false',
)
JUDGING_FAILURE_MARKERS = (
    "unaccepted",
    "wrong answer",
    "runtime error",
    "compile error",
    "time limit exceeded",
    "memory limit exceeded",
    "output limit exceeded",
)
VALID_TERMINATION_STATUSES = {"success", "failure"}

RETRIEVAL_INTENT_RE = re.compile(
    r"搜索|查询|查找|检索|浏览|访问|打开(?:网页|网站|页面)|"
    r"看一下|查看|读取|检查|审查|最新|当前|"
    r"search|look\s*up|find|browse|research",
    re.IGNORECASE,
)
ACTION_INTENT_RE = re.compile(
    r"发送|发邮件|发布|评论|回复|提交|上传|登录|注册|"
    r"购买|下单|删除|点赞|关注|填写|send|publish|post|"
    r"submit|upload|delete|purchase|place\s+(?:an\s+)?order",
    re.IGNORECASE,
)
ARTIFACT_INTENT_RE = re.compile(
    r"(?:生成|创建|制作|导出|保存|写入|编辑|修改|编写|实现|修复)"
    r".{0,18}(?:文件|文档|表格|简历|报告|代码|程序|项目|pdf|word|excel|docx|xlsx)|"
    r"(?:create|generate|export|save|edit|modify|implement|fix|write)"
    r".{0,18}(?:file|document|spreadsheet|report|code|program|project|pdf|word|excel)",
    re.IGNORECASE,
)
EXECUTION_INTENT_RE = re.compile(
    r"运行|执行|测试|验证|编译|调试|" r"\b(?:run|execute|test|verify|compile|pytest)\b",
    re.IGNORECASE,
)

READ_TOOL_MARKERS = (
    "search",
    "snapshot",
    "screenshot",
    "navigate",
    "view",
    "read",
    "get",
    "list",
    "inspect",
    "crawl",
    "wait_for",
)
ATOMIC_ACTION_TOOL_MARKERS = (
    "send",
    "publish",
    "post_",
    "submit",
    "upload",
    "delete",
    "create_event",
    "update_event",
)
COMMIT_ARGUMENT_MARKERS = (
    "发送",
    "发布",
    "提交",
    "上传",
    "删除",
    "确认",
    "send",
    "publish",
    "post",
    "submit",
    "upload",
    "delete",
    "confirm",
)


class ToolAttempt(BaseModel):
    name: str
    fingerprint: str
    result_digest: str
    sequence: int = 0
    failed: bool = False
    result_excerpt: str = ""


class CompletionContract(BaseModel):
    """Deterministic minimum evidence required before a successful finish."""

    task_type: str = "answer"
    required_evidence_kinds: list[str] = Field(default_factory=list)
    required_targets: list[str] = Field(default_factory=list)
    requires_explicit_terminate: bool = False


class EvidenceReceipt(BaseModel):
    receipt_id: str
    sequence: int
    tool_name: str
    kinds: list[str] = Field(default_factory=list)
    arguments_excerpt: str = ""
    result_excerpt: str = ""


class TaskController(BaseModel):
    objective: str = ""
    initialized: bool = False
    domain: str = "general"
    requires_luogu_accept: bool = False
    attempts: list[ToolAttempt] = Field(default_factory=list)
    completion_contract: CompletionContract = Field(default_factory=CompletionContract)
    evidence_receipts: list[EvidenceReceipt] = Field(default_factory=list)
    last_evidence_receipt: str = ""
    tool_result_sequence: int = 0
    evidence_counter: int = 0
    browser_failure_streak: int = 0
    current_url: str = ""
    expected_problem_id: str = ""
    current_problem_id: str = ""
    current_record_id: str = ""
    accepted_record_id: str = ""
    record_problem_id: str = ""
    problem_statement_seen: bool = False
    submit_panel_seen: bool = False
    selected_language: str = ""
    code_entry_seen: bool = False
    submission_attempt_seen: bool = False
    submission_action_seen: bool = False
    luogu_submission_seen: bool = False
    luogu_judging_seen: bool = False
    luogu_accepted_seen: bool = False
    luogu_failed_judgement: str = ""
    captcha_seen: bool = False
    captcha_first_seen_sequence: int = 0
    post_captcha_observation_seen: bool = False
    last_recovery_directive: str = ""

    def initialize(self, objective: str) -> Optional[str]:
        if self.initialized:
            return None

        self.objective = self._extract_user_objective(objective)
        lowered = self.objective.lower()
        if any(marker in lowered for marker in LUOGU_MARKERS):
            self.domain = "luogu"
            self.requires_luogu_accept = any(
                marker in lowered for marker in SUBMISSION_MARKERS
            )
            match = re.search(r"\bP\d{3,6}\b", self.objective, re.IGNORECASE)
            if match:
                self.expected_problem_id = match.group(0).upper()
                self.current_problem_id = self.expected_problem_id

        self.completion_contract = self._build_completion_contract(self.objective)

        self.initialized = True
        return LUOGU_SUBMISSION_GUIDE if self.requires_luogu_accept else None

    @staticmethod
    def _extract_user_objective(objective: str) -> str:
        text = (objective or "").strip()
        marker = "本轮用户任务："
        if marker in text:
            text = text.rsplit(marker, 1)[-1].strip()
        return text

    def _build_completion_contract(self, objective: str) -> CompletionContract:
        required: list[str] = []
        task_types: list[str] = []
        classification_text = re.sub(
            r"https?://\S+|\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            " ",
            objective,
            flags=re.IGNORECASE,
        )
        classification_text = re.sub(
            r"(?:不要|无需|不需要|禁止|请勿|不能|不必|"
            r"do\s+not|don't|must\s+not|without)"
            r"[^,，。;；\n]{0,40}(?=[,，。;；\n]|$)",
            " ",
            classification_text,
            flags=re.IGNORECASE,
        )

        if RETRIEVAL_INTENT_RE.search(classification_text):
            required.append("retrieval")
            task_types.append("retrieval")
        if ACTION_INTENT_RE.search(classification_text):
            required.extend(("action", "verification"))
            task_types.append("action")
        if ARTIFACT_INTENT_RE.search(classification_text):
            required.append("artifact")
            task_types.append("artifact")
        if EXECUTION_INTENT_RE.search(classification_text):
            required.append("execution")
            task_types.append("execution")

        if self.requires_luogu_accept:
            required = ["action", "verification"]
            task_types = ["luogu_submission"]

        return CompletionContract(
            task_type="+".join(task_types) if task_types else "answer",
            required_evidence_kinds=list(dict.fromkeys(required)),
            required_targets=self._extract_required_targets(
                objective, bind_quoted=bool(required)
            ),
            requires_explicit_terminate=bool(required),
        )

    @staticmethod
    def _extract_required_targets(objective: str, *, bind_quoted: bool) -> list[str]:
        targets: list[str] = []
        patterns = (
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            r"https?://[^\s\]\[()<>\"'，。；、,;!?！？]+",
            r"\bP\d{3,6}\b",
            r"(?:[\w.-]+\.(?:docx|xlsx|pdf|csv|json|md|txt|py|cpp|java))\b",
        )
        for pattern in patterns:
            targets.extend(re.findall(pattern, objective, flags=re.IGNORECASE))
        if bind_quoted:
            for pattern in (
                r'"([^"\n]{1,160})"',
                r"'([^'\n]{1,160})'",
                r"“([^”\n]{1,160})”",
                r"「([^」\n]{1,160})」",
                r"『([^』\n]{1,160})』",
            ):
                targets.extend(
                    match.strip()
                    for match in re.findall(pattern, objective)
                    if match.strip()
                )
        return list(dict.fromkeys(targets))

    @staticmethod
    def _normalized_arguments(arguments: Any) -> str:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return re.sub(r"\s+", " ", arguments).strip()[:2000]
        try:
            return json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(arguments)[:2000]

    @classmethod
    def fingerprint(cls, tool_name: str, arguments: Any) -> str:
        payload = f"{tool_name}\n{cls._normalized_arguments(arguments)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _digest_result(result: str) -> str:
        normalized = re.sub(r"\s+", " ", result or "").strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _looks_failed(result: str) -> bool:
        lowered = (result or "").lower()
        return any(marker in lowered for marker in FAILURE_MARKERS)

    def preflight_tool(self, tool_name: str, arguments: Any) -> Optional[str]:
        if tool_name == "terminate":
            if not isinstance(arguments, dict):
                return "Terminate arguments must be a JSON object."
            final_answer = arguments.get("final_answer")
            if not isinstance(final_answer, str) or not final_answer.strip():
                return "Terminate requires a non-empty user-facing final_answer."
            if not isinstance(arguments.get("evidence_ids"), list):
                return "Terminate requires evidence_ids as an array."
            allowed, rejection = self.validate_termination(
                arguments.get("status"),
                evidence_ids=arguments.get("evidence_ids"),
                explicit=True,
                reason=arguments.get("reason", ""),
            )
            return None if allowed else rejection

        arguments_text = self._normalized_arguments(arguments).lower()
        if self.requires_luogu_accept:
            validated_test_seen = self.has_current_validated_test()
            target_url = (
                str((arguments or {}).get("url", ""))
                if isinstance(arguments, dict)
                else ""
            )
            if tool_name.endswith("navigate") and re.search(
                r"luogu\.com\.cn/problem/P\d{3,6}/submit(?:[/?#]|$)",
                target_url,
                re.IGNORECASE,
            ):
                return (
                    "Luogu has no standalone `/problem/<id>/submit` route for this "
                    "workflow. Stay on the exact problem page and use the fresh "
                    "`提交答案` snapshot ref after local tests pass."
                )
            opening_submit_panel = tool_name.endswith(
                ("click", "run_code_unsafe")
            ) and ("提交答案" in arguments_text or "submit answer" in arguments_text)
            if opening_submit_panel and not validated_test_seen:
                return (
                    "Luogu stage guard blocked the submit panel because no validated "
                    "local test evidence exists. Use one `python_execute` call with "
                    "assertions for the official samples only, then submit immediately. "
                    "Do not add speculative hand-written cases before submission."
                )
            if (
                tool_name.endswith("snapshot")
                and self.problem_statement_seen
                and not validated_test_seen
            ):
                return (
                    "The Luogu problem statement is already in memory. Do not capture "
                    "the same full page again. Solve it now and use `python_execute` "
                    "for sample and boundary tests."
                )

        if self.requires_luogu_accept and self.submit_panel_seen:
            submit_attempt = tool_name.endswith(("click", "run_code_unsafe")) and (
                "提交评测" in arguments_text or "submit" in arguments_text
            )
            if submit_attempt and not self.code_entry_seen:
                return (
                    "Luogu stage guard blocked submission because this run has not "
                    "entered or uploaded source code. Replace the current editor "
                    "content (or upload a source file), verify it, then click 提交评测."
                )
            if not self.submission_action_seen and tool_name.endswith("navigate"):
                target_url = (
                    str((arguments or {}).get("url", ""))
                    if isinstance(arguments, dict)
                    else ""
                )
                if "#submit" not in target_url:
                    return (
                        "Luogu stage guard blocked navigation away from the open submit "
                        "panel before this run submitted code. Stay on the panel, select "
                        "the language, fill the editor (or upload a source file), and "
                        "click `提交评测`."
                    )
            if not self.submission_action_seen and (
                "提交记录" in arguments_text or "/record/" in arguments_text
            ):
                return (
                    "Luogu stage guard blocked opening old records before this run "
                    "submitted code. Complete the current submit panel first."
                )

        if tool_name.endswith(VIEWPORT_MOVE_SUFFIXES):
            consecutive_moves = 0
            for attempt in reversed(self.attempts):
                if not attempt.name.endswith(VIEWPORT_MOVE_SUFFIXES):
                    break
                consecutive_moves += 1
            if consecutive_moves >= 3:
                return (
                    "Viewport-movement circuit breaker blocked a fourth consecutive "
                    "scroll/move without semantic progress. Stop moving the viewport. "
                    "Take one fresh deep browser_snapshot with boxes, then use an exact "
                    "ref or a focused screenshot/coordinate action."
                )

        fingerprint = self.fingerprint(tool_name, arguments)
        consecutive = 0
        last_digest = None
        for attempt in reversed(self.attempts):
            if attempt.fingerprint != fingerprint:
                break
            if last_digest is None:
                last_digest = attempt.result_digest
            if attempt.result_digest != last_digest:
                break
            consecutive += 1

        if consecutive < 2:
            return None

        if tool_name.startswith(BROWSER_TOOL_PREFIX):
            return (
                "Tool circuit breaker blocked a third identical browser call after "
                "two unchanged results. Do not repeat it. Capture a fresh "
                "`browser_snapshot` with boxes or a `browser_take_screenshot`, then "
                "use a different exact ref, coordinate action, navigation path, or "
                "file-upload strategy."
            )
        return (
            "Tool circuit breaker blocked a third identical call after two unchanged "
            "results. Change the arguments or choose another strategy."
        )

    def record_tool_result(
        self, tool_name: str, arguments: Any, result: str
    ) -> Optional[str]:
        if tool_name == "terminate":
            return None

        failed = self._looks_failed(result)
        self.tool_result_sequence += 1
        attempt = ToolAttempt(
            name=tool_name,
            fingerprint=self.fingerprint(tool_name, arguments),
            result_digest=self._digest_result(result),
            sequence=self.tool_result_sequence,
            failed=failed,
            result_excerpt=re.sub(r"\s+", " ", result or "").strip()[:500],
        )
        self.attempts.append(attempt)
        self.attempts = self.attempts[-40:]
        self.last_evidence_receipt = ""

        if not failed:
            kinds = self._classify_evidence(tool_name, arguments)
            if kinds:
                self.evidence_counter += 1
                receipt = EvidenceReceipt(
                    receipt_id=f"E{self.evidence_counter}",
                    sequence=self.tool_result_sequence,
                    tool_name=tool_name,
                    kinds=sorted(kinds),
                    arguments_excerpt=self._normalized_arguments(arguments)[:1000],
                    result_excerpt=attempt.result_excerpt,
                )
                self.evidence_receipts.append(receipt)
                self.evidence_receipts = self.evidence_receipts[-80:]
                self._escalate_contract_from_evidence(kinds)
                self.last_evidence_receipt = (
                    f"EVIDENCE RECEIPT {receipt.receipt_id}: "
                    f"kinds={','.join(receipt.kinds)}; tool={tool_name}. "
                    "Reference this receipt in terminate.evidence_ids only when it "
                    "directly proves a completion criterion."
                )

        if tool_name.startswith(BROWSER_TOOL_PREFIX):
            self.browser_failure_streak = (
                self.browser_failure_streak + 1 if failed else 0
            )
            self._record_browser_evidence(tool_name, arguments, result, failed)

        if (
            failed
            and self.browser_failure_streak >= 2
            and tool_name.endswith(DOM_INTERACTION_SUFFIXES)
        ):
            directive = (
                "RECOVERY DIRECTIVE: DOM interaction has failed repeatedly. Stop "
                "guessing selectors. Take a screenshot and a fresh snapshot with "
                "boxes, inspect the actual page state, then switch to exact refs, "
                "coordinate interaction, direct navigation, or source-file upload."
            )
            self.last_recovery_directive = directive
            return directive
        return None

    def _classify_evidence(self, tool_name: str, arguments: Any) -> set[str]:
        name = tool_name.lower()
        arguments_text = self._normalized_arguments(arguments).lower()
        kinds: set[str] = set()

        is_browser = name.startswith(BROWSER_TOOL_PREFIX) or "browser" in name
        is_read = any(marker in name for marker in READ_TOOL_MARKERS)
        if is_read:
            kinds.add("retrieval")

        if name in {"python_execute", "bash", "sandbox_shell"} or any(
            marker in name for marker in ("execute", "run_tests", "pytest", "compile")
        ):
            kinds.add("execution")
        if name == "python_execute" and re.search(
            r"assert\b|assertionerror|unittest|pytest",
            arguments_text,
            re.IGNORECASE,
        ):
            kinds.add("test")

        if name in {"create_word_document", "create_excel_workbook"}:
            kinds.add("artifact")
        elif name == "str_replace_editor":
            if '"command": "view"' in arguments_text:
                kinds.add("retrieval")
            elif any(
                f'"command": "{command}"' in arguments_text
                for command in ("create", "str_replace", "insert", "undo_edit")
            ):
                kinds.add("artifact")

        atomic_action = any(marker in name for marker in ATOMIC_ACTION_TOOL_MARKERS)
        browser_commit = is_browser and any(
            marker in arguments_text for marker in COMMIT_ARGUMENT_MARKERS
        )
        browser_fill = is_browser and name.endswith(("fill_form", "type"))
        if atomic_action or browser_commit or browser_fill:
            kinds.add("action")
        if atomic_action and not is_browser:
            # API-backed mutation tools normally return an authoritative receipt or ID.
            kinds.add("verification")

        previous_action = any(
            "action" in receipt.kinds for receipt in self.evidence_receipts
        )
        if is_read and previous_action:
            kinds.add("verification")

        return kinds

    def has_current_validated_test(self) -> bool:
        latest_success = max(
            (
                receipt.sequence
                for receipt in self.evidence_receipts
                if "test" in receipt.kinds
            ),
            default=0,
        )
        latest_failure = max(
            (
                attempt.sequence
                for attempt in self.attempts
                if attempt.name == "python_execute" and attempt.failed
            ),
            default=0,
        )
        return latest_success > latest_failure

    def _escalate_contract_from_evidence(self, kinds: set[str]) -> None:
        """Make unclassified tool-using tasks obey the same evidence gate."""
        if self.completion_contract.task_type != "answer":
            return

        required = list(self.completion_contract.required_evidence_kinds)
        if "action" in kinds:
            required.extend(("action", "verification"))
        for kind in ("artifact", "execution", "retrieval"):
            if kind in kinds:
                required.append(kind)
        required = list(dict.fromkeys(required))
        if not required:
            return
        self.completion_contract.required_evidence_kinds = required
        self.completion_contract.requires_explicit_terminate = True
        if self.completion_contract.task_type == "answer":
            self.completion_contract.task_type = "tool_assisted"

    def _record_browser_evidence(
        self, tool_name: str, arguments: Any, result: str, failed: bool
    ) -> None:
        text = result or ""
        lowered = text.lower()
        if (
            self.captcha_first_seen_sequence
            and self.tool_result_sequence > self.captcha_first_seen_sequence
            and not failed
        ):
            self.post_captcha_observation_seen = True
        url_match = re.search(r"(?:Page URL|url)\s*:\s*(https?://\S+)", text)
        if url_match:
            self.current_url = url_match.group(1).rstrip("'\"),")
            if "#submit" in self.current_url.lower():
                self.submit_panel_seen = True

        record_id_match = re.search(
            r"luogu\.com\.cn/record/(\d+)(?:\b|[/?#])", self.current_url.lower()
        )
        if record_id_match:
            self.current_record_id = record_id_match.group(1)

        problem_match = re.search(r"\bP\d{3,6}\b", text, re.IGNORECASE)
        if problem_match:
            self.current_problem_id = problem_match.group(0).upper()

        if "验证码" in text or "captcha" in lowered:
            self.captcha_seen = True
            if not self.captcha_first_seen_sequence:
                self.captcha_first_seen_sequence = self.tool_result_sequence

        if "提交代码" in text and "提交文件" in text:
            self.submit_panel_seen = True
        if "题目描述" in text and "输入格式" in text and "输出格式" in text:
            self.problem_statement_seen = True

        arguments_text = self._normalized_arguments(arguments).lower()
        for language in ("Python 3", "PyPy 3", "C++14", "C++17", "C++20", "C++23"):
            if language.lower() in arguments_text and not failed:
                self.selected_language = language
                break

        if (
            self.submit_panel_seen
            and not failed
            and tool_name.endswith(
                ("type", "fill_form", "evaluate", "run_code_unsafe", "file_upload")
            )
        ):
            self.code_entry_seen = True

        mutating_submission_call = tool_name.endswith(
            ("click", "run_code_unsafe")
        ) and (
            "提交答案" in arguments_text
            or "提交评测" in arguments_text
            or "submit" in arguments_text
        )
        submission_commit_call = tool_name.endswith(("click", "run_code_unsafe")) and (
            "提交评测" in arguments_text
            or (self.submit_panel_seen and "submit" in arguments_text)
        )
        if not failed and submission_commit_call:
            # Browser clicks can return the old problem URL even when Luogu has
            # already started the navigation to the newly-created record page.
            self.submission_attempt_seen = True

        current_page_is_record = bool(record_id_match)
        submission_transition = (
            "代码提交状态" in text
            or "验证码" in text
            or "captcha" in lowered
            or current_page_is_record
        )
        if not failed and mutating_submission_call and submission_transition:
            self.submission_action_seen = True

        record_context = current_page_is_record or "代码提交状态" in text
        if record_context:
            self.luogu_submission_seen = True
            if self.submission_attempt_seen:
                self.submission_action_seen = True
            if problem_match:
                self.record_problem_id = problem_match.group(0).upper()

        if record_context and "judging" in lowered:
            self.luogu_judging_seen = True

        if record_context and re.search(r"\baccepted\b", lowered):
            # A record detail may include accepted test points while the overall
            # record is still Unaccepted. Overall failure takes precedence.
            if not any(marker in lowered for marker in JUDGING_FAILURE_MARKERS):
                self.luogu_accepted_seen = True
                self.accepted_record_id = self.current_record_id
                self.luogu_judging_seen = False
                self.luogu_failed_judgement = ""

        if record_context:
            for marker in JUDGING_FAILURE_MARKERS:
                if marker in lowered:
                    self.luogu_failed_judgement = marker
                    self.luogu_accepted_seen = False
                    self.luogu_judging_seen = False
                    break

    def validate_termination(
        self,
        status: Any,
        evidence_ids: Any = None,
        *,
        explicit: bool = True,
        reason: str = "",
    ) -> tuple[bool, str]:
        if status not in VALID_TERMINATION_STATUSES:
            return (
                False,
                "Invalid termination status. Use exactly `success` or `failure`.",
            )
        if status == "failure":
            if not isinstance(reason, str) or not reason.strip():
                return False, "terminate(failure) requires a concrete non-empty reason."
            if (
                self.completion_contract.requires_explicit_terminate
                and not self.attempts
            ):
                return (
                    False,
                    "terminate(failure) is premature: no tool attempt or observed blocker "
                    "exists for this external task.",
                )
            if (
                self.requires_luogu_accept
                and self.captcha_seen
                and not self.luogu_submission_seen
                and not self.post_captcha_observation_seen
            ):
                return (
                    False,
                    "A CAPTCHA was observed, but no fresh browser observation was "
                    "made afterwards. The human may have completed it. Wait briefly, "
                    "then take a new browser snapshot and inspect the resulting record "
                    "before terminating as failure.",
                )
            return True, ""

        if self.requires_luogu_accept:
            allowed, rejection = self._validate_luogu_completion()
            if not allowed:
                return allowed, rejection

        return self._validate_contract_evidence(evidence_ids, explicit=explicit)

    def _validate_luogu_completion(self) -> tuple[bool, str]:
        if self.luogu_accepted_seen and self.submission_action_seen:
            if not self.accepted_record_id:
                return (
                    False,
                    "Accepted text was observed without an exact Luogu record ID. Open "
                    "the new record detail page and verify its overall status.",
                )
            if (
                self.expected_problem_id
                and self.record_problem_id
                and self.record_problem_id != self.expected_problem_id
            ):
                return (
                    False,
                    f"Accepted record belongs to {self.record_problem_id}, but the task "
                    f"requires {self.expected_problem_id}.",
                )
            return True, ""
        if self.luogu_accepted_seen and not self.submission_action_seen:
            return (
                False,
                "An Accepted record is visible, but no submission action from this "
                "run has been observed. Do not reuse an older accepted record; open "
                "the submit panel and submit the current solution first.",
            )
        if self.captcha_seen and not self.luogu_submission_seen:
            return (
                False,
                "A Luogu CAPTCHA was observed and no judged submission record has "
                "been verified. Ask the human to complete it, then inspect the record.",
            )
        if self.luogu_judging_seen:
            return (
                False,
                "The Luogu submission is still Judging. Wait with a bounded delay and "
                "inspect the current submission record again.",
            )
        if self.luogu_failed_judgement:
            return (
                False,
                "The current Luogu record is not accepted "
                f"({self.luogu_failed_judgement}). Diagnose, fix, retest, and resubmit.",
            )
        return (
            False,
            "No current Luogu submission record with overall status Accepted has been "
            "observed. Submit the solution and verify the resulting record before "
            "terminating successfully.",
        )

    def _validate_contract_evidence(
        self, evidence_ids: Any, *, explicit: bool
    ) -> tuple[bool, str]:
        contract = self.completion_contract
        required = set(contract.required_evidence_kinds)
        if not required:
            return True, ""
        if contract.requires_explicit_terminate and not explicit:
            return (
                False,
                "This task changes or reads external state and must finish with an "
                "explicit terminate(success) call that cites evidence receipt IDs.",
            )
        if not isinstance(evidence_ids, list) or not evidence_ids:
            available = (
                ", ".join(receipt.receipt_id for receipt in self.evidence_receipts[-8:])
                or "none"
            )
            return (
                False,
                "Completion evidence is required. Cite the relevant receipt IDs in "
                f"terminate.evidence_ids. Available receipts: {available}.",
            )

        normalized_ids = {
            item.strip()
            for item in evidence_ids
            if isinstance(item, str) and item.strip()
        }
        receipts_by_id = {
            receipt.receipt_id: receipt for receipt in self.evidence_receipts
        }
        unknown = sorted(normalized_ids - receipts_by_id.keys())
        if unknown:
            return (
                False,
                f"Unknown or stale evidence receipt IDs: {', '.join(unknown)}.",
            )

        selected = [receipts_by_id[receipt_id] for receipt_id in normalized_ids]
        observed = {kind for receipt in selected for kind in receipt.kinds}
        missing = sorted(required - observed)
        if missing:
            return (
                False,
                "The cited evidence does not satisfy the completion contract. Missing "
                f"evidence kinds: {', '.join(missing)}.",
            )

        evidence_text = "\n".join(
            f"{receipt.arguments_excerpt}\n{receipt.result_excerpt}"
            for receipt in selected
        ).lower()
        missing_targets = [
            target
            for target in contract.required_targets
            if target.lower() not in evidence_text
        ]
        if missing_targets:
            return (
                False,
                "The cited evidence is not bound to all explicit task targets. Missing "
                f"targets: {', '.join(missing_targets)}.",
            )

        if {"action", "verification"}.issubset(required):
            atomic = any(
                {"action", "verification"}.issubset(set(receipt.kinds))
                for receipt in selected
            )
            action_sequences = [
                receipt.sequence for receipt in selected if "action" in receipt.kinds
            ]
            verification_sequences = [
                receipt.sequence
                for receipt in selected
                if "verification" in receipt.kinds
            ]
            ordered = bool(
                action_sequences
                and verification_sequences
                and max(verification_sequences) > min(action_sequences)
            )
            if not atomic and not ordered:
                return (
                    False,
                    "External actions require verification evidence produced after the "
                    "action, or one authoritative atomic-tool receipt.",
                )
        return True, ""

    def compact_observation(self, tool_name: str, result: str) -> str:
        """Trim Luogu problem snapshots to task controls and statement content."""
        if (
            not self.requires_luogu_accept
            or not tool_name.endswith("snapshot")
            or len(result) <= 16000
            or "luogu.com.cn/problem/" not in result
        ):
            return result

        lines = result.splitlines()
        statement_start = next(
            (
                index
                for index, line in enumerate(lines)
                if 'heading "题目背景"' in line or 'heading "题目描述"' in line
            ),
            None,
        )
        if statement_start is None:
            return result

        statement_end = next(
            (
                index
                for index, line in enumerate(
                    lines[statement_start + 1 :], statement_start + 1
                )
                if "- contentinfo" in line
            ),
            len(lines),
        )
        metadata = [
            line
            for line in lines[:statement_start]
            if (
                line.startswith("### Page")
                or line.startswith("- Page URL:")
                or line.startswith("- Page Title:")
                or 'heading "P' in line
                or "提交答案" in line
                or "历史分数" in line
            )
        ]
        compacted = "\n".join(
            [
                "[Luogu snapshot compacted to controls and problem statement]",
                *metadata,
                *lines[statement_start:statement_end],
            ]
        )
        if len(compacted) <= 16000:
            return compacted
        return (
            compacted[:10000]
            + "\n\n...[middle of long statement snapshot omitted]...\n\n"
            + compacted[-5000:]
        )

    def allowed_tool_names(self, names: Iterable[str]) -> set[str]:
        available = set(names)
        if self.domain != "luogu":
            return available

        local = {"python_execute", "str_replace_editor", "ask_human", "terminate"}
        browser = {
            name
            for name in available
            if name.startswith(BROWSER_TOOL_PREFIX)
            and name.removeprefix(BROWSER_TOOL_PREFIX) in LUOGU_BROWSER_TOOL_SUFFIXES
        }
        return (local | browser) & available

    def progress_text(self) -> str:
        contract = self.completion_contract
        receipt_summary = (
            ", ".join(
                f"{receipt.receipt_id}({'+'.join(receipt.kinds)})"
                for receipt in self.evidence_receipts[-8:]
            )
            or "none"
        )
        contract_text = (
            "COMPLETION CONTRACT: "
            f"task_type={contract.task_type}; "
            f"required_evidence={','.join(contract.required_evidence_kinds) or 'none'}; "
            f"required_targets={','.join(contract.required_targets) or 'none'}; "
            f"evidence_receipts={receipt_summary}."
        )
        if contract.requires_explicit_terminate:
            contract_text += (
                " Finish only with terminate(status='success', evidence_ids=[...]) "
                "after the cited receipts satisfy every required evidence kind."
            )
        if not self.requires_luogu_accept:
            return contract_text
        stage_directive = ""
        validated_test_seen = self.has_current_validated_test()
        if self.problem_statement_seen and not validated_test_seen:
            stage_directive = (
                " The complete problem statement has already been observed. Stop "
                "reading or navigating; implement the solution and run assertions for "
                "the official samples only in python_execute now. Once they pass, "
                "open the submit panel immediately without adding hand-written cases."
            )
        elif (
            self.captcha_seen
            and not self.luogu_submission_seen
            and not self.post_captcha_observation_seen
        ):
            stage_directive = (
                " A CAPTCHA was observed. The human may complete it in the browser. "
                "Wait briefly and take a fresh browser snapshot before deciding the "
                "task failed; inspect any resulting record to its final judge state."
            )
        elif self.submit_panel_seen and not self.submission_action_seen:
            stage_directive = (
                " Stay on the current submit panel. Do not inspect old records yet; "
                "select the language, replace the editor content, and click 提交评测."
            )
        elif self.submission_action_seen and not self.luogu_accepted_seen:
            stage_directive = (
                " This run submitted code. Now inspect the resulting current record "
                "until it reaches a final judge state."
            )
        luogu_text = (
            "LUOGU TASK STATE: "
            f"expected_problem={self.expected_problem_id or 'unknown'}, "
            f"current_problem={self.current_problem_id or 'unknown'}, "
            f"record={self.current_record_id or 'none'}, "
            f"submit_panel={self.submit_panel_seen}, "
            f"language={self.selected_language or 'unverified'}, "
            f"code_entered={self.code_entry_seen}, "
            f"submitted_this_run={self.submission_action_seen}, "
            f"submission_seen={self.luogu_submission_seen}, "
            f"judging={self.luogu_judging_seen}, "
            f"accepted={self.luogu_accepted_seen}, "
            f"failed_status={self.luogu_failed_judgement or 'none'}, "
            f"captcha_seen={self.captcha_seen}."
            f"{stage_directive}"
        )
        return f"{contract_text}\n\n{luogu_text}"
