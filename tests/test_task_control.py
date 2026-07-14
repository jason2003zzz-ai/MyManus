import json

import pytest

from app.agent.task_control import TaskController
from app.agent.toolcall import ToolCallAgent
from app.schema import AgentState, Function, Message, ToolCall
from app.tool import Terminate, ToolCollection
from app.tool.base import BaseTool
from app.tool.mcp import MCPClients


def luogu_controller() -> TaskController:
    controller = TaskController()
    guide = controller.initialize("请在洛谷完成 P1001，提交代码并确认通过")
    assert guide
    assert controller.requires_luogu_accept
    return controller


def test_colloquial_luogu_submit_intent_is_detected():
    controller = TaskController()
    guide = controller.initialize(
        "https://www.luogu.com.cn/problem/P17036，自己把这道题做出来交上。"
    )

    assert guide
    assert controller.requires_luogu_accept
    assert controller.expected_problem_id == "P17036"


@pytest.mark.asyncio
async def test_agent_uses_current_task_objective_in_continued_conversation(monkeypatch):
    async def fake_base_run(self, request=None):
        return "done"

    async def fake_cleanup(self):
        return None

    monkeypatch.setattr("app.agent.base.BaseAgent.run", fake_base_run)
    monkeypatch.setattr(ToolCallAgent, "cleanup", fake_cleanup)

    agent = ToolCallAgent()
    history_prompt = """
    历史对话：上一轮完成了 https://www.luogu.com.cn/problem/P1838。
    如果本轮出现“他们”“上述”“里面”“这两个”“这些”等指代，需要解析。
    本轮用户追加要求：https://www.luogu.com.cn/problem/P17036，自己把这道题做出来交上。
    """
    current_prompt = "https://www.luogu.com.cn/problem/P17036，自己把这道题做出来交上。"

    await agent.run(history_prompt, task_objective=current_prompt)

    assert agent.task_controller.objective == current_prompt
    assert agent.task_controller.expected_problem_id == "P17036"
    assert agent.task_controller.completion_contract.required_targets == [
        "https://www.luogu.com.cn/problem/P17036",
        "P17036",
    ]


def test_luogu_guards_guessed_submit_url():
    controller = luogu_controller()

    rejection = controller.preflight_tool(
        "mcp_playwright_browser_navigate",
        {"url": "https://www.luogu.com.cn/problem/P1001/submit"},
    )

    assert rejection
    assert "no standalone" in rejection


def test_luogu_requires_local_test_before_submit_panel():
    controller = luogu_controller()

    rejection = controller.preflight_tool(
        "mcp_playwright_browser_click",
        {"element": "提交答案", "target": "f1e10"},
    )
    assert rejection
    assert "validated local test evidence" in rejection

    controller.record_tool_result(
        "python_execute",
        {"code": "# Expected: 2\nprint(1 + 1)"},
        "2\nProcess finished successfully",
    )
    rejection = controller.preflight_tool(
        "mcp_playwright_browser_click",
        {"element": "提交答案", "target": "f1e10"},
    )
    assert rejection
    assert "official samples only" in rejection

    controller.record_tool_result(
        "python_execute",
        {"code": "assert 1 + 1 == 2"},
        "Process finished successfully\n",
    )
    assert (
        controller.preflight_tool(
            "mcp_playwright_browser_click",
            {"element": "提交答案", "target": "f1e10"},
        )
        is None
    )


def test_multiline_assertion_is_classified_as_test_evidence():
    controller = luogu_controller()
    controller.record_tool_result(
        "python_execute",
        {
            "code": (
                "def solve(value):\n"
                "    return value\n\n"
                "assert solve('sample') == 'sample'\n"
                "print('official samples passed')"
            )
        },
        "{'observation': 'official samples passed\\n', 'success': True}",
    )

    assert controller.has_current_validated_test()
    assert "test" in controller.evidence_receipts[-1].kinds


def test_luogu_progress_does_not_request_speculative_tests():
    controller = luogu_controller()
    controller.problem_statement_seen = True

    progress = controller.progress_text()

    assert "official samples only" in progress
    assert "adversarial" not in progress


def test_failed_test_after_success_requires_fresh_validation():
    controller = luogu_controller()
    controller.record_tool_result(
        "python_execute",
        {"code": "assert solve('sample') == 'expected'"},
        "Process finished successfully",
    )
    assert controller.has_current_validated_test()

    controller.record_tool_result(
        "python_execute",
        {"code": "assert solve('invalid-case') == 'wrong-expectation'"},
        "{'observation': 'AssertionError', 'success': False}",
    )
    assert not controller.has_current_validated_test()

    rejection = controller.preflight_tool(
        "mcp_playwright_browser_click",
        {"element": "提交答案", "target": "f1e10"},
    )
    assert rejection
    assert "validated local test evidence" in rejection


def test_viewport_movement_fuse_ignores_changed_scroll_amounts():
    controller = luogu_controller()
    name = "mcp_playwright_browser_mouse_wheel"
    for delta in (300, -500, 1000):
        controller.record_tool_result(
            name,
            {"deltaX": 0, "deltaY": delta},
            "Scrolled viewport",
        )

    rejection = controller.preflight_tool(name, {"deltaX": 0, "deltaY": -300})

    assert rejection
    assert "fourth consecutive" in rejection


def test_luogu_blocks_repeated_statement_snapshot_before_tests():
    controller = luogu_controller()
    controller.record_tool_result(
        "mcp_playwright_browser_snapshot",
        {},
        "Page URL: https://www.luogu.com.cn/problem/P1001\n题目描述\n输入格式\n输出格式",
    )

    rejection = controller.preflight_tool("mcp_playwright_browser_snapshot", {})

    assert rejection
    assert "already in memory" in rejection


def test_luogu_problem_snapshot_is_compacted():
    controller = luogu_controller()
    navigation_noise = "\n".join(f"- link navigation-{index}" for index in range(2000))
    statement = (
        '- heading "题目描述" [level=2]\n'
        "- paragraph: solve this problem\n"
        '- heading "输入格式" [level=2]\n'
        "- paragraph: two integers\n"
        '- heading "输出格式" [level=2]\n'
        "- paragraph: their sum\n"
    )
    snapshot = (
        "### Page\n"
        "- Page URL: https://www.luogu.com.cn/problem/P1001\n"
        "- Page Title: P1001\n"
        f"{navigation_noise}\n"
        "- generic [ref=f1e9]: 提交答案\n"
        f"{statement}"
        "- contentinfo\n"
    )

    compacted = controller.compact_observation(
        "mcp_playwright_browser_snapshot", snapshot
    )

    assert len(compacted) < len(snapshot) / 2
    assert "solve this problem" in compacted
    assert "提交答案" in compacted


def test_repeated_identical_tool_call_is_blocked_on_third_attempt():
    controller = luogu_controller()
    controller.record_tool_result(
        "python_execute",
        {"code": "assert True"},
        "Process finished successfully",
    )
    name = "mcp_playwright_browser_click"
    arguments = {"target": "提交答案"}
    result = 'Error: "提交答案" does not match any elements.'

    controller.record_tool_result(name, arguments, result)
    controller.record_tool_result(name, arguments, result)

    rejection = controller.preflight_tool(name, arguments)
    assert rejection
    assert "third identical browser call" in rejection


def test_repeated_dom_failures_request_visual_recovery():
    controller = luogu_controller()
    first = controller.record_tool_result(
        "mcp_playwright_browser_click",
        {"target": "missing-a"},
        "Error: target not found",
    )
    second = controller.record_tool_result(
        "mcp_playwright_browser_type",
        {"target": "missing-b", "text": "code"},
        "Error: target not visible",
    )

    assert first is None
    assert second
    assert "screenshot" in second.lower()


def test_unaccepted_record_cannot_pass_completion_gate():
    controller = luogu_controller()
    controller.record_tool_result(
        "mcp_playwright_browser_snapshot",
        {},
        """
        Page URL: https://www.luogu.com.cn/record/123
        Overall status: Unaccepted
        Test point 1: Accepted
        Test point 2: Wrong Answer
        """,
    )

    allowed, reason = controller.validate_termination("success")
    assert not allowed
    assert "not accepted" in reason


def test_current_accepted_record_passes_completion_gate():
    controller = luogu_controller()
    controller.record_tool_result(
        "mcp_playwright_browser_snapshot",
        {},
        "Page URL: https://www.luogu.com.cn/problem/P1001\n提交代码\n提交文件",
    )
    controller.record_tool_result(
        "mcp_playwright_browser_click",
        {"element": "提交答案", "target": "f1e100"},
        "Page URL: https://www.luogu.com.cn/problem/P1001\n代码提交状态：Judging",
    )
    controller.record_tool_result(
        "mcp_playwright_browser_snapshot",
        {},
        """
        Page URL: https://www.luogu.com.cn/record/456
        P1001 submission record
        Overall status: Accepted
        """,
    )

    allowed, reason = controller.validate_termination(
        "success", evidence_ids=["E2", "E3"]
    )
    assert allowed
    assert reason == ""


@pytest.mark.asyncio
async def test_terminate_success_is_rejected_until_acceptance():
    agent = ToolCallAgent()
    agent.task_controller = luogu_controller()

    await agent._handle_special_tool(
        "terminate",
        "done",
        tool_input={
            "status": "success",
            "final_answer": "submitted",
            "evidence_ids": [],
        },
    )
    assert agent.state == AgentState.IDLE
    assert agent._termination_rejection

    agent.task_controller.submit_panel_seen = True
    agent.task_controller.record_tool_result(
        "mcp_playwright_browser_click",
        {"element": "提交评测", "target": "f1e100"},
        "Page URL: https://www.luogu.com.cn/problem/P1001\n代码提交状态：Judging",
    )
    agent.task_controller.record_tool_result(
        "mcp_playwright_browser_snapshot",
        {},
        "Page URL: https://www.luogu.com.cn/record/789\nP1001\nStatus: Accepted",
    )
    await agent._handle_special_tool(
        "terminate",
        "done",
        tool_input={
            "status": "success",
            "final_answer": "P1001 Accepted",
            "evidence_ids": ["E1", "E2"],
        },
    )
    assert agent.state == AgentState.FINISHED
    assert agent.final_answer == "P1001 Accepted"


def test_old_accepted_record_does_not_count_as_current_submission():
    controller = luogu_controller()
    controller.record_tool_result(
        "mcp_playwright_browser_navigate",
        {"url": "https://www.luogu.com.cn/record/list?pid=P1001"},
        "Page URL: https://www.luogu.com.cn/record/list?pid=P1001\nStatus: Accepted",
    )

    allowed, reason = controller.validate_termination("success")
    assert not allowed
    assert "No current Luogu submission record" in reason


def test_submit_stage_blocks_navigation_to_old_records():
    controller = luogu_controller()
    controller.record_tool_result(
        "mcp_playwright_browser_snapshot",
        {},
        "Page URL: https://www.luogu.com.cn/problem/P1001#submit\n提交代码\n提交文件",
    )

    rejection = controller.preflight_tool(
        "mcp_playwright_browser_navigate",
        {"url": "https://www.luogu.com.cn/record/list?pid=P1001"},
    )
    assert rejection
    assert "submit panel" in rejection


def test_submit_button_name_is_recognized():
    controller = luogu_controller()
    controller.record_tool_result(
        "mcp_playwright_browser_snapshot",
        {},
        "Page URL: https://www.luogu.com.cn/problem/P1001#submit\n提交代码\n提交文件",
    )
    controller.record_tool_result(
        "mcp_playwright_browser_type",
        {"target": "f1e564", "text": "print(sum(map(int, input().split())))"},
        "Code editor updated",
    )
    controller.record_tool_result(
        "mcp_playwright_browser_click",
        {"element": "提交评测", "target": "f1e578"},
        "Page URL: https://www.luogu.com.cn/problem/P1001#submit\n代码提交状态：Judging",
    )

    assert controller.submission_action_seen
    assert controller.luogu_judging_seen


def test_submit_click_can_be_confirmed_by_a_later_record_snapshot():
    controller = luogu_controller()
    controller.record_tool_result(
        "mcp_playwright_browser_snapshot",
        {},
        "Page URL: https://www.luogu.com.cn/problem/P1001#submit\n提交代码\n提交文件",
    )
    controller.record_tool_result(
        "mcp_playwright_browser_type",
        {"target": "f1e564", "text": "print(sum(map(int, input().split())))"},
        "Code editor updated",
    )
    controller.record_tool_result(
        "mcp_playwright_browser_click",
        {"element": "提交评测", "target": "f1e578"},
        "Page URL: https://www.luogu.com.cn/problem/P1001#submit",
    )

    assert controller.submission_attempt_seen
    assert not controller.submission_action_seen

    controller.record_tool_result(
        "mcp_playwright_browser_snapshot",
        {},
        "Page URL: https://www.luogu.com.cn/record/790\nP1001\nStatus: Accepted",
    )

    assert controller.submission_action_seen
    allowed, reason = controller.validate_termination(
        "success", evidence_ids=["E2", "E4"]
    )
    assert allowed
    assert reason == ""


def test_luogu_contract_is_not_expanded_by_incidental_tool_kinds():
    controller = luogu_controller()

    controller.record_tool_result(
        "python_execute",
        {"code": "assert solve('sample') == 'expected'"},
        "Process finished successfully",
    )
    controller.record_tool_result(
        "str_replace_editor",
        {"command": "create", "path": "solution.py", "file_text": "print(1)"},
        "File created successfully",
    )

    assert controller.completion_contract.required_evidence_kinds == [
        "action",
        "verification",
    ]


def test_exact_chinese_luogu_prompt_can_finish_with_submit_and_record_evidence():
    controller = TaskController()
    controller.initialize("https://www.luogu.com.cn/problem/P17036，自己把这道题做出来交上。")
    controller.record_tool_result(
        "mcp_playwright_browser_snapshot",
        {},
        "Page URL: https://www.luogu.com.cn/problem/P17036#submit\n提交代码\n提交文件",
    )
    controller.record_tool_result(
        "mcp_playwright_browser_type",
        {"target": "f1e457", "text": "print('ok')"},
        "Code editor updated",
    )
    controller.record_tool_result(
        "mcp_playwright_browser_click",
        {"element": "提交评测", "target": "f1e475"},
        "Page URL: https://www.luogu.com.cn/problem/P17036#submit",
    )
    controller.record_tool_result(
        "mcp_playwright_browser_snapshot",
        {},
        "Page URL: https://www.luogu.com.cn/record/285098732\nP17036\nStatus: Accepted",
    )

    allowed, reason = controller.validate_termination(
        "success", evidence_ids=["E3", "E4"]
    )

    assert allowed
    assert reason == ""


def test_captcha_requires_fresh_browser_observation_before_failure():
    controller = luogu_controller()
    controller.record_tool_result(
        "mcp_playwright_browser_snapshot",
        {},
        "Page URL: https://www.luogu.com.cn/problem/P1001#submit\n提交代码\n提交文件",
    )
    controller.record_tool_result(
        "mcp_playwright_browser_type",
        {"target": "f1e457", "text": "print('ok')"},
        "Code editor updated",
    )
    controller.record_tool_result(
        "mcp_playwright_browser_click",
        {"element": "提交评测", "target": "f1e475"},
        "Page URL: https://www.luogu.com.cn/problem/P1001#submit\n请输入验证码",
    )

    allowed, reason = controller.validate_termination(
        "failure", reason="CAPTCHA blocked submission"
    )
    assert not allowed
    assert "fresh browser observation" in reason

    controller.record_tool_result(
        "mcp_playwright_browser_snapshot",
        {},
        "Page URL: https://www.luogu.com.cn/record/123456\nP1001\nStatus: Accepted",
    )
    allowed, reason = controller.validate_termination(
        "success", evidence_ids=["E3", "E4"]
    )
    assert allowed
    assert reason == ""


def test_submit_is_blocked_until_code_is_entered():
    controller = luogu_controller()
    controller.record_tool_result(
        "mcp_playwright_browser_snapshot",
        {},
        "Page URL: https://www.luogu.com.cn/problem/P1001#submit\n提交代码\n提交文件",
    )

    rejection = controller.preflight_tool(
        "mcp_playwright_browser_click",
        {"element": "提交评测", "target": "f1e578"},
    )
    assert rejection
    assert "has not entered or uploaded source code" in rejection


def test_memory_compaction_preserves_task_and_recent_context():
    agent = ToolCallAgent(max_context_chars=300, keep_recent_messages=4)
    agent.memory.add_message(Message.user_message("original task"))
    for index in range(10):
        agent.memory.add_message(
            Message.assistant_message(f"old result {index} " + "x" * 100)
        )

    agent._compact_memory_if_needed()

    assert agent.memory.messages[0].content == "original task"
    assert any(
        "COMPACTED EXECUTION HISTORY" in (message.content or "")
        for message in agent.memory.messages
    )
    assert agent.memory.messages[-1].content.startswith("old result 9")


def test_mcp_sessions_are_instance_scoped():
    first = MCPClients()
    second = MCPClients()
    first.sessions["playwright"] = object()

    assert "playwright" not in second.sessions
    assert first.sessions is not second.sessions


def test_general_action_requires_cited_evidence():
    controller = TaskController()
    controller.initialize("请给 test@example.com 发送一封邮件")

    allowed, reason = controller.validate_termination(
        "success", evidence_ids=[], explicit=True
    )

    assert not allowed
    assert "Completion evidence is required" in reason
    assert controller.completion_contract.required_evidence_kinds == [
        "action",
        "verification",
    ]


def test_atomic_action_receipt_satisfies_action_contract():
    controller = TaskController()
    controller.initialize("请给 test@example.com 发送一封邮件")
    controller.record_tool_result(
        "mcp_gmail_send_email",
        {"to": "test@example.com", "subject": "hello"},
        "Message sent successfully. Message ID: abc123",
    )

    allowed, reason = controller.validate_termination(
        "success", evidence_ids=["E1"], explicit=True
    )

    assert allowed
    assert reason == ""


def test_action_evidence_must_match_explicit_target():
    controller = TaskController()
    controller.initialize("请给 target@example.com 发送一封邮件")
    controller.record_tool_result(
        "mcp_gmail_send_email",
        {"to": "wrong@example.com", "subject": "hello"},
        "Message sent successfully. Message ID: abc123",
    )

    allowed, reason = controller.validate_termination(
        "success", evidence_ids=["E1"], explicit=True
    )

    assert not allowed
    assert "target@example.com" in reason


def test_successful_tool_use_escalates_unclassified_task_contract():
    controller = TaskController()
    controller.initialize("帮我处理一下")
    assert not controller.completion_contract.requires_explicit_terminate

    controller.record_tool_result(
        "str_replace_editor",
        {"command": "create", "path": "/workspace/result.txt"},
        "File created successfully at: /workspace/result.txt",
    )

    assert controller.completion_contract.requires_explicit_terminate
    assert "artifact" in controller.completion_contract.required_evidence_kinds


def test_negated_artifact_intent_does_not_require_file_creation():
    controller = TaskController()
    controller.initialize("请查看 README.md 的标题。不要修改文件。")

    assert "retrieval" in controller.completion_contract.required_evidence_kinds
    assert "artifact" not in controller.completion_contract.required_evidence_kinds


def test_browser_action_requires_later_verification_receipt():
    controller = TaskController()
    controller.initialize("请在网页上填写并提交表单")
    controller.record_tool_result(
        "mcp_playwright_browser_click",
        {"element": "提交", "target": "f1e20"},
        "Button clicked",
    )

    allowed, reason = controller.validate_termination(
        "success", evidence_ids=["E1"], explicit=True
    )
    assert not allowed
    assert "Missing evidence kinds: verification" in reason

    controller.record_tool_result(
        "mcp_playwright_browser_snapshot",
        {},
        "Page URL: https://example.com/success\n表单提交成功",
    )
    allowed, reason = controller.validate_termination(
        "success", evidence_ids=["E1", "E2"], explicit=True
    )
    assert allowed
    assert reason == ""


def test_artifact_task_requires_artifact_receipt():
    controller = TaskController()
    controller.initialize("请创建一个 Word 文档")
    controller.record_tool_result(
        "create_word_document",
        {"output_path": "report.docx"},
        "Document created successfully at /workspace/report.docx",
    )

    allowed, reason = controller.validate_termination(
        "success", evidence_ids=["E1"], explicit=True
    )
    assert allowed
    assert reason == ""


def test_non_answer_task_cannot_finish_with_plain_text_only():
    controller = TaskController()
    controller.initialize("请查询今天的天气")

    allowed, reason = controller.validate_termination(
        "success", evidence_ids=[], explicit=False
    )

    assert not allowed
    assert "explicit terminate" in reason


def test_invalid_termination_status_is_rejected():
    controller = TaskController()
    controller.initialize("请回答一个问题")

    allowed, reason = controller.validate_termination(
        "done", evidence_ids=[], explicit=True
    )

    assert not allowed
    assert "Invalid termination status" in reason


def test_external_task_cannot_fail_before_any_attempt():
    controller = TaskController()
    controller.initialize("请发送一封邮件")

    allowed, reason = controller.validate_termination(
        "failure",
        evidence_ids=[],
        explicit=True,
        reason="I cannot do it.",
    )

    assert not allowed
    assert "premature" in reason

    controller.record_tool_result(
        "mcp_gmail_send_email",
        {"to": "missing@example.com"},
        "Error: authentication failed",
    )
    allowed, reason = controller.validate_termination(
        "failure",
        evidence_ids=[],
        explicit=True,
        reason="Gmail authentication failed.",
    )
    assert allowed
    assert reason == ""


class CountingTool(BaseTool):
    name: str = "counting_tool"
    description: str = "Count executions."
    parameters: dict = {"type": "object", "properties": {}}
    calls: int = 0

    async def execute(self) -> str:
        self.calls += 1
        return "counted"


@pytest.mark.asyncio
async def test_accepted_terminate_stops_remaining_batched_tools():
    counter = CountingTool()
    agent = ToolCallAgent(
        available_tools=ToolCollection(Terminate(), counter),
        tool_calls=[
            ToolCall(
                id="terminate-1",
                function=Function(
                    name="terminate",
                    arguments=json.dumps(
                        {
                            "status": "success",
                            "final_answer": "done",
                            "evidence_ids": [],
                        }
                    ),
                ),
            ),
            ToolCall(
                id="counter-1",
                function=Function(name="counting_tool", arguments="{}"),
            ),
        ],
    )
    agent.task_controller.initialize("请回答一个问题")

    await agent.act()

    assert agent.state == AgentState.FINISHED
    assert counter.calls == 0
