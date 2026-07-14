import asyncio
import json
from types import SimpleNamespace

import pytest

from app.agent.task_control import TaskController
from app.team import ScopedManus, TeamCoordinator, TeamRole


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)

    async def ask(self, **kwargs):
        return self.responses.pop(0)


class FakeWorker:
    def __init__(self, answer, *, finish_status="success"):
        self.answer = answer
        self.final_answer = answer
        self.finish_status = finish_status
        self.finish_reason = "worker failed" if finish_status == "failure" else None
        self.memory = SimpleNamespace(messages=[])
        self.prompts = []
        self.cleaned = False
        self.max_steps = 0

    async def run(self, prompt, *, task_objective):
        self.prompts.append((prompt, task_objective))
        self.cleaned = True
        return self.answer

    async def cleanup(self):
        self.cleaned = True


@pytest.mark.asyncio
async def test_team_coordinator_passes_dependency_handoff_and_emits_progress():
    plan = {
        "summary": "Research then analyze",
        "tasks": [
            {
                "id": "t1",
                "title": "Collect evidence",
                "role": "browser",
                "objective": "Read the source page",
                "deliverable": "Source facts",
                "depends_on": [],
            },
            {
                "id": "t2",
                "title": "Analyze evidence",
                "role": "data",
                "objective": "Calculate the result",
                "deliverable": "Checked calculation",
                "depends_on": ["t1"],
            },
        ],
    }
    workers = {
        TeamRole.BROWSER: FakeWorker("source value: 42"),
        TeamRole.DATA: FakeWorker("calculation complete"),
    }
    events = []

    async def worker_factory(role, task):
        return workers[role]

    coordinator = TeamCoordinator(
        llm=FakeLLM(json.dumps(plan), "final synthesized answer"),
        worker_factory=worker_factory,
        event_sink=lambda event_type, payload: events.append((event_type, payload)),
        worker_max_steps=11,
    )

    outcome = await coordinator.execute("Complete the compound task", "shared context")

    assert outcome.success is True
    assert outcome.answer == "final synthesized answer"
    assert "source value: 42" in workers[TeamRole.DATA].prompts[0][0]
    assert workers[TeamRole.DATA].max_steps == 11
    assert all(worker.cleaned for worker in workers.values())
    assert events[0][0] == "team_plan"
    assert events[-1] == (
        "team_summary",
        {"success": True, "completed": 2, "partial": 0, "total": 2},
    )


@pytest.mark.asyncio
async def test_worker_factory_and_run_share_one_async_task_boundary():
    plan = {
        "summary": "Task-affine resource lifecycle",
        "tasks": [
            {
                "id": "t1",
                "title": "Use task-affine resource",
                "role": "browser",
                "objective": "Create, use, and close the resource",
                "depends_on": [],
            }
        ],
    }
    lifecycle = {}

    class TaskAffineWorker(FakeWorker):
        async def run(self, prompt, *, task_objective):
            lifecycle["run_task"] = asyncio.current_task()
            assert lifecycle["factory_task"] is lifecycle["run_task"]
            return await super().run(prompt, task_objective=task_objective)

    async def worker_factory(role, task):
        lifecycle["factory_task"] = asyncio.current_task()
        return TaskAffineWorker("resource lifecycle completed")

    coordinator = TeamCoordinator(
        llm=FakeLLM(json.dumps(plan), "final answer"),
        worker_factory=worker_factory,
    )

    outcome = await coordinator.execute("Exercise a task-affine resource")

    assert outcome.success is True
    assert lifecycle["factory_task"] is lifecycle["run_task"]


@pytest.mark.asyncio
async def test_failed_dependency_blocks_downstream_worker():
    plan = {
        "summary": "Failure propagation",
        "tasks": [
            {
                "id": "t1",
                "title": "First",
                "role": "general",
                "objective": "Fail",
                "depends_on": [],
            },
            {
                "id": "t2",
                "title": "Second",
                "role": "data",
                "objective": "Should not run",
                "depends_on": ["t1"],
            },
        ],
    }
    created_roles = []

    async def worker_factory(role, task):
        created_roles.append(role)
        return FakeWorker("failed", finish_status="failure")

    coordinator = TeamCoordinator(
        llm=FakeLLM(json.dumps(plan), "partial summary"),
        worker_factory=worker_factory,
    )

    outcome = await coordinator.execute("Run dependent work")

    assert outcome.success is False
    assert created_roles == [TeamRole.GENERAL]
    assert coordinator.results["t1"].status == "failed"
    assert coordinator.results["t2"].status == "blocked"


def test_scoped_manus_enforces_role_tool_boundaries():
    general = ScopedManus(
        team_role="general",
        denied_tool_prefixes=["mcp_playwright_", "mcp_stepsearch_"],
    )
    browser = ScopedManus(
        team_role="browser",
        allowed_tool_names={"terminate", "ask_human"},
        allowed_tool_prefixes=["mcp_playwright_", "mcp_stepsearch_"],
    )

    assert general._profile_allows("python_execute")
    assert not general._profile_allows("mcp_playwright_browser_click")
    assert not general._profile_allows("mcp_stepsearch_web_search")
    assert browser._profile_allows("mcp_playwright_browser_snapshot")
    assert browser._profile_allows("mcp_stepsearch_web_search")
    assert browser._profile_allows("terminate")
    assert not browser._profile_allows("python_execute")

    browser_tools = {
        item["function"]["name"] for item in browser.get_tool_params_for_step()
    }
    assert "terminate" in browser_tools
    assert "python_execute" not in browser_tools


@pytest.mark.asyncio
async def test_evidence_backed_step_limit_handoff_unblocks_dependencies():
    plan = {
        "summary": "Research then write",
        "tasks": [
            {
                "id": "t1",
                "title": "Research",
                "role": "browser",
                "objective": "查询公开来源",
                "depends_on": [],
            },
            {
                "id": "t2",
                "title": "Write",
                "role": "general",
                "objective": "根据前序资料生成文件",
                "depends_on": ["t1"],
            },
        ],
    }
    controller = TaskController()
    controller.initialize("查询公开来源")
    controller.record_tool_result(
        "mcp_stepsearch_web_search",
        {"query": "source"},
        "Authoritative source with complete facts and URL https://example.com",
    )
    research = FakeWorker(
        "Evidence-backed handoff with complete source facts and URL "
        "https://example.com. " * 3,
        finish_status=None,
    )
    research.task_controller = controller

    async def research_run(prompt, *, task_objective):
        research.cleaned = True
        return f"{research.answer}\nTerminated: Reached max steps (8)"

    research.run = research_run
    writer = FakeWorker("file generated")

    async def worker_factory(role, task):
        return research if task.id == "t1" else writer

    coordinator = TeamCoordinator(
        llm=FakeLLM(json.dumps(plan), "partial but useful final answer"),
        worker_factory=worker_factory,
    )
    outcome = await coordinator.execute("Research and write")

    assert outcome.success is False
    assert coordinator.results["t1"].status == "partial"
    assert coordinator.results["t2"].status == "completed"
    assert "https://example.com" in writer.prompts[0][0]


@pytest.mark.asyncio
async def test_internal_worker_cancellation_does_not_cancel_coordinator():
    plan = {
        "summary": "Cancellation isolation",
        "tasks": [
            {
                "id": "t1",
                "title": "Worker",
                "role": "general",
                "objective": "Run isolated work",
                "depends_on": [],
            }
        ],
    }

    class SelfCancellingWorker(FakeWorker):
        async def run(self, prompt, *, task_objective):
            asyncio.current_task().cancel()
            await asyncio.sleep(0)

    coordinator = TeamCoordinator(
        llm=FakeLLM(json.dumps(plan), "fallback team summary"),
        worker_factory=lambda role, task: _async_value(SelfCancellingWorker("")),
    )

    outcome = await coordinator.execute("Test cancellation isolation")

    assert outcome.answer == "fallback team summary"
    assert coordinator.results["t1"].status == "failed"


@pytest.mark.asyncio
async def test_leaked_internal_cancellation_is_drained_at_worker_boundary():
    async def scenario():
        coordinator = TeamCoordinator(
            llm=FakeLLM(),
            cancel_requested=lambda: False,
        )
        asyncio.current_task().cancel("simulated MCP cleanup cancellation")
        await coordinator._drain_internal_cancellation()
        await asyncio.sleep(0)
        return asyncio.current_task().cancelling()

    task = asyncio.create_task(scenario())

    assert await task == 0


@pytest.mark.asyncio
async def test_user_requested_cancellation_propagates_at_worker_boundary():
    async def scenario():
        coordinator = TeamCoordinator(
            llm=FakeLLM(),
            cancel_requested=lambda: True,
        )
        asyncio.current_task().cancel("user requested cancellation")
        await coordinator._drain_internal_cancellation()

    task = asyncio.create_task(scenario())

    with pytest.raises(asyncio.CancelledError):
        await task


async def _async_value(value):
    return value
