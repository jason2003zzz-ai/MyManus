import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional

from app.agent.data_analysis import DataAnalysis
from app.llm import LLM
from app.logger import logger
from app.schema import Message
from app.team.models import TeamPlan, TeamRole, TeamTask, TeamTaskResult
from app.team.worker import ScopedManus


EventSink = Callable[[str, Dict[str, Any]], None]
WorkerFactory = Callable[[TeamRole, TeamTask], Awaitable[Any]]
CancelCheck = Callable[[], bool]

MAX_TEAM_TASKS = 6
MAX_SHARED_CONTEXT_CHARS = 30000
MAX_SYNTHESIS_RESULT_CHARS = 6000


@dataclass
class TeamOutcome:
    answer: str
    trace: str
    success: bool
    snapshot: dict[str, Any]


class TeamCoordinator:
    """Bounded delegation with isolated workers and a shared result board."""

    def __init__(
        self,
        *,
        llm: Optional[LLM] = None,
        worker_factory: Optional[WorkerFactory] = None,
        event_sink: Optional[EventSink] = None,
        worker_max_steps: int = 32,
        worker_step_limits: Optional[Dict[TeamRole, int]] = None,
        cancel_requested: Optional[CancelCheck] = None,
    ):
        self.llm = llm or LLM()
        self.worker_factory = worker_factory or self._default_worker_factory
        self.event_sink = event_sink
        self.worker_max_steps = worker_max_steps
        self.worker_step_limits = worker_step_limits or {}
        self.cancel_requested = cancel_requested
        self.plan: Optional[TeamPlan] = None
        self.results: Dict[str, TeamTaskResult] = {}

    async def execute(self, objective: str, shared_context: str = "") -> TeamOutcome:
        self.plan = await self.create_plan(objective, shared_context)
        self._emit(
            "team_plan",
            summary=self.plan.summary,
            tasks=[task.model_dump() for task in self.plan.tasks],
        )

        for task in self.plan.tasks:
            failed_dependencies = [
                dependency
                for dependency in task.depends_on
                if dependency not in self.results
                or self.results[dependency].status not in {"completed", "partial"}
            ]
            if failed_dependencies:
                now = self._now()
                result = TeamTaskResult(
                    task_id=task.id,
                    title=task.title,
                    role=task.role,
                    status="blocked",
                    error=(
                        "Blocked by incomplete dependencies: "
                        + ", ".join(failed_dependencies)
                    ),
                    started_at=now,
                    finished_at=now,
                )
                self.results[task.id] = result
                self._emit_task(result)
                continue

            await self._execute_task(task, objective, shared_context)

        answer = await self.synthesize(objective)
        success = bool(self.results) and all(
            result.status == "completed" for result in self.results.values()
        )
        snapshot = self.snapshot()
        self._emit(
            "team_summary",
            success=success,
            completed=sum(
                result.status == "completed" for result in self.results.values()
            ),
            partial=sum(result.status == "partial" for result in self.results.values()),
            total=len(self.results),
        )
        return TeamOutcome(
            answer=answer,
            trace=self.trace_text(),
            success=success,
            snapshot=snapshot,
        )

    async def create_plan(self, objective: str, shared_context: str) -> TeamPlan:
        system = Message.system_message(
            "You coordinate a small software agent team. Decompose only when useful; "
            "simple requests may have one task. Return strict JSON with this shape: "
            '{"summary":"...", "tasks":[{"id":"t1","title":"...",'
            '"role":"general|browser|data","objective":"...",'
            '"deliverable":"...","depends_on":[]}]}.'
            "Use at most 6 tasks. Use browser for web search, Playwright navigation, "
            "page reading, or interaction; data for Python/statistics/tables/charts; "
            "and general for local files, coding, email services, documents, and "
            "implementation work. Preserve every scope qualifier from the user, such "
            "as year, cohort, degree type, entity, location, and output format, in each "
            "relevant task objective. Keep one coherent research deliverable with one "
            "owner: do not split discovery and verification into multiple browser tasks "
            "when one authoritative source directly states the requested facts, and do "
            "not create per-entity searches unless the user explicitly requests "
            "independent corroboration or the primary source is incomplete. "
            "For research tasks, the browser work package must open or fetch the strongest "
            "first-party source; search snippets alone are not a completed deliverable, "
            "especially for any claim that information was not found. A file-"
            "generation task should depend directly on the task that produces its "
            "structured source data. Do not reinterpret or expand a user-provided name "
            "or abbreviation without evidence. Never assign URL access or web "
            "verification to data or general roles. Use data only for substantive "
            "calculation, statistics, transformation, or visualization; do not insert a "
            "data task merely to reformat a browser handoff that the general document "
            "worker can consume directly. Dependencies "
            "may reference only earlier task ids. Do not include a final synthesis task; "
            "the coordinator performs final synthesis."
        )
        user = Message.user_message(
            f"USER OBJECTIVE:\n{objective}\n\n"
            "AVAILABLE SHARED CONTEXT:\n"
            + self._compact(shared_context, MAX_SHARED_CONTEXT_CHARS)
        )
        try:
            response = await self.llm.ask(
                messages=[user],
                system_msgs=[system],
                stream=False,
            )
            return self._normalize_plan(self._parse_json_object(response))
        except Exception as exc:
            logger.warning(f"Team planning fell back to deterministic routing: {exc}")
            return self._fallback_plan(objective)

    async def _execute_task(
        self,
        task: TeamTask,
        objective: str,
        shared_context: str,
    ) -> None:
        started_at = self._now()
        self._emit(
            "team_task",
            task_id=task.id,
            title=task.title,
            role=task.role.value,
            status="running",
            content=task.objective,
        )
        worker = None
        raw_result = ""
        try:

            async def create_and_run_worker():
                # MCP transports use AnyIO cancel scopes, which must be entered and
                # exited by the same asyncio task. Keep worker construction, execution,
                # and ToolCallAgent.run's final cleanup inside one task boundary.
                task_worker = await self.worker_factory(task.role, task)
                task_worker.max_steps = self.worker_step_limits.get(
                    task.role, self.worker_max_steps
                )
                prompt = self._worker_prompt(task, objective, shared_context)
                task_result = await task_worker.run(
                    prompt, task_objective=task.objective
                )
                return task_worker, task_result

            worker_task = asyncio.create_task(
                create_and_run_worker(), name=f"team-worker-{task.id}"
            )
            try:
                worker, raw_result = await worker_task
            except asyncio.CancelledError:
                if self._should_propagate_cancellation():
                    if not worker_task.done():
                        worker_task.cancel()
                    await asyncio.gather(worker_task, return_exceptions=True)
                    raise
                self._uncancel_current_task()
                raise RuntimeError(
                    f"{task.role.value} worker was internally cancelled"
                ) from None
            await self._drain_internal_cancellation()
            answer = self._worker_answer(worker, raw_result)
            hit_step_limit = "Terminated: Reached max steps" in raw_result
            failed = getattr(worker, "finish_status", None) == "failure"
            valid_handoff = self._valid_evidence_backed_handoff(worker, answer)
            if failed:
                status = "failed"
            elif hit_step_limit and valid_handoff:
                status = "partial"
            elif hit_step_limit:
                status = "failed"
            elif getattr(worker, "finish_status", None) == "success":
                status = "completed"
            elif valid_handoff:
                status = "partial"
            else:
                status = "failed"
            error = None
            if hit_step_limit and status == "partial":
                error = (
                    f"{task.role.value} worker reached its step limit after producing "
                    "an evidence-backed handoff"
                )
            elif hit_step_limit:
                error = f"{task.role.value} worker reached its step limit"
            elif failed:
                error = (
                    getattr(worker, "finish_reason", None) or "Worker reported failure"
                )
            elif status == "failed":
                error = "Worker ended without a valid completion handoff"

            result = TeamTaskResult(
                task_id=task.id,
                title=task.title,
                role=task.role,
                status=status,
                answer=answer,
                raw_result=raw_result,
                error=error,
                started_at=started_at,
                finished_at=self._now(),
            )
        except Exception as exc:
            logger.exception(f"Team task {task.id} failed: {exc}")
            result = TeamTaskResult(
                task_id=task.id,
                title=task.title,
                role=task.role,
                status="failed",
                raw_result=raw_result,
                error=str(exc),
                started_at=started_at,
                finished_at=self._now(),
            )
        self.results[task.id] = result
        self._emit_task(result)

    async def synthesize(self, objective: str) -> str:
        result_payload = [
            {
                "task_id": result.task_id,
                "role": result.role.value,
                "status": result.status,
                "answer": self._compact(
                    result.answer or result.raw_result,
                    MAX_SYNTHESIS_RESULT_CHARS,
                ),
                "error": result.error,
            }
            for result in self.results.values()
        ]
        system = Message.system_message(
            "You are the lead coordinator. Answer the user from the team result board. "
            "Combine compatible findings, retain concrete URLs/numbers/artifact paths, "
            "strictly preserve the user's requested scope, and omit tangential cohorts, "
            "entities, or time periods even if an intermediate worker mentioned them. "
            "When intermediate results disagree, prefer authoritative evidence from the "
            "work package that most directly matches the requested scope. "
            "state any blocked or failed work honestly, and omit internal prompts and "
            "raw tool chatter. Do not claim work that no worker completed."
        )
        user = Message.user_message(
            f"ORIGINAL OBJECTIVE:\n{objective}\n\nTEAM RESULT BOARD:\n"
            + json.dumps(result_payload, ensure_ascii=False, indent=2)
        )
        try:
            return await self.llm.ask(
                messages=[user],
                system_msgs=[system],
                stream=False,
            )
        except Exception as exc:
            logger.error(f"Team synthesis failed: {exc}")
            return self._fallback_summary(objective)

    async def _default_worker_factory(self, role: TeamRole, task: TeamTask):
        if role == TeamRole.DATA:
            return DataAnalysis()
        return await ScopedManus.create_for_role(role.value)

    def _worker_prompt(
        self,
        task: TeamTask,
        objective: str,
        shared_context: str,
    ) -> str:
        dependencies = []
        for dependency in task.depends_on:
            result = self.results.get(dependency)
            if result:
                dependencies.append(
                    f"{dependency} ({result.role.value}):\n{result.handoff_text()}"
                )
        return (
            f"TEAM ROLE: {task.role.value}\n"
            f"ORIGINAL USER OBJECTIVE:\n{objective}\n\n"
            f"YOUR WORK PACKAGE ({task.id} - {task.title}):\n{task.objective}\n\n"
            f"EXPECTED DELIVERABLE:\n{task.deliverable or 'A concise, reusable result'}\n\n"
            "DEPENDENCY HANDOFFS:\n"
            + ("\n\n".join(dependencies) if dependencies else "(none)")
            + "\n\nSHARED USER CONTEXT:\n"
            + self._compact(shared_context, MAX_SHARED_CONTEXT_CHARS)
            + "\n\nComplete only this work package. End with a concise handoff for "
            "the coordinator. Preserve exact user-provided text and constraints."
        )

    def _normalize_plan(self, payload: dict[str, Any]) -> TeamPlan:
        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError("Planner returned no tasks")

        tasks = []
        seen = set()
        for index, item in enumerate(raw_tasks[:MAX_TEAM_TASKS], start=1):
            if not isinstance(item, dict):
                continue
            task_id = self._safe_task_id(item.get("id"), index)
            if task_id in seen:
                base_task_id = f"t{index}"
                task_id = base_task_id
                suffix = 2
                while task_id in seen:
                    task_id = f"{base_task_id}_{suffix}"
                    suffix += 1
            role_value = str(item.get("role") or TeamRole.GENERAL.value).lower()
            try:
                role = TeamRole(role_value)
            except ValueError:
                role = TeamRole.GENERAL
            raw_dependencies = item.get("depends_on") or []
            if not isinstance(raw_dependencies, list):
                raw_dependencies = []
            dependencies = [
                str(value) for value in raw_dependencies if str(value) in seen
            ]
            objective = str(item.get("objective") or item.get("title") or "").strip()
            if not objective:
                continue
            tasks.append(
                TeamTask(
                    id=task_id,
                    title=str(item.get("title") or objective[:60]).strip(),
                    role=role,
                    objective=objective,
                    deliverable=str(item.get("deliverable") or "").strip(),
                    depends_on=dependencies,
                )
            )
            seen.add(task_id)
        if not tasks:
            raise ValueError("Planner returned no valid tasks")
        return TeamPlan(
            summary=str(payload.get("summary") or "Coordinated execution plan"),
            tasks=tasks,
        )

    def _fallback_plan(self, objective: str) -> TeamPlan:
        lowered = objective.lower()
        browser_markers = (
            "browser",
            "website",
            "web page",
            "click",
            "网页",
            "浏览器",
            "网站",
            "页面",
        )
        data_markers = (
            "data",
            "statistics",
            "chart",
            "spreadsheet",
            "数据",
            "统计",
            "图表",
            "表格",
        )
        if any(marker in lowered for marker in browser_markers):
            role = TeamRole.BROWSER
        elif any(marker in lowered for marker in data_markers):
            role = TeamRole.DATA
        else:
            role = TeamRole.GENERAL
        return TeamPlan(
            summary="Fallback single work package",
            tasks=[
                TeamTask(
                    id="t1",
                    title="Complete the requested work",
                    role=role,
                    objective=objective,
                    deliverable="A verified result for the user",
                )
            ],
        )

    @staticmethod
    def _parse_json_object(value: str) -> dict[str, Any]:
        text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", value, flags=re.I)
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Planner response did not contain a JSON object")
        payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("Planner JSON root must be an object")
        return payload

    @staticmethod
    def _safe_task_id(value: Any, index: int) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "")).strip("_")
        return normalized[:32] or f"t{index}"

    @staticmethod
    def _worker_answer(worker: Any, raw_result: str) -> str:
        final_answer = getattr(worker, "final_answer", None)
        if final_answer:
            return str(final_answer).strip()
        memory = getattr(worker, "memory", None)
        for message in reversed(getattr(memory, "messages", [])):
            for tool_call in getattr(message, "tool_calls", None) or []:
                if getattr(tool_call.function, "name", "") != "terminate":
                    continue
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                candidate = arguments.get("final_answer")
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            if (
                getattr(message.role, "value", message.role) == "assistant"
                and message.content
                and not message.tool_calls
            ):
                return message.content.strip()
        return TeamCoordinator._compact(raw_result, MAX_SYNTHESIS_RESULT_CHARS)

    @staticmethod
    def _valid_evidence_backed_handoff(worker: Any, answer: str) -> bool:
        if len((answer or "").strip()) < 80:
            return False
        controller = getattr(worker, "task_controller", None)
        if controller is None:
            return False
        evidence_ids = [
            receipt.receipt_id
            for receipt in getattr(controller, "evidence_receipts", [])
        ]
        try:
            allowed, _ = controller.validate_termination(
                "success",
                evidence_ids=evidence_ids,
                explicit=True,
                final_answer=answer,
            )
        except Exception:
            return False
        return allowed

    def _should_propagate_cancellation(self) -> bool:
        if self.cancel_requested is not None:
            return bool(self.cancel_requested())
        task = asyncio.current_task()
        return bool(task and task.cancelling())

    @staticmethod
    def _uncancel_current_task() -> None:
        task = asyncio.current_task()
        if task is None or not hasattr(task, "uncancel"):
            return
        while task.cancelling():
            task.uncancel()

    async def _drain_internal_cancellation(self) -> None:
        if self.cancel_requested is None:
            return
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            if self.cancel_requested():
                raise
            self._uncancel_current_task()
            logger.warning("Discarded leaked MCP worker cancellation")
        else:
            if not self.cancel_requested():
                self._uncancel_current_task()

    def _emit_task(self, result: TeamTaskResult) -> None:
        self._emit(
            "team_task",
            task_id=result.task_id,
            title=result.title,
            role=result.role.value,
            status=result.status,
            content=result.answer or result.error or "",
        )

    def _emit(self, event_type: str, **payload: Any) -> None:
        if self.event_sink:
            self.event_sink(event_type, payload)

    def snapshot(self) -> dict[str, Any]:
        return {
            "plan": self.plan.model_dump() if self.plan else None,
            "results": [
                self.results[task.id].model_dump()
                for task in self.plan.tasks
                if task.id in self.results
            ]
            if self.plan
            else [],
        }

    def trace_text(self) -> str:
        parts = []
        if self.plan:
            parts.append(
                "Team plan:\n"
                + json.dumps(self.plan.model_dump(), ensure_ascii=False, indent=2)
            )
        parts.append(
            "Team results:\n"
            + json.dumps(
                [result.model_dump() for result in self.results.values()],
                ensure_ascii=False,
                indent=2,
            )
        )
        return "\n\n".join(parts)

    def _fallback_summary(self, objective: str) -> str:
        lines = [f"多智能体任务已结束：{objective}"]
        for result in self.results.values():
            lines.append(
                f"\n- {result.title}（{result.role.value} / {result.status}）\n"
                f"{result.answer or result.error or result.raw_result}"
            )
        return "\n".join(lines)

    @staticmethod
    def _compact(value: str, limit: int) -> str:
        text = str(value or "")
        if len(text) <= limit:
            return text
        head = int(limit * 0.65)
        return text[:head] + "\n...[context truncated]...\n" + text[-(limit - head) :]

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
