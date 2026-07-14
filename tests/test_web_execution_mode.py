from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.team import TeamOutcome
from app.web import server
from app.web.skill_matcher import SkillDocument, SkillMatch


def make_run(mode="team"):
    now = datetime.now(timezone.utc).isoformat()
    return server.RunSession(
        id="test-team-run",
        prompt="Complete a compound task",
        max_steps=80,
        parent_run_id=None,
        skill_ids=[],
        auto_skill_matches=[],
        attachments=[],
        mode=mode,
        created_at=now,
        updated_at=now,
    )


def test_run_request_accepts_only_supported_modes():
    assert server.RunRequest(prompt="test").mode == "single"
    assert server.RunRequest(prompt="test", mode="team").mode == "team"
    with pytest.raises(ValidationError):
        server.RunRequest(prompt="test", mode="unknown")


def test_execution_prompt_injects_recalled_long_term_memory():
    run = make_run(mode="single")
    run.prompt = "继续完善上次的报告"
    run.memory_matches = [
        {
            "run_id": "previous-run",
            "task": "整理多智能体调研材料",
            "answer": "已生成 report.docx",
            "observations": "文件已经通过渲染检查。",
            "score": 0.73,
            "embedding_model": "fake-memory-embedding",
        }
    ]

    prompt = server.build_execution_prompt(run)

    assert "Agent Memory RAG" in prompt
    assert "previous-run" in prompt
    assert "report.docx" in prompt
    assert "继续完善上次的报告" in prompt


def test_completed_run_becomes_persistent_memory_record():
    run = make_run(mode="single")
    run.status = "completed"
    run.prompt = "生成项目报告"
    run.answer = "报告已生成"
    run.result = "Word document created at /workspace/report.docx"

    record = server.memory_record_from_run(run)

    assert record is not None
    assert record.run_id == run.id
    assert record.task == "生成项目报告"
    assert "报告已生成" in record.answer
    assert "report.docx" in record.observations


def test_unverified_negative_research_run_does_not_pollute_memory_retrieval():
    run = make_run(mode="single")
    run.status = "completed"
    run.prompt = "搜索某实验室2026届毕业去向"
    run.answer = "目前公开渠道未找到完整毕业去向。"
    run.result = (
        "Step 1: Observed output of cmd `mcp_stepsearch_web_search` executed:\n"
        '{"results": []}'
    )

    record = server.memory_record_from_run(run)

    assert record is not None
    assert record.quality == "unverified_negative"
    assert record.retrieval_eligible is False


def test_source_verified_negative_research_run_can_enter_memory():
    run = make_run(mode="single")
    run.status = "completed"
    run.prompt = "查询某机构公开名单"
    run.answer = "核验官网后，未找到该名单。"
    run.result = (
        "Step 1: Observed output of cmd `mcp_stepsearch_web_search` executed:\n{}\n"
        "Step 2: Observed output of cmd `mcp_playwright_browser_snapshot` "
        "executed:\nPage URL: https://official.example.org/archive"
    )

    record = server.memory_record_from_run(run)

    assert record is not None
    assert record.quality == "completed"
    assert record.retrieval_eligible is True


@pytest.mark.asyncio
async def test_auto_skill_retrieval_uses_tfidf_matcher(monkeypatch):
    def fake_matcher(query, documents, **kwargs):
        assert "导师来信" in query
        assert documents[0].id == "gmail"
        return [
            SkillMatch(
                id="gmail",
                name="Gmail 邮箱处理",
                summary="处理邮件",
                score=0.82,
                matched_terms=("邮件",),
            )
        ]

    monkeypatch.setattr(server, "match_skills", fake_matcher)
    monkeypatch.setattr(
        server,
        "list_skill_documents",
        lambda: [
            SkillDocument(
                id="gmail",
                name="Gmail 邮箱处理",
                summary="处理邮件",
                content="读取并回复 Gmail 邮件。",
            )
        ],
    )

    matches = await server.retrieve_auto_skill_matches("处理导师来信", [], [])

    assert matches[0]["id"] == "gmail"
    assert matches[0]["retrieval_method"] == "tfidf"
    assert matches[0]["matched_terms"] == ["邮件"]


def test_artifact_links_detect_files_created_by_generic_executor(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact = workspace / "analysis report.docx"
    artifact.write_bytes(b"generated document")
    monkeypatch.setattr("app.config.WORKSPACE_ROOT", workspace)

    answer = server.extract_artifact_answer(
        "Observed output of cmd `python_execute` executed:\n"
        f"Word document created successfully at: {artifact}\n"
    )

    assert "Word" in answer
    assert "analysis report.docx" in answer
    assert "/api/files/analysis%20report.docx" in answer


@pytest.mark.asyncio
async def test_execute_run_routes_team_mode_without_creating_single_agent(monkeypatch):
    run = make_run()

    async def fake_team_run(run_session, execution_prompt):
        run_session.team = {"results": [{"status": "completed"}]}
        run_session.answer = "team answer"
        run_session.result = "team trace"
        return TeamOutcome(
            answer="team answer",
            trace="team trace",
            success=True,
            snapshot=run_session.team,
        )

    async def fail_single_agent_create(*args, **kwargs):
        raise AssertionError("single-agent path should not be used")

    monkeypatch.setattr(server, "execute_team_run", fake_team_run)
    monkeypatch.setattr(server.Manus, "create", fail_single_agent_create)
    monkeypatch.setattr(server, "save_runs", lambda: None)
    monkeypatch.setattr(server, "remember_run", lambda run: None)
    monkeypatch.setattr(server, "list_skill_documents", lambda: [])

    await server.execute_run(run)

    assert run.status == "completed"
    assert run.answer == "team answer"
    assert any(
        event["type"] == "mode" and event["mode"] == "team" for event in run.events
    )


@pytest.mark.asyncio
async def test_partial_team_handoff_is_returned_as_completed_run(monkeypatch):
    run = make_run()

    async def fake_team_run(run_session, execution_prompt):
        run_session.team = {"results": [{"status": "partial"}]}
        run_session.answer = "evidence-backed partial answer"
        run_session.result = "partial trace"
        return TeamOutcome(
            answer=run_session.answer,
            trace=run_session.result,
            success=False,
            snapshot=run_session.team,
        )

    monkeypatch.setattr(server, "execute_team_run", fake_team_run)
    monkeypatch.setattr(server, "save_runs", lambda: None)
    monkeypatch.setattr(server, "remember_run", lambda run: None)
    monkeypatch.setattr(server, "list_skill_documents", lambda: [])

    await server.execute_run(run)

    assert run.status == "completed"
    assert run.answer == "evidence-backed partial answer"
