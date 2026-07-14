from app.agent.manus import Manus


def test_snapshot_target_text_is_reduced_to_pure_ref():
    args = Manus._normalize_snapshot_targets(
        "mcp_playwright_browser_click",
        {"target": 'button "提交评测" [ref=f4e291]', "element": "提交评测"},
    )

    assert args["target"] == "f4e291"


def test_multiline_keyboard_entry_into_code_editor_is_blocked():
    agent = Manus()
    rejection = agent._guard_playwright_command(
        "mcp_playwright_browser_run_code_unsafe",
        {"code": "const code = `a\\nb`; await editor.type(code);"},
    )

    assert rejection
    assert "CodeMirror.setValue" in rejection


def test_virtual_snapshot_ref_is_not_allowed_as_dom_selector():
    agent = Manus()
    rejection = agent._guard_playwright_command(
        "mcp_playwright_browser_run_code_unsafe",
        {"code": "await page.locator('[ref=\"f4e274\"]').click();"},
    )

    assert rejection
    assert "virtual identifiers" in rejection
