import pytest

from app.tool.python_execute import PythonExecute


@pytest.mark.asyncio
async def test_python_execute_preserves_output_before_system_exit():
    result = await PythonExecute().execute(
        "print('compile failed')\nraise SystemExit(1)"
    )

    assert result["success"] is False
    assert "compile failed" in result["observation"]
    assert "SystemExit: 1" in result["observation"]


@pytest.mark.asyncio
async def test_python_execute_names_empty_assertion_errors():
    result = await PythonExecute().execute("assert False")

    assert result == {"observation": "AssertionError", "success": False}
