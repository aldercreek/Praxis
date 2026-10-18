import json
from copy import deepcopy

import pytest

from praxis_agent.context.capacity import ContextPolicy
from praxis_agent.context.operations import ContextOperations
from praxis_agent.context.tool_output import preview_tool_outputs
from praxis_agent.planning import RunState
from praxis_agent.tools import ToolContext, create_default_registry


@pytest.mark.parametrize("wrapped", [False, True])
def test_long_previews_keep_the_start_and_final_error_without_mutating_history(wrapped):
    body = "begin\n" + "middle output\n" * 100 + "fatal: build failed\n"
    output = json.dumps({"success": False, "output": body}) if wrapped else body
    messages = ContextOperations().project([
        {"type": "function_call_output", "call_id": "command", "output": output},
    ])
    original = deepcopy(messages)

    preview = preview_tool_outputs(messages, 100)[0]

    assert messages == original
    assert preview["message_id"] == original[0]["message_id"]
    assert preview["payload"]["call_id"] == "command"
    text = preview["payload"]["output"]
    assert body[:50] in text
    assert text.endswith(body[-50:])
    assert "fatal: build failed" in text
    assert f"{len(body) - 100} characters omitted" in text
    assert "saved as m1." in text
    if wrapped:
        assert "success=false;" in text


@pytest.mark.parametrize("limit,head,tail", [(1, "首", ""), (5, "首部中", "尾部"), (6, "首部中", "文尾部")])
def test_preview_budgets_count_decoded_characters_and_handle_tiny_limits(limit, head, tail):
    body = "首部中文中文中文尾部"
    messages = [{"message_id": "m3", "payload": {
        "type": "function_call_output", "call_id": "c",
        "output": json.dumps({"success": True, "output": body}, ensure_ascii=True),
    }}]

    output = preview_tool_outputs(messages, limit)[0]["payload"]["output"]
    excerpt = output.split("]\n\n", 1)[1]

    assert excerpt == head + f"\n\n[... {len(body) - limit} characters omitted ...]\n\n" + tail


def test_complete_outputs_and_non_tool_messages_are_unchanged():
    messages = [
        {"message_id": "m1", "payload": {"type": "function_call_output", "call_id": "c",
                                         "output": json.dumps({"success": True, "output": "你好\n"})}},
        {"message_id": "m2", "payload": {"type": "function_call_output", "call_id": "d", "output": "abc"}},
        {"message_id": "m3", "payload": {"role": "assistant", "content": "long text" * 100}},
    ]
    assert preview_tool_outputs(messages, 3) == messages


def test_file_continuation_metadata_survives_tool_preview(tmp_path):
    text = "中文\\\"\t\r\n" * 120
    (tmp_path / "notes.txt").write_bytes(text.encode("utf-8"))
    registry = create_default_registry(ContextPolicy(max_file_read_chars=420, max_tool_output_chars=420))
    context = ToolContext(tmp_path, RunState("read file"))
    result = registry.execute("read_file", {"relative_path": "notes.txt", "end_line": 1000}, context=context)
    assert result.success, result.output
    messages = [{"message_id": "m1", "payload": {
        "type": "function_call_output", "call_id": "c",
        "output": json.dumps({"success": result.success, "output": result.output}),
    }}]

    preview = preview_tool_outputs(messages, 420)

    assert preview == messages
    data = json.loads(json.loads(preview[0]["payload"]["output"])["output"])
    assert data["truncated"] is True
    assert data["next_position"]
    assert "next_position" in data["note"]
    assert text.startswith(data["text"])


@pytest.mark.parametrize("limit", [1, 100, 400])
def test_command_preview_preserves_metadata_and_both_stream_endings(limit):
    result = {"status": "exited", "exit_code": 7,
              "stdout": "BEGIN\n" + "ordinary log\n" * 100 + "FINAL_RESULT=42",
              "stderr": "WARN\n" + "stack frame\n" * 100 + "fatal: invalid value",
              "error": "命令已退出，退出码为 7。请结合输出判断原因。"}
    messages = [{"message_id": "m7", "payload": {"type": "function_call_output", "call_id": "c",
                 "output": json.dumps({"success": True, "output": result}, ensure_ascii=False)}}]
    original = deepcopy(messages)

    output = preview_tool_outputs(messages, limit)[0]["payload"]["output"]
    data = json.loads(output.split("\n\n", 1)[1])["output"]

    assert messages == original
    assert "saved as m7." in output
    assert {key: data[key] for key in ("status", "exit_code", "error")} == {
        key: result[key] for key in ("status", "exit_code", "error")}
    if limit >= 100:
        assert data["stdout"].startswith("BEGIN\n") and data["stdout"].endswith("FINAL_RESULT=42")
        assert data["stderr"].startswith("WARN\n") and data["stderr"].endswith("fatal: invalid value")


def test_command_preview_does_not_truncate_short_logs_or_metadata():
    result = {"status": "spawn_failed", "exit_code": None, "stdout": "", "stderr": "",
              "error": "无法启动命令：找不到可执行文件。"}
    messages = [{"message_id": "m1", "payload": {"type": "function_call_output", "call_id": "c",
                 "output": json.dumps({"success": False, "output": result})}}]
    assert preview_tool_outputs(messages, 1) == messages
