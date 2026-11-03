import json

import pytest

from praxis_agent.context.manager import ContextManager
from praxis_agent.context.models import ConversationHistory
from praxis_agent.context.operations import ContextOperation
from praxis_agent.context.retrieval import read_history, search_history
from praxis_agent.context.tools import create_context_tools
from praxis_agent.tools.registry import ToolContext


@pytest.mark.parametrize("text,start", [("", 0), ("abc", 0), ("中文😀\r\nxyz", 2), ("abc", 3)])
def test_read_history_clamps_end_and_reports_full_message_length(text, start):
    history = ConversationHistory([{"role": "assistant", "content": text}])
    result = read_history(history, [], "m1", start, len(text) + 10000)
    assert result == {"text": text[start:], "total_chars": len(text),
                      "truncated": False, "next_offset": None, "note": ""}


def test_read_history_reports_total_length_for_a_partial_range():
    history = ConversationHistory([{"role": "assistant", "content": "abcdef"}])
    result = read_history(history, [], "m1", 1, 3)
    assert result == {"text": "bc", "total_chars": 6,
                      "truncated": False, "next_offset": None, "note": ""}


def test_read_history_paginates_a_clamped_range_within_the_complete_output_budget():
    text = "中文😀\\\"\t\r\n" * 120
    history = ConversationHistory([{"role": "assistant", "content": text}])
    start, chunks = 2, []
    for _ in range(100):
        result = read_history(history, [], "m1", start, len(text) + 10000, character_limit=320)
        assert len(json.dumps(result, ensure_ascii=False)) <= 320
        assert result["total_chars"] == len(text)
        chunks.append(result["text"])
        if not result["truncated"]:
            assert result["next_offset"] is None
            break
        assert result["text"]
        assert result["next_offset"] == start + len(result["text"])
        start = result["next_offset"]
    else:
        pytest.fail("History read did not reach the end of the message")
    assert "".join(chunks) == text[2:]


@pytest.mark.parametrize("start,end", [(4, 10000), (2, 1), (2, True)])
def test_read_history_clamping_still_rejects_invalid_positions(start, end):
    history = ConversationHistory([{"role": "assistant", "content": "abc"}])
    with pytest.raises(ValueError):
        read_history(history, [], "m1", start, end)


@pytest.mark.parametrize("programmatic", [False, True])
def test_read_history_tool_returns_length_and_clamps_direct_and_programmatic_calls(tmp_path, programmatic):
    text = "historical evidence " * 100
    history = ConversationHistory([{"role": "assistant", "content": text}])
    context = ToolContext(tmp_path, None, history=history, context_manager=ContextManager(),
                          programmatic_call=programmatic)
    tool = next(tool for tool in create_context_tools(max_history_read_chars=320) if tool.name == "read_history")
    result = tool.handler(context, "m1", end_offset=10000)
    assert result.success is True
    assert result.structured_output == json.loads(result.output)
    assert result.structured_output["total_chars"] == len(text)
    if programmatic:
        assert result.structured_output["text"] == text
        assert result.structured_output["truncated"] is False
    else:
        assert len(result.output) <= 320
        assert result.structured_output["truncated"] is True


def test_search_does_not_treat_its_own_query_as_evidence():
    history = ConversationHistory([
        {"role": "user", "content": "Review the supplied archive."},
        {"type": "function_call", "call_id": "search", "name": "search_history",
         "arguments": json.dumps({"query": "missing-archive-7319", "max_results": 10})},
    ])
    assert search_history(history, [], "missing-archive-7319") == {"matches": [], "truncated": False}
    assert "missing-archive-7319" in read_history(history, [], "m2")["text"]


def test_repeated_searches_and_history_reads_keep_original_evidence_discoverable():
    source = "Verified record-8271: value is 42."
    history = ConversationHistory([
        {"type": "function_call", "call_id": "source", "name": "read_record", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "source", "output": source},
    ])
    for index in range(12):
        call_id = f"search-{index}"
        history.append({"type": "function_call", "call_id": call_id, "name": "search_history",
                        "arguments": json.dumps({"query": "record-8271", "max_results": 10})})
        found = search_history(history, [], "record-8271")
        assert found == {"matches": [{"message_id": "m2", "offset": 0, "text": source}], "truncated": False}
        history.append({"type": "function_call_output", "call_id": call_id,
                        "output": json.dumps({"success": True, "output": json.dumps(found)})})
        read_id = f"read-{index}"
        history.extend([
            {"type": "function_call", "call_id": read_id, "name": "read_history",
             "arguments": '{"message_id":"m2"}'},
            {"type": "function_call_output", "call_id": read_id,
             "output": json.dumps(read_history(history, [], "m2"))},
        ])
    assert source in read_history(history, [], f"m{len(history)}")["text"]


def test_search_limits_distinct_sources_and_preserves_readable_offsets():
    source = "中文😀" * 40 + "needle is verified. " * 20
    history = ConversationHistory([
        {"role": "assistant", "content": "Earlier needle observation."},
        {"role": "assistant", "content": source},
    ])
    operation = ContextOperation("op1", "compress", ("m1", "m2"), replacement={
        "message_id": "summary1", "payload": {"role": "assistant", "content": "needle summary " * 20},
    })
    result = search_history(history, [operation], "NEEDLE", max_results=2)
    assert [match["message_id"] for match in result["matches"]] == ["summary1", "m2"]
    assert result["truncated"] is True
    result = search_history(history, [operation], "needle", max_results=3)
    assert [match["message_id"] for match in result["matches"]] == ["summary1", "m2", "m1"]
    assert result["truncated"] is False
    for match in result["matches"]:
        assert read_history(history, [operation], match["message_id"], match["offset"],
                            match["offset"] + len(match["text"]))["text"] == match["text"]


def test_programmatic_history_queries_do_not_search_their_own_code_or_echoes():
    source = "Verified needle amount is 42."
    history = ConversationHistory([{"role": "assistant", "content": source}])
    for index in range(12):
        call_id = f"code-{index}"
        history.append({"type": "function_call", "call_id": call_id, "name": "run_code",
                        "arguments": json.dumps({"code": 'call_tool("search_history", {"query":"needle"})'})})
        found = search_history(history, [], "needle")
        assert found == {"matches": [{"message_id": "m1", "offset": 0, "text": source}], "truncated": False}
        history.append({"type": "function_call_output", "call_id": call_id,
                        "output": json.dumps({"success": True, "output": json.dumps({
                            "result": found, "calls": [{"tool": "search_history", "success": True}],
                        })})})
    assert "needle" in read_history(history, [], "m2")["text"]
    assert "needle" in read_history(history, [], "m3")["text"]


@pytest.mark.parametrize("nested", [[], [{"tool": "read_file", "success": True}],
                                  [{"tool": "search_history", "success": True},
                                   {"tool": "read_file", "success": True}]])
def test_search_keeps_code_results_with_new_evidence(nested):
    history = ConversationHistory([
        {"type": "function_call", "call_id": "code", "name": "run_code", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "code", "output": json.dumps({
            "success": True, "output": json.dumps({"result": "verified needle evidence", "calls": nested}),
        })},
    ])
    assert [m["message_id"] for m in search_history(history, [], "needle")["matches"]] == ["m2"]
