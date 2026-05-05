import json

from sandbox.result_formatter import format_tool_result


def make_response(data):
    return {
        "code": 0,
        "message": "success",
        "data": data,
        "meta": {
            "tool": "mcp:canvas.canvas_list_courses",
            "resource_type": "mcp",
            "execution_time_ms": 12.3,
            "session_id": "session-1",
            "trace_id": "trace-1",
        },
    }


def test_format_tool_result_formats_mcp_text_content():
    response = make_response(
        {
            "content": [
                {"type": "text", "text": "course list"},
            ]
        }
    )

    assert format_tool_result(response) == "course list"


def test_format_tool_result_formats_mcp_structured_content_string():
    response = make_response(
        {
            "structuredContent": {
                "content": "structured course list",
            }
        }
    )

    assert format_tool_result(response) == "structured course list"


def test_format_tool_result_formats_mcp_error_content():
    response = make_response(
        {
            "isError": True,
            "content": [
                {"type": "text", "text": "Error: failed to list courses"},
            ],
        }
    )

    assert format_tool_result(response) == "[Error] Error: failed to list courses"


def test_format_tool_result_formats_mcp_error_text_block_without_top_level_error():
    response = make_response(
        {
            "content": [
                {"type": "text", "text": "permission denied", "error": True},
            ],
        }
    )

    assert format_tool_result(response) == "[Error] permission denied"


def test_format_tool_result_formats_mcp_json_fallback():
    response = make_response(
        {
            "structuredContent": {
                "courses": [{"id": 1, "name": "Math"}],
            }
        }
    )

    assert format_tool_result(response) == json.dumps(
        {"courses": [{"id": 1, "name": "Math"}]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def test_format_tool_result_formats_mcp_json_fallback_with_empty_structured_content():
    response = make_response(
        {
            "content": [
                {"type": "image", "mimeType": "image/png", "data": "AAA"},
            ],
            "structuredContent": {},
        }
    )

    assert format_tool_result(response) == json.dumps(
        {
            "content": [
                {"type": "image", "mimeType": "image/png", "data": "AAA"},
            ],
            "structuredContent": {},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
