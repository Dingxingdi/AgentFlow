"""
Tests for code tool schemas integration.
"""

from sandbox.tool_schemas import get_all_tool_names, get_tool_schemas, get_tools_by_resource


EXPECTED_CODE_TOOLS = {
    "code-read",
    "code-glob",
    "code-grep",
    "code-bash",
    "code-edit",
    "code-write",
}


def _code_schemas_by_name():
    schemas = get_tools_by_resource("code")
    return {schema["name"]: schema for schema in schemas}


def test_code_tools_visible_in_global_catalog():
    """Code tools should appear in the global tool name catalog."""
    names = get_all_tool_names()

    assert "code-read" in names
    assert "code-bash" in names


def test_code_wildcard_filtering():
    """Wildcard filtering should support code-* patterns."""
    schemas = get_tool_schemas(["code-*"])
    names = {schema["name"] for schema in schemas}

    assert names == EXPECTED_CODE_TOOLS


def test_get_tools_by_resource_code():
    """Resource filtering should return all code tools."""
    schemas = get_tools_by_resource("code")
    names = {schema["name"] for schema in schemas}

    assert names == EXPECTED_CODE_TOOLS


def test_code_tool_parameter_contract():
    """Each code tool should expose the exact expected parameter names."""
    expected_params = {
        "code-read": {"file_path", "offset", "limit"},
        "code-glob": {"pattern", "path"},
        "code-grep": {"pattern", "path", "glob"},
        "code-bash": {"command"},
        "code-edit": {"file_path", "old_string", "new_string", "replace_all"},
        "code-write": {"file_path", "content"},
    }
    schemas = _code_schemas_by_name()

    for tool_name, expected in expected_params.items():
        actual = {param["name"] for param in schemas[tool_name]["parameters"]}
        assert actual == expected


def test_code_read_description_mentions_line_numbered_and_1_indexed_offset():
    """code-read docs should preserve line-numbered output and 1-indexed offset semantics."""
    schema = _code_schemas_by_name()["code-read"]
    offset = next(
        param for param in schema["parameters"] if param["name"] == "offset"
    )

    assert "line" in schema["description"].lower()
    assert "number" in schema["description"].lower()
    assert "1-indexed" in offset["description"].lower()


def test_code_bash_description_mentions_backend_config_availability():
    """code-bash docs should mention backend-config-dependent availability."""
    schema = _code_schemas_by_name()["code-bash"]
    description = schema["description"].lower()

    assert "backend" in description
    assert "config" in description
    assert "depend" in description
