"""
Tests for the Code backend skeleton and bridge-tool registration.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

from sandbox.server.backends.base import BackendConfig
from sandbox.server.backends.error_codes import ErrorCode
from sandbox.server.core.tool_executor import ToolExecutor

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "server"
    / "backends"
    / "resources"
    / "code.py"
)


def load_code_backend_module():
    package_name = "sandbox.server.backends.resources"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(MODULE_PATH.parent)]
        sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.code",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeServer:
    def __init__(self):
        self._tools = {}
        self._tool_resource_types = {}

    def register_tool(self, name, func, resource_type=None):
        self._tools[name] = func
        if resource_type is not None:
            self._tool_resource_types[name] = resource_type


def build_backend_config(tmp_path):
    return BackendConfig(
        enabled=True,
        default_config={
            "claude_code_root": str(tmp_path / "claude-code-py"),
            "workspace_root": str(tmp_path / "agentflow_code"),
            "allow_bash": True,
        },
        description="Code backend",
    )


def create_fake_claude_code_root(tmp_path):
    root = tmp_path / "claude-code-py"
    tools_dir = root / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)

    (root / "tool.py").write_text(
        "class Tool:\n"
        "    async def call(self, params, ctx):\n"
        "        raise NotImplementedError\n",
        encoding="utf-8",
    )

    (tools_dir / "__init__.py").write_text(
        "raise RuntimeError('tools package import should not happen')\n",
        encoding="utf-8",
    )

    (tools_dir / "file_tools.py").write_text(
        "from tool import Tool\n"
        "\n"
        "class ReadTool(Tool):\n"
        "    async def call(self, params, ctx):\n"
        "        return {'content': [{'type': 'text', 'text': f\"read:{ctx.cwd}\"}], 'params': params}\n"
        "\n"
        "class GlobTool(Tool):\n"
        "    async def call(self, params, ctx):\n"
        "        return {'glob': True, 'cwd': ctx.cwd, 'params': params}\n"
        "\n"
        "class GrepTool(Tool):\n"
        "    async def call(self, params, ctx):\n"
        "        return {'grep': True, 'cwd': ctx.cwd, 'params': params}\n"
        "\n"
        "class BashTool(Tool):\n"
        "    async def call(self, params, ctx):\n"
        "        raise RuntimeError('bash tool should not be invoked')\n",
        encoding="utf-8",
    )

    (tools_dir / "edit_tools.py").write_text(
        "from tool import Tool\n"
        "\n"
        "class EditTool(Tool):\n"
        "    async def call(self, params, ctx):\n"
        "        return {'edit': True, 'cwd': ctx.cwd, 'params': params}\n"
        "\n"
        "class WriteTool(Tool):\n"
        "    async def call(self, params, ctx):\n"
        "        return {'write': True, 'cwd': ctx.cwd, 'params': params}\n",
        encoding="utf-8",
    )
    return root


class FakeResourceRouter:
    def __init__(self, session_info):
        self._session_info = session_info

    async def get_session(self, worker_id, resource_type):
        del worker_id, resource_type
        return self._session_info

    async def get_or_create_session(self, worker_id, resource_type, auto_created=False):
        del worker_id, resource_type, auto_created
        raise AssertionError("unexpected temporary session creation")

    async def refresh_session(self, worker_id, resource_type):
        del worker_id, resource_type
        return True

    async def destroy_session(self, worker_id, resource_type):
        del worker_id, resource_type
        return True


def test_bind_server_registers_code_tools(tmp_path):
    module = load_code_backend_module()
    backend = module.CodeBackend(config=build_backend_config(tmp_path))
    fake_server = FakeServer()

    backend.bind_server(fake_server)

    assert "code:read" in fake_server._tools
    assert "code:bash" in fake_server._tools
    assert fake_server._tool_resource_types["code:read"] == "code"
    assert fake_server._tool_resource_types["code:bash"] == "code"


def test_initialize_creates_worker_workspace(tmp_path):
    module = load_code_backend_module()
    create_fake_claude_code_root(tmp_path)
    backend = module.CodeBackend(config=build_backend_config(tmp_path))

    session = asyncio.run(backend.initialize("runner_123", {}))

    assert session["workspace"].endswith("runner_123")
    assert Path(session["workspace"]).exists()


def test_initialize_copies_source_dir(tmp_path):
    module = load_code_backend_module()
    create_fake_claude_code_root(tmp_path)
    backend = module.CodeBackend(config=build_backend_config(tmp_path))
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True)
    (source_dir / "demo.py").write_text("print('hi')\n", encoding="utf-8")

    session = asyncio.run(
        backend.initialize("runner_123", {"source_dir": str(source_dir)})
    )

    copied = Path(session["workspace"]) / "demo.py"
    assert copied.exists()
    assert copied.read_text(encoding="utf-8") == "print('hi')\n"


def test_load_claude_code_tools_uses_direct_file_loading(tmp_path):
    module = load_code_backend_module()
    create_fake_claude_code_root(tmp_path)
    backend = module.CodeBackend(config=build_backend_config(tmp_path))

    tools = backend._load_claude_code_tools()

    assert set(tools.keys()) == {"read", "glob", "grep", "bash", "edit", "write"}


def test_tool_executor_code_dispatch_returns_standard_success_response(tmp_path):
    module = load_code_backend_module()
    create_fake_claude_code_root(tmp_path)
    backend = module.CodeBackend(config=build_backend_config(tmp_path))
    fake_server = FakeServer()
    backend.bind_server(fake_server)

    executor = ToolExecutor(
        tools=fake_server._tools,
        tool_name_index={},
        tool_resource_types=fake_server._tool_resource_types,
        resource_router=FakeResourceRouter(
            {
                "session_id": "code-session-1",
                "data": {"workspace": str(tmp_path / "runtime-workspace")},
            }
        ),
    )

    result = asyncio.run(
        executor.execute(
            action="code:read",
            params={"path": "demo.py"},
            worker_id="worker-1",
            trace_id="trace-1",
        )
    )

    assert result["code"] == ErrorCode.SUCCESS
    assert result["data"]["content"][0]["text"].startswith("read:")


def test_tool_executor_blocks_bash_when_allow_bash_false(tmp_path):
    module = load_code_backend_module()
    create_fake_claude_code_root(tmp_path)
    config = build_backend_config(tmp_path)
    config.default_config["allow_bash"] = False
    backend = module.CodeBackend(config=config)
    fake_server = FakeServer()
    backend.bind_server(fake_server)

    executor = ToolExecutor(
        tools=fake_server._tools,
        tool_name_index={},
        tool_resource_types=fake_server._tool_resource_types,
        resource_router=FakeResourceRouter(
            {
                "session_id": "code-session-2",
                "data": {"workspace": str(tmp_path / "runtime-workspace")},
            }
        ),
    )

    result = asyncio.run(
        executor.execute(
            action="code:bash",
            params={"command": "echo hi"},
            worker_id="worker-1",
            trace_id="trace-1",
        )
    )

    assert result["code"] == ErrorCode.BUSINESS_FAILURE
    assert "disabled" in result["message"].lower()
