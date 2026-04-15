## Code Backend Design

Date: 2026-04-15
Status: Approved for planning

## Summary

Add a new `code` resource backend to AgentFlow that provides a lightweight coding environment with an isolated per-session workspace.

The backend should reuse the existing lightweight tool implementations from `claude-code-py` instead of reimplementing them inside AgentFlow. AgentFlow only adds a thin adaptation layer for:

- backend and session lifecycle
- workspace initialization
- path-boundary enforcement for file-oriented tools
- rollout and sandbox integration

The result is intentionally narrower than integrating the full `claude-code-py` runtime. It exposes only six lightweight coding tools:

- `code:read`
- `code:glob`
- `code:grep`
- `code:bash`
- `code:edit`
- `code:write`

## Goals

- Give AgentFlow a native coding environment with a per-session workspace.
- Reuse `claude-code-py` tool semantics as much as possible.
- Keep AgentFlow changes minimal, following the same integration style as the current MCP backend.
- Support rollout usage through existing `resource_types`, `available_tools`, and `resource_init_configs`.
- Isolate worker workspaces from one another.

## Non-Goals

- Do not embed the full `claude-code-py` agent runtime.
- Do not integrate `QueryEngine`, `Skill`, sub-agents, trace, or auto-compaction.
- Do not convert `claude-code-py` into an MCP server.
- Do not redesign AgentFlow's tool-loading or rollout abstractions.
- Do not provide hard OS-level sandboxing for shell execution in v1.
- Do not add Toolathlon-style `preprocess` support in v1.

## Constraints

- Maximize code reuse from `/home/a1/sdb/dxd/claude-code-py`.
- Prefer the smallest viable AgentFlow adaptation layer.
- Preserve the existing distinction between prompt-visible tool schemas and runtime-enabled resources.
- Keep `code` as a backend/resource because the environment needs session-scoped workspace state.

## Recommended Approach

Implement a new stateful backend:

- `sandbox/server/backends/resources/code.py`

Add a new static schema catalog:

- `sandbox/tool_schemas/code_tools.py`

Merge the new schemas into the global tool catalog:

- `sandbox/tool_schemas/__init__.py`

Add a sandbox config example:

- `configs/sandbox-server/code_config.json`

Add focused tests:

- `sandbox/tests/test_code_backend.py`
- `sandbox/tests/test_code_tool_schemas.py`

This approach mirrors the MCP backend structurally while remaining much simpler operationally.

## Why `code` Is a Resource Backend

`code` should not be implemented as a lightweight stateless `apis` tool bundle.

The reason is not that it must be "heavyweight" in terms of compute. The reason is that the coding environment has session-scoped state:

- each worker needs its own workspace
- multiple tool calls in one task must target the same workspace
- session init config may specify a task-specific `source_dir`
- different rollout workers must remain isolated from one another

These are backend/resource concerns, not stateless API-tool concerns.

## Reuse Strategy from `claude-code-py`

### Reused directly

Reuse the existing lightweight tool classes from `claude-code-py`:

- `tools.file_tools.BashTool`
- `tools.file_tools.ReadTool`
- `tools.file_tools.GlobTool`
- `tools.file_tools.GrepTool`
- `tools.edit_tools.EditTool`
- `tools.edit_tools.WriteTool`

The intended contract is:

- AgentFlow reuses their parameter schemas and operational semantics
- AgentFlow does not reimplement their core read/write/grep/edit behavior

### Not reused

Do not reuse:

- `QueryEngine`
- `AgentContext` as a full runtime model
- `SkillTool`
- `AgentTool`
- `SendMessageTool`
- web tools
- notebook tools
- trace and compaction logic

### Import strategy

`claude-code-py` is currently a source tree, not a packaged library. Therefore the backend should dynamically load only the required modules from `claude_code_root`.

Recommended import strategy:

- accept `claude_code_root` in backend config
- temporarily add that root to `sys.path`
- import `tools.file_tools` and `tools.edit_tools`
- instantiate only the six required tool classes

No packaging change to `claude-code-py` is required for v1.

## Architecture

### Backend object

`CodeBackend` owns:

- backend config
- dynamically loaded `claude-code-py` tool instances
- session workspace lifecycle
- bridge wrappers for `code:*` tools

### Session data

`CodeBackend.initialize(worker_id, config)` returns session data containing at least:

- `workspace`: absolute workspace path
- `source_dir`: resolved source directory if provided

No subprocess registry is needed in v1.

### Minimal context adapter

The reused `claude-code-py` tools only need a very small subset of runtime context.

For the six selected tools, a minimal context object with `cwd=<workspace>` is sufficient.

That adapter should be intentionally tiny, for example logically equivalent to:

```python
SimpleNamespace(cwd=str(workspace))
```

This preserves tool reuse without dragging the rest of the `claude-code-py` runtime into AgentFlow.

## Tool Set and Semantics

### Tool naming

Prompt-visible schema names should follow AgentFlow's current non-MCP naming style:

- `code-read`
- `code-glob`
- `code-grep`
- `code-bash`
- `code-edit`
- `code-write`

Runtime tool names should be registered as:

- `code:read`
- `code:glob`
- `code:grep`
- `code:bash`
- `code:edit`
- `code:write`

This matches current AgentFlow conventions and relies on the existing tool-name normalization logic.

### Parameter contracts

The backend should preserve `claude-code-py` parameter names:

- `code-read`: `file_path`, `offset`, `limit`
- `code-glob`: `pattern`, `path`
- `code-grep`: `pattern`, `path`, `glob`
- `code-bash`: `command`
- `code-edit`: `file_path`, `old_string`, `new_string`, `replace_all`
- `code-write`: `file_path`, `content`

### Tool behavior

V1 behavior should stay aligned with the reused implementations:

- `read` returns line-numbered text
- `glob` returns matching paths
- `grep` performs recursive regex search
- `edit` performs exact replacement with uniqueness checks
- `write` creates parent directories and overwrites full file contents
- `bash` executes a shell command in `cwd=workspace`

## Workspace Lifecycle

### Initialization

`CodeBackend.initialize()` should:

1. create `workspace_root/<worker_id>`
2. optionally copy `source_dir` contents into that workspace
3. return session data for later bridge calls

### Session init config

V1 session init config should stay minimal:

- `source_dir`: optional path to an existing code directory to copy into the workspace

No `preprocess` support is included in v1.

### Cleanup

Default behavior should leave workspace contents in place for debugging and inspection.

Unlike MCP, there are no long-lived stdio server subprocesses to terminate in v1.

## Path Boundary Enforcement

This is the main capability AgentFlow must add on top of `claude-code-py`.

### File-oriented tools

For:

- `read`
- `glob`
- `grep`
- `edit`
- `write`

the backend wrapper must:

1. resolve user-provided paths relative to the session workspace
2. normalize them to absolute paths
3. reject any path that escapes the workspace root

This enforcement belongs in AgentFlow's backend wrapper, not in `claude-code-py`.

### Why enforcement is needed

`claude-code-py` currently uses a `cwd` convention but does not provide strict workspace-boundary checks for file paths. AgentFlow must supply that missing safety layer.

## Bash Handling

### V1 behavior

`code:bash` should reuse `claude-code-py`'s `BashTool` semantics:

- input: `command`
- execution directory: `cwd=workspace`

### Limitation

This is not a hard sandbox.

Setting `cwd` to the workspace constrains the default working directory but does not stop a shell command from referencing paths outside the workspace.

### V1 protections

AgentFlow should add only minimal protections in v1:

- backend config flag `allow_bash`
- command timeout enforced by the AgentFlow wrapper

### Deferred work

Hard shell isolation is explicitly deferred to a future version. That would require a lower-level sandboxing approach such as containers or OS-level isolation.

## Configuration Design

### Sandbox config

Add a config similar in shape to the MCP backend config:

```json
{
  "server": {
    "url": "http://127.0.0.1:18890",
    "port": 18890,
    "session_ttl": 300
  },
  "resources": {
    "code": {
      "enabled": true,
      "description": "Lightweight coding backend powered by claude-code-py tools",
      "backend_class": "sandbox.server.backends.resources.code.CodeBackend",
      "config": {
        "claude_code_root": "/home/a1/sdb/dxd/claude-code-py",
        "workspace_root": "/tmp/agentflow_code",
        "allow_bash": true
      }
    }
  },
  "warmup": {
    "enabled": false,
    "resources": []
  }
}
```

### Session init config

Use existing AgentFlow rollout plumbing:

- `resource_types=["code"]`
- `resource_init_configs["code"]["content"]`

with content such as:

```json
{
  "source_dir": "/path/to/repo"
}
```

## Rollout Integration

No new rollout abstractions are required.

The current rollout flow already supports resource backends and prompt-visible tool filtering.

Recommended usage:

- `resource_types=["code"]`
- `available_tools=["code-*"]`
- `sandbox_config_path="configs/sandbox-server/code_config.json"`

This keeps rollout changes minimal and consistent with MCP usage.

## Error Handling

### Backend-level errors

Use AgentFlow backend error responses for:

- failed dynamic import of `claude-code-py`
- invalid or missing `claude_code_root`
- invalid or missing `source_dir`
- path escape attempts
- `allow_bash=false`
- bash timeout

### Tool-level errors

Preserve the reused tool behavior for file-operation errors such as:

- file not found
- non-unique `old_string`
- missing `old_string`

These should pass through as normal tool outputs wrapped by AgentFlow's standard backend response builder.

## Testing Strategy

### 1. Schema tests

Add tests similar to the MCP schema tests:

- code tools appear in the global catalog
- `get_tool_schemas(["code-*"])` filters correctly
- single-tool filtering works

### 2. Backend tests

Add focused backend tests similar in spirit to `test_mcp_backend.py`:

- `bind_server()` registers all six tools
- `initialize()` creates a worker workspace
- `initialize()` copies `source_dir` into the workspace
- path escape attempts are rejected
- `allow_bash=false` blocks `code:bash`
- bridge tools dispatch successfully to reused tool implementations

### 3. Local smoke tests

Use a temporary directory as a mini repo and validate:

- `read -> edit -> read`
- `write -> glob -> grep`
- `bash` sees the session workspace as its current directory

### 4. Compatibility contract tests

Add a thin test that verifies the expected six tool classes can still be loaded from `claude-code-py`.

This catches upstream renames or import-signature drift early.

## Implementation Order

1. Add static `code` tool schemas
2. Add `CodeBackend` with workspace init and `source_dir` copy
3. Wire dynamic loading of the six `claude-code-py` tools
4. Add file-tool bridge wrappers with path-boundary checks
5. Add `code:bash` with `allow_bash` and timeout handling
6. Add backend and schema tests
7. Add a sandbox config example

## Open Risks

- `claude-code-py` is not packaged as an installable library, so import logic must be robust and explicit.
- `bash` remains only weakly constrained in v1.
- Tool behavior drift in `claude-code-py` could affect AgentFlow unless guarded by compatibility tests.

## Final Recommendation

Adopt a thin `code` backend that treats `claude-code-py` as the source of truth for lightweight coding tool behavior, while AgentFlow adds only the missing environment layer:

- session lifecycle
- workspace initialization
- path-boundary enforcement
- rollout integration

This gives AgentFlow a native coding environment with minimal architectural disruption.
