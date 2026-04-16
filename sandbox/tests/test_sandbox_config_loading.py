import json

from sandbox.sandbox import Sandbox, SandboxConfig


def test_load_server_config_keeps_workspace_root_for_code_backend(tmp_path):
    config_path = tmp_path / "code_config.json"
    raw_config = {
        "resources": {
            "code": {
                "enabled": True,
                "config": {
                    "workspace_root": "/tmp/agentflow_code"
                },
            }
        }
    }
    config_path.write_text(json.dumps(raw_config), encoding="utf-8")

    sandbox = Sandbox(config=SandboxConfig(server_config_path=str(config_path)))

    loaded = sandbox._load_server_config()

    assert loaded["resources"]["code"]["config"]["workspace_root"] == "/tmp/agentflow_code"
