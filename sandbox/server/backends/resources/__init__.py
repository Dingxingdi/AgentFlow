# sandbox/server/backends/resources/__init__.py
"""
Stateful resource backend exports.

Backends are loaded lazily so importing one backend module does not force
unrelated heavyweight dependencies for the others.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "VMBackend": (".vm", "VMBackend"),
    "create_vm_backend": (".vm", "create_vm_backend"),
    "RAGBackend": (".rag", "RAGBackend"),
    "create_rag_backend": (".rag", "create_rag_backend"),
    "MCPBackend": (".mcp", "MCPBackend"),
    "CodeBackend": (".code", "CodeBackend"),
}

__all__ = [
    "VMBackend",
    "RAGBackend",
    "MCPBackend",
    "CodeBackend",
    "create_vm_backend",
    "create_rag_backend",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
