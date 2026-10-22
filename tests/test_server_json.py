"""Unit test validating server.json schema and completeness."""

import json
from pathlib import Path


def test_server_json_schema():
    server_path = Path("server.json")
    assert server_path.exists(), "server.json must exist in repository root"

    data = json.loads(server_path.read_text(encoding="utf-8"))

    # Required MCP schema fields
    assert data["$schema"] == "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json"
    assert data["name"] == "io.github.dev-solanki-09/watermark-remover-mcp"
    assert "description" in data
    assert "version" in data
    assert "repository" in data
    assert data["repository"]["type"] == "git"
    assert "packages" in data
    assert any(pkg.get("packageName") == "wmclean" for pkg in data["packages"])
    assert "transports" in data
    transport_types = {t["type"] for t in data["transports"]}
    assert "stdio" in transport_types
    assert "streamable-http" in transport_types

    # Capabilities
    assert "capabilities" in data
    assert "tools" in data["capabilities"]
    tool_names = {t["name"] for t in data["capabilities"]["tools"]}
    assert "remove_watermark_auto" in tool_names
    assert "inspect_watermark_layers" in tool_names
    assert "benchmark_page" in tool_names
