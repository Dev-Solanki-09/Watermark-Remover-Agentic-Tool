"""Unit test validating llms.txt and llms-full.txt specification compliance."""

from pathlib import Path


def test_llms_txt_structure():
    p = Path("llms.txt")
    assert p.exists(), "llms.txt must exist"
    text = p.read_text(encoding="utf-8")

    # Specification checks
    assert text.startswith("# "), "llms.txt must begin with H1"
    assert "\n> " in text, "llms.txt must contain a blockquote summary"
    assert "## Core Capabilities" in text
    assert "## Documentation" in text
    assert "## MCP Tools" in text
    assert "## CLI Usage" in text

    # Mandatory constraint: zero occurrences of guarantee / guarantees
    lower = text.lower()
    assert "guarantee" not in lower, "llms.txt must not contain 'guarantee'"


def test_llms_full_txt_structure():
    p = Path("llms-full.txt")
    assert p.exists(), "llms-full.txt must exist"
    text = p.read_text(encoding="utf-8")

    assert text.startswith("# "), "llms-full.txt must begin with H1"
    assert "\n> " in text, "llms-full.txt must contain a blockquote summary"
    assert "Tier 0" in text
    assert "Tier 1" in text
    assert "Tier 2" in text
    assert "Tier 3" in text
    assert "Betti" in text
    assert "remove_watermark_auto" in text

    # Mandatory constraint: zero occurrences of guarantee / guarantees
    lower = text.lower()
    assert "guarantee" not in lower, "llms-full.txt must not contain 'guarantee'"
